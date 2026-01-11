from pathlib import Path
from agno.agent import Agent

from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.core.factory import AgentFactory
from src.core.storage import storage
from src.core.smart_ingest import SmartIngestor

class Indexer:
    def __init__(self, db_path: str = None):
        # Update global storage if a path is provided
        if db_path:
            storage.reset(db_path)
            
        # Summarization agent is now configured in agents.yaml
        self.agent = AgentFactory.create_agent("summarization-agent")

    def summarize_text(self, text: str) -> str:
        prompt = f"Summarize:\n\n{text}"
        response = self.agent.run(prompt)
        return str(response.content)

    def ingest_file(self, file_path: str, target_chunk_tokens: int = 1000, group_size: int = 2) -> None:
        """
        Ingests a file into the database using smart chunking and recursive summarization.
        
        Args:
            file_path: Path to the file to be indexed.
            target_chunk_tokens: The approximate number of tokens desired per chunk. 
                               The SmartIngestor will find the best semantic cut point near this target.
            group_size: How many chunks (or lower-level summaries) to group together when 
                        creating the next level of the summary hierarchy. Defaults to 2 (i.e., binary tree).
        """
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            full_text = f.read()

        print(f"Ingesting {file_path} with Smart Mode...")
        
        # 1. Smart Chunking
        ingestor = SmartIngestor()
        
        chunks_data = []
        current_idx = 0
        
        while current_idx < len(full_text):
            lookahead_chars = target_chunk_tokens * 5 
            end_candidate = min(current_idx + lookahead_chars, len(full_text))
            
            segment = full_text[current_idx:end_candidate]
            
            if end_candidate == len(full_text):
                chunks_data.append({
                    "text": segment,
                    "start_index": current_idx,
                    "end_index": len(full_text),
                    "reasoning": "End of file."
                })
                break
                
            decision = ingestor.find_cut_point(segment)
            
            real_cut_index = current_idx + decision['cut_index']
            real_next_start = current_idx - decision['next_chunk_start_index']
            
            # Ensure continuity: next chunk must start at or before current chunk ends
            if real_next_start > real_cut_index:
                print(f"Warning: Gap detected (Next: {real_next_start}, Cut: {real_cut_index}). Clamping next start to cut index to prevent data loss.")
                real_next_start = real_cut_index
            
            chunk_text = full_text[current_idx:real_cut_index]
            
            chunks_data.append({
                "text": chunk_text,
                "start_index": current_idx,
                "end_index": real_cut_index,
                "reasoning": decision['reasoning']
            })
            
            print(f"Chunk created: {len(chunk_text)} chars. Reason: {decision['reasoning']}")
            
            current_idx = real_next_start
            
            if current_idx >= len(full_text):
                break
            if len(chunk_text) == 0:
                print("Zero length chunk detected. Force advancing.")
                current_idx += 100 
        
        # Batch write chunks to LanceDB
        # We need to assign IDs or let storage handle it. 
        # But we need the IDs for the next step. 
        # storage.add_chunks handles ID generation if not provided, 
        # but to get them back effectively we might want to pre-assign or query back.
        # The storage implementation assigns ID = count + index. 
        # Let's rely on that logic, but to be safe/explicit, let's fetch current count first?
        # Actually, let's just create the dicts and pass them.
        
        # To get IDs back, we can just query the last N items or assume sequentiality.
        # Or better, refactor add_chunks to return IDs? 
        # The storage class I wrote didn't return IDs for chunks. 
        # Let's update indexer to just re-read or trust the sequential nature.
        # Actually, simplest is to just assume standard auto-inc behavior since we are single threaded here.
        
        # Let's verify storage.add_chunks behavior in previous step... 
        # It assigns: c["id"] = current_count + i + 1.
        
        # So we can calculate IDs here locally.
        chunk_table = storage.db.open_table("chunks")
        start_id = len(chunk_table) + 1
        
        chunks_payload = []
        db_chunks_simulated = []
        
        for i, c in enumerate(chunks_data):
            c_id = start_id + i
            payload = {
                "id": c_id,
                "text": c['text'],
                "start_index": c['start_index'],
                "end_index": c['end_index']
            }
            chunks_payload.append(payload)
            
            # Create a simple object to mimic the old 'Chunk' object for the next loop
            # capable of accessing .text and .id
            class SimpleChunk:
                pass
            sc = SimpleChunk()
            sc.id = c_id
            sc.text = c['text']
            db_chunks_simulated.append(sc)
            
        storage.add_chunks(chunks_payload)
        print(f"Stored {len(db_chunks_simulated)} chunks (LanceDB).")

        # 2. Level 0 Summarization
        current_level_summaries = []
        # Use db_chunks_simulated instead of db_chunks
        db_chunks = db_chunks_simulated 
        
        for i in range(0, len(db_chunks), group_size):
            group = db_chunks[i:i+group_size]
            combined_text = "\n\n".join([c.text for c in group])
            summary_text = self.summarize_text(combined_text)
            
            chunk_ids = ",".join([str(c.id) for c in group])
            
            # We collect them to batch add? Or add one by one?
            # storage.add_summaries is batched.
            # Let's collect them.
            current_level_summaries.append({
                "summary_text": summary_text,
                "level": 0,
                "chunk_ids": chunk_ids
            })
            
        # Write Level 0
        l0_ids = storage.add_summaries(current_level_summaries)
        
        # Convert to objects for next level
        next_gen_objects = []
        for idx, s_data in enumerate(current_level_summaries):
            class SimpleSummary:
                pass
            ss = SimpleSummary()
            ss.id = l0_ids[idx]
            ss.summary_text = s_data["summary_text"]
            ss.level = 0
            next_gen_objects.append(ss)
            
        current_level_summaries = next_gen_objects # Swap variable for the loop
        
        print(f"Created {len(current_level_summaries)} Level 0 summaries.")

        # 3. Recursive Summarization
        level = 1
        while len(current_level_summaries) > 1:
            next_level_payloads = []
            
            for i in range(0, len(current_level_summaries), group_size):
                group = current_level_summaries[i:i+group_size]
                combined_text = "\n\n".join([s.summary_text for s in group])
                summary_text = self.summarize_text(combined_text)
                
                next_level_payloads.append({
                    "summary_text": summary_text,
                    "level": level,
                    # We need to link children to this new parent.
                    # But in the old code: s.parent_id = new_summary.id
                    # Here we can't update children easily without a re-write/update query.
                    # LanceDB supports updates. 
                    # But we don't have the parent ID yet.
                    # Strategy: Create parents first, get IDs, then update children.
                })
            
            # Batch create parents
            parent_ids = storage.add_summaries(next_level_payloads)
            
            # Now update children with parent_ids
            # We need to map group -> parent_id
            
            # Create next gen objects for next iteration
            next_gen_objects = []
            
            summary_table = storage.db.open_table("summaries")
            
            # We iterate again to match groups to parent IDs
            parent_idx = 0
            for i in range(0, len(current_level_summaries), group_size):
                group = current_level_summaries[i:i+group_size]
                p_id = parent_ids[parent_idx]
                
                # Update children
                for child in group:
                    # Execute Update: UPDATE summaries SET parent_id = p_id WHERE id = child.id
                    # LanceDB update syntax:
                    summary_table.update(where=f"id = {child.id}", values={"parent_id": p_id})
                
                # Prepare object for next level
                class SimpleSummary: pass
                ss = SimpleSummary()
                ss.id = p_id
                ss.summary_text = next_level_payloads[parent_idx]["summary_text"]
                ss.level = level
                next_gen_objects.append(ss)
                
                parent_idx += 1
            
            print(f"Created {len(next_gen_objects)} Level {level} summaries.")
            current_level_summaries = next_gen_objects
            level += 1

        print("Indexing complete.")