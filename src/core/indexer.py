from pathlib import Path
import time
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
        try:
            prompt = f"Summarize:\n\n{text}"
            response = self.agent.run(prompt)
            return str(response.content)
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return "Error generating summary."

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
        print(f"DEBUG: Starting ingest_file with target_chunk_tokens={target_chunk_tokens}, group_size={group_size}")
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            full_text = f.read()

        print(f"Ingesting {file_path} with Smart Mode...")
        
        # 1. Smart Chunking
        ingestor = SmartIngestor()
        
        chunks_data = []
        current_idx = 0
        
        while current_idx < len(full_text):
            lookahead_chars = (target_chunk_tokens) * 4  # Approximate char count with buffer
            end_candidate = min(current_idx + lookahead_chars, len(full_text))
            
            # The segment we are analyzing (from current position forward)
            segment = full_text[current_idx:end_candidate]
            
            if end_candidate == len(full_text):
                chunks_data.append({
                    "text": segment,
                    "start_index": current_idx,
                    "end_index": len(full_text),
                    "reasoning": "End of file."
                })
                break
            
            # SmartIngestor returns indices relative to the start of 'segment'
            decision = ingestor.find_cut_point(segment)
            
            relative_cut_index = decision['cut_index']
            relative_next_start_index = decision['next_chunk_start_index']
            
            print(f"DEBUG: SmartIngest Returned Relative -> Cut: {relative_cut_index}, NextStart: {relative_next_start_index}")

            # Convert to absolute indices in full_text
            # Note: We ADD because these are indices relative to the start of the segment (current_idx)
            real_cut_index = current_idx + relative_cut_index
            real_next_start = current_idx + relative_next_start_index
            
            print(f"DEBUG: Absolute Indices -> Cut: {real_cut_index}, NextStart: {real_next_start}")
            print(f"DEBUG: Calculated Overlap: {real_cut_index - real_next_start} chars")

            # Overlap happens if real_next_start < real_cut_index
            
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
            
            # --- IMMEDIATE STORAGE & SUMMARIZATION ---
            
            # 1. Store Chunk
            chunk_id = storage.get_max_chunk_id() + 1
            storage.add_chunks([{
                "id": chunk_id,
                "text": chunk_text,
                "start_index": current_idx,
                "end_index": real_cut_index
            }])
            
            # 2. Store Level 1 Summary (Simultaneously generated)
            summary_text = decision.get("summary", "No summary available.")
            l1_id = storage.add_summaries([{
                "summary_text": summary_text,
                "level": 1,
                "chunk_ids": str(chunk_id)
            }])[0]
            
            # Keep track for next level
            class SimpleSummary: pass
            ss = SimpleSummary()
            ss.id = l1_id
            ss.summary_text = summary_text
            ss.level = 1
            
            # We add to a list that will be used for Level 2
            if 'level_1_summaries_accumulator' not in locals():
                level_1_summaries_accumulator = []
            level_1_summaries_accumulator.append(ss)
            
            current_idx = real_next_start
            
            if current_idx >= len(full_text):
                break
            if len(chunk_text) == 0:
                print("Zero length chunk detected. Force advancing.")
                current_idx += 100 
        
        print(f"Ingestion Pass Complete. Stored {len(level_1_summaries_accumulator)} chunks and Level 1 summaries.")

        # 3. Recursive Summarization
        current_level_summaries = level_1_summaries_accumulator
        level = 2
        while len(current_level_summaries) > 1:
            next_level_payloads = []
            
            for i in range(0, len(current_level_summaries), group_size):
                group = current_level_summaries[i:i+group_size]
                combined_text = "\n\n".join([s.summary_text for s in group])
                summary_text = self.summarize_text(combined_text)
                time.sleep(1) # Prevent rate limiting
                
                next_level_payloads.append({
                    "summary_text": summary_text,
                    "level": level,
                    # Parent IDs will be assigned after batch insertion
                })
            
            # Batch create parents
            parent_ids = storage.add_summaries(next_level_payloads)
            
            # Now update children with parent_ids
            
            # Create next gen objects for next iteration
            next_gen_objects = []
            
            summary_table = None # Not needed in SQLite version
            
            # We iterate again to match groups to parent IDs
            parent_idx = 0
            for i in range(0, len(current_level_summaries), group_size):
                group = current_level_summaries[i:i+group_size]
                p_id = parent_ids[parent_idx]
                
                # Update children
                for child in group:
                    # Execute Update: UPDATE summaries SET parent_id = p_id WHERE id = child.id
                    storage.update_summary_parent(child.id, p_id)
                
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