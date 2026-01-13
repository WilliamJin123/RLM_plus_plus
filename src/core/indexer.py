from pathlib import Path
import time
from typing import List

from src.core.factory import AgentFactory
from src.core.storage import StorageEngine
from src.core.smart_ingest import SmartIngestor
from src.utils.token_buffer import TokenBuffer

class Indexer:
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.storage = StorageEngine(self.db_path)
        self.summarizer = AgentFactory.create_agent("summarization-agent")
        self.smart_ingestor = SmartIngestor()

        self.token_buffer = TokenBuffer(model_name="gpt-4o")

    def _summarize_text(self, text: str) -> str:
        """Helper to summarize text using the configured agent."""
        resp = self.summarizer.run(f"Summarize the following list of document summaries into a single cohesive summary:\n\n{text}")
        return resp.content

    def ingest_file(self, file_path: str, max_chunk_tokens: int = 1000, group_size: int = 5) -> None:
        """
        Ingests a file into the database using smart chunking and recursive summarization.
        
        Args:
            file_path: Path to the file to be indexed.
            target_chunk_tokens: The approximate number of tokens desired per chunk. The SmartIngestor will find the best semantic cut point near this target.
            group_size: How many chunks (or lower-level summaries) to group together when creating the next level of the summary hierarchy. Defaults to 2 (i.e., binary tree).
        """
        print(f"--- Indexing {file_path} ---")
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            full_text = f.read()
        
        level_0_ids = self._process_chunks(full_text, max_chunk_tokens, str(path.name))
        self._build_hierarchy(level_0_ids, group_size=group_size)
        print("--- Indexing Complete ---")

    def _process_chunks(self, full_text: str, max_chunk_tokens: int, filename: str) -> List[int]:
        """
        Slices text, stores chunks, stores L0 summaries, and links them.
        Returns a list of the created L0 Summary IDs.
        """
        current_idx = 0
        summary_ids = []
        
        # We grab a 'safe' lookahead window of characters that is definitely larger 
        # than the token limit, then let TokenBuffer trim it down precisely.
        char_lookahead = max_chunk_tokens * 5 

        while current_idx < len(full_text):
            self.token_buffer.clear()

            # Define window
            raw_end = min(current_idx + char_lookahead, len(full_text))
            raw_segment = full_text[current_idx:raw_end]
            self.token_buffer.add_text(raw_segment)
            valid_window_text = self.token_buffer.get_chunk_at(max_chunk_tokens)
            
            # Analysis
            result = self.smart_ingestor.analyze_segment(valid_window_text)
            
            # Calculate Absolute Positions
            abs_cut = current_idx + result['cut_index']
            abs_next = current_idx + result['next_chunk_start_index']
            
            chunk_text = full_text[current_idx:abs_cut]
            
            # Store Chunk
            chunk_id = self.storage.add_chunk(
                text=chunk_text,
                start=current_idx,
                end=abs_cut,
                source=filename
            )
            
            # Store Level 0 Summary
            sum_id = self.storage.add_summary(
                text=result['summary'],
                level=0,
                parent_id=None # Will be filled later
            )
            
            # Link
            self.storage.link_summary_to_chunk(sum_id, chunk_id)
            summary_ids.append(sum_id)
            
            print(f"Created L0 Node {sum_id} -> Chunk {chunk_id} ({len(chunk_text)} chars)")
            
            # Move pointer
            if abs_next >= len(full_text):
                break
            current_idx = abs_next

        return summary_ids

    def _build_hierarchy(self, child_ids: List[int], group_size: int):
        """
        Recursively groups summaries and builds parents until root.
        """
        current_ids = child_ids
        current_level = 0
        
        while len(current_ids) > 1:
            print(f"Building Level {current_level + 1} from {len(current_ids)} nodes...")
            next_level_ids = []
            
            # Process in batches
            for i in range(0, len(current_ids), group_size):
                batch_ids = current_ids[i : i + group_size]
               
                batch_texts = self.storage.get_chunk_texts(batch_ids)
                combined_text = "\n\n".join(batch_texts)
                
                # Generate Higher Level Summary
                new_summary = self._summarize_text(combined_text)
                
                # Store Parent Node
                parent_id = self.storage.add_summary(
                    text=new_summary,
                    level=current_level + 1
                )
                next_level_ids.append(parent_id)
                
                # Link Children to Parent
                for child_id in batch_ids:
                    self.storage.update_summary_parent(child_id, parent_id)
                
                time.sleep(0.5) # Rate limit safety
            
            current_ids = next_level_ids
            current_level += 1