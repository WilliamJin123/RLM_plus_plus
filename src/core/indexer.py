from pathlib import Path
from agno.agent import Agent

from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.core.get_model import get_model
from src.core.db import SessionLocal, Chunk, Summary, init_db
from src.core.smart_ingest import SmartIngestor

class Indexer:
    def __init__(self, db_path: str = None):
        init_db(db_path)
        self.model = get_model()
        # Summarization agent's own DB (different from RLM content DB)
        agent_db_path = Path(__file__).resolve().parent.parent / "data" / "sumarization_agent.db"
        agent_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent = Agent(
            id="summarization-agent",
            name="Summarization Agent",
            model=self.model, 
            description="You are an expert summarizer.",
            instructions="Summarize texts concisely, capturing key facts and entities. They must encapsulate the main points clearly, and give enough context for a person to realize when they should read the full text.",
            db=SqliteDb(
                db_file=agent_db_path.as_posix(),
                session_table="summarization_agent_sessions",
            )
        )
        setup_tracing(db=self.agent.db, batch_processing=True)

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
            real_next_start = current_idx + decision['next_chunk_start_index']
            
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
        
        session = SessionLocal()
        db_chunks = []
        for c in chunks_data:
            chunk = Chunk(text=c['text'], start_index=c['start_index'], end_index=c['end_index'])
            session.add(chunk)
            db_chunks.append(chunk)
        
        session.commit()
        print(f"Stored {len(db_chunks)} chunks.")

        # 2. Level 0 Summarization
        current_level_summaries = []
        for i in range(0, len(db_chunks), group_size):
            group = db_chunks[i:i+group_size]
            combined_text = "\n\n".join([c.text for c in group])
            summary_text = self.summarize_text(combined_text)
            
            chunk_ids = ",".join([str(c.id) for c in group])
            summary = Summary(summary_text=summary_text, level=0, chunk_ids=chunk_ids)
            session.add(summary)
            current_level_summaries.append(summary)
        
        session.commit()
        print(f"Created {len(current_level_summaries)} Level 0 summaries.")

        # 3. Recursive Summarization
        level = 1
        while len(current_level_summaries) > 1:
            next_level_summaries = []
            for i in range(0, len(current_level_summaries), group_size):
                group = current_level_summaries[i:i+group_size]
                combined_text = "\n\n".join([s.summary_text for s in group])
                summary_text = self.summarize_text(combined_text)
                
                new_summary = Summary(summary_text=summary_text, level=level)
                session.add(new_summary)
                session.flush() 
                
                for s in group:
                    s.parent_id = new_summary.id
                
                next_level_summaries.append(new_summary)
            
            session.commit()
            print(f"Created {len(next_level_summaries)} Level {level} summaries.")
            current_level_summaries = next_level_summaries
            level += 1

        session.close()
        print("Indexing complete.")