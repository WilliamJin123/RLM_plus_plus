from pathlib import Path
from agno.agent import Agent

from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.core import get_model
from src.core.db import SessionLocal, Chunk, Summary, init_db
from src.utils.ingest import sliding_window_chunker

class Indexer:
    def __init__(self):
        init_db()
        self.model = get_model()
        db_path = Path(__file__).resolve().parent.parent / "data" / "sumarization_agent.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent = Agent(
            id="summarization-agent",
            name="Summarization Agent",
            model=self.model, 
            description="You are an expert summarizer.",
            instructions="Summarize texts concisely, capturing key facts and entities. They must encapsulate the main points clearly, and give enough context for a person to realize when they should read the full text.",
            db=SqliteDb(
                db_file=db_path.as_posix(),
                session_table="summarization_agent_sessions",
            )
        )
        setup_tracing(db=self.agent.db, batch_processing=True)

    def summarize_text(self, text: str) -> str:
        prompt = f"Summarize:\n\n{text}"
        response = self.agent.run(prompt)
        return response.content

    def ingest_file(self, file_path: str, target_chunk_tokens=1000, group_size=5):
        from src.core.smart_ingest import SmartIngestor
        
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