import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field
from typing import List, Optional
from pathlib import Path
from src.config.config import config

# Define Schema using Pydantic (LanceDB style)
class ChunkModel(LanceModel):
    id: int
    text: str
    start_index: int
    end_index: int
    # Vector support ready for future (optional for now)
    # vector: Vector(1536) = None 

class SummaryModel(LanceModel):
    id: int
    summary_text: str
    level: int
    parent_id: Optional[int] = None
    chunk_ids: Optional[str] = None # Comma separated for now

class StorageEngine:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use a 'lancedb' folder alongside the old sqlite one
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "lancedb_store")
        
        self.db = lancedb.connect(db_path)
        
        # Initialize tables if they don't exist
        self._init_tables()

    def reset(self, db_path: str):
        """
        Re-connects to a different database path. 
        Useful for benchmarks or isolated tests.
        """
        if db_path:
            self.db = lancedb.connect(db_path)
            self._init_tables()

    def _init_tables(self):
        # Create tables if not exist. 
        # Note: LanceDB 'create_table' with schema creates an empty table.
        try:
            self.db.create_table("chunks", schema=ChunkModel, exist_ok=True)
            self.db.create_table("summaries", schema=SummaryModel, exist_ok=True)
        except Exception as e:
            print(f"Error initializing tables: {e}")

    def add_chunks(self, chunks: List[dict]):
        """
        Bulk add chunks.
        chunks: List of dicts matching ChunkModel fields.
        """
        if not chunks:
            return
            
        # Get current max ID to auto-increment
        table = self.db.open_table("chunks")
        # Optimization: fast count
        current_count = len(table)
        
        # Assign IDs
        processed = []
        for i, c in enumerate(chunks):
            # If ID not provided, assign one
            if "id" not in c:
                c["id"] = current_count + i + 1
            processed.append(ChunkModel(**c))
            
        table.add(processed)

    def add_summaries(self, summaries: List[dict]) -> List[int]:
        """
        Bulk add summaries. Returns list of assigned IDs.
        """
        if not summaries:
            return []
            
        table = self.db.open_table("summaries")
        current_count = len(table)
        
        processed = []
        ids = []
        for i, s in enumerate(summaries):
            if "id" not in s:
                s_id = current_count + i + 1
                s["id"] = s_id
            else:
                s_id = s["id"]
                
            processed.append(SummaryModel(**s))
            ids.append(s_id)
            
        table.add(processed)
        return ids

    def get_document_structure(self) -> str:
        summary_table = self.db.open_table("summaries")
        df = summary_table.to_pandas()
        
        if df.empty:
            return "Document index is empty."
            
        highest_level = df["level"].max()
        roots = df[df["level"] == highest_level]
        
        structure = f"Document Structure (Highest Level: {highest_level}):\n"
        for _, r in roots.iterrows():
            structure += f"- [ID: {r['id']}] {r['summary_text'][:200]}...\n"
            
        return structure

    def get_summary_children(self, summary_id: int) -> str:
        summary_table = self.db.open_table("summaries")
        
        # LanceDB SQL filtering (via DuckDB under hood usually, or pandas)
        # For simple ID lookup, we can use filtering
        s_df = summary_table.search().where(f"id = {summary_id}").limit(1).to_pandas()
        
        if s_df.empty:
            return f"Summary with ID {summary_id} not found."
            
        summary = s_df.iloc[0]
        
        # Find children
        children_df = summary_table.search().where(f"parent_id = {summary_id}").to_pandas()
        
        result = f"Children of Summary {summary_id} (Level {summary['level']}):\n"
        
        if not children_df.empty:
            for _, c in children_df.iterrows():
                result += f"- [Summary ID: {c['id']}] {c['summary_text'][:200]}...\n"
        elif summary['level'] == 0:
            # Parse chunk IDs
            chunk_ids_str = summary['chunk_ids']
            if chunk_ids_str:
                chunk_table = self.db.open_table("chunks")
                # Using 'IN' clause for filtering
                # LanceDB SQL support: "id IN (1, 2, 3)"
                
                # Sanitize
                clean_ids = chunk_ids_str.replace("[", "").replace("]", "")
                if clean_ids:
                    chunks_df = chunk_table.search().where(f"id IN ({clean_ids})").to_pandas()
                    
                    result += f"This is a leaf summary covering chunks: {clean_ids}\n"
                    for _, ch in chunks_df.iterrows():
                        result += f"- [Chunk ID: {ch['id']}] {ch['text'][:100]}...\n"
                else:
                    result += "No chunk IDs found in summary."
            else:
                result += "No chunk IDs linked."
                
        return result

    def get_chunk_text(self, chunk_id: int) -> str:
        chunk_table = self.db.open_table("chunks")
        df = chunk_table.search().where(f"id = {chunk_id}").limit(1).to_pandas()
        if df.empty:
            return None
        return df.iloc[0]['text']

    def search_summaries(self, query: str) -> str:
        # Full text search? 
        # LanceDB has FTS via .search(query) if index exists, or we can just scan for now.
        # Since we haven't built an FTS index, we'll do a simple contains in pandas for MVP parity
        # (LanceDB native FTS requires creating an index first).
        
        summary_table = self.db.open_table("summaries")
        
        # Efficient iteration or pandas filter
        # For "faster than DB", let's use the Lance filtering if possible, 
        # but Lance SQL 'LIKE' might not be fully available in basic mode without FTS.
        # Fallback to pandas for flexibility in this prototype.
        df = summary_table.to_pandas()
        
        # Simple case-insensitive search
        matches = df[df['summary_text'].str.contains(query, case=False, na=False)].head(10)
        
        if matches.empty:
            return f"No summaries found matching '{query}'."
        
        output = f"Search results for '{query}':\n"
        for _, r in matches.iterrows():
            output += f"- [Summary ID: {r['id']}, Level: {r['level']}] {r['summary_text'][:200]}...\n"
            
        return output

# Global instance for simplicity, similar to SessionLocal pattern but cleaner
storage = StorageEngine()
