import sqlite3
from typing import List, Optional
from pathlib import Path

class StorageEngine:
    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            # Switch to .db file
            db_path = str(project_root / "data" / "rlm_storage.db")
        
        self.db_path = db_path
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def reset(self, db_path: str):
        """
        Re-connects to a different database path. 
        """
        self.db_path = db_path
        if self.db_path:
             Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT,
                start_index INTEGER,
                end_index INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY,
                summary_text TEXT,
                level INTEGER,
                parent_id INTEGER,
                chunk_ids TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def get_chunk_count(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_max_chunk_id(self) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(id) FROM chunks")
        max_id = cursor.fetchone()[0]
        conn.close()
        return max_id if max_id is not None else 0

    def add_chunks(self, chunks: List[dict]):
        """
        Bulk add chunks.
        chunks: List of dicts with keys: text, start_index, end_index, (optional) id.
        """
        if not chunks:
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for c in chunks:
            if "id" in c:
                cursor.execute(
                    "INSERT OR REPLACE INTO chunks (id, text, start_index, end_index) VALUES (?, ?, ?, ?)",
                    (c["id"], c["text"], c["start_index"], c["end_index"])
                )
            else:
                 cursor.execute(
                    "INSERT INTO chunks (text, start_index, end_index) VALUES (?, ?, ?)",
                    (c["text"], c["start_index"], c["end_index"])
                )
        
        conn.commit()
        conn.close()

    def add_summaries(self, summaries: List[dict]) -> List[int]:
        """
        Bulk add summaries. Returns list of assigned IDs.
        """
        if not summaries:
            return []
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        ids = []
        for s in summaries:
            s_id = s.get("id")
            parent_id = s.get("parent_id")
            chunk_ids = s.get("chunk_ids")
            
            if s_id:
                cursor.execute(
                    "INSERT OR REPLACE INTO summaries (id, summary_text, level, parent_id, chunk_ids) VALUES (?, ?, ?, ?, ?)",
                    (s_id, s["summary_text"], s["level"], parent_id, chunk_ids)
                )
                ids.append(s_id)
            else:
                cursor.execute(
                    "INSERT INTO summaries (summary_text, level, parent_id, chunk_ids) VALUES (?, ?, ?, ?)",
                    (s["summary_text"], s["level"], parent_id, chunk_ids)
                )
                ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        return ids

    def update_summary_parent(self, summary_id: int, parent_id: int):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE summaries SET parent_id = ? WHERE id = ?", (parent_id, summary_id))
        conn.commit()
        conn.close()

    def get_document_structure(self) -> str:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(level) FROM summaries")
        highest_level = cursor.fetchone()[0]
        
        if highest_level is None:
             conn.close()
             return "Document index is empty."
             
        cursor.execute("SELECT id, summary_text FROM summaries WHERE level = ?", (highest_level,))
        roots = cursor.fetchall()
        
        structure = f"Document Structure (Highest Level: {highest_level}):\n"
        for r in roots:
            structure += f"- [ID: {r[0]}] {r[1][:200]}...\n"
            
        conn.close()
        return structure

    def get_summary_children(self, summary_id: int) -> str:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT level, chunk_ids FROM summaries WHERE id = ?", (summary_id,))
        res = cursor.fetchone()
        
        if not res:
            conn.close()
            return f"Summary with ID {summary_id} not found."
            
        level, chunk_ids_str = res
        
        cursor.execute("SELECT id, summary_text FROM summaries WHERE parent_id = ?", (summary_id,))
        children = cursor.fetchall()
        
        result = f"Children of Summary {summary_id} (Level {level}):\n"
        
        if children:
            for c in children:
                result += f"- [Summary ID: {c[0]}] {c[1][:200]}...\n"
        elif level == 0:
             if chunk_ids_str:
                 clean_ids = chunk_ids_str.replace("[", "").replace("]", "")
                 if clean_ids:
                     # Basic input sanitization to ensure only numbers and commas
                     if not all(c.isdigit() or c == ',' or c.isspace() for c in clean_ids):
                         result += "Invalid chunk IDs format."
                     else:
                         query = f"SELECT id, text FROM chunks WHERE id IN ({clean_ids})"
                         cursor.execute(query)
                         chunks = cursor.fetchall()
                         
                         result += f"This is a leaf summary covering chunks: {clean_ids}\n"
                         for ch in chunks:
                             result += f"- [Chunk ID: {ch[0]}] {ch[1][:100]}...\n"
                 else:
                     result += "No chunk IDs found in summary."
             else:
                 result += "No chunk IDs linked."
        
        conn.close()
        return result

    def get_chunk_text(self, chunk_id: int) -> str:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
        res = cursor.fetchone()
        conn.close()
        if res:
            return res[0]
        return None

    def search_summaries(self, query: str) -> str:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        sql_query = "SELECT id, level, summary_text FROM summaries WHERE summary_text LIKE ?"
        cursor.execute(sql_query, (f"%{query}%",))
        matches = cursor.fetchmany(10)
        
        if not matches:
            conn.close()
            return f"No summaries found matching '{query}'."
            
        output = f"Search results for '{query}':\n"
        for r in matches:
            output += f"- [Summary ID: {r[0]}, Level: {r[1]}] {r[2][:200]}...\n"
            
        conn.close()
        return output

# Global instance
storage = StorageEngine()