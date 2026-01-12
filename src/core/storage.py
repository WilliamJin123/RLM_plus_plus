import sqlite3
from typing import List, Optional
from pathlib import Path

class StorageEngine:
    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "rlm_storage.db")
        
        self.db_path = db_path
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Raw Text Chunks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    start_index INTEGER,
                    end_index INTEGER,
                    file_source TEXT
                )
            """)
            
            # 2. Summaries (The Tree Nodes)
            # level 0 = summary of chunks
            # level 1+ = summary of lower level summaries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_text TEXT,
                    level INTEGER,
                    parent_id INTEGER,
                    FOREIGN KEY(parent_id) REFERENCES summaries(id)
                )
            """)
            
            # 3. Linking Table (Many-to-Many: Summaries <-> Chunks)
            # Only used for Level 0 summaries linking to raw chunks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_chunks (
                    summary_id INTEGER,
                    chunk_id INTEGER,
                    PRIMARY KEY (summary_id, chunk_id),
                    FOREIGN KEY(summary_id) REFERENCES summaries(id),
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
                )
            """)
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def add_chunk(self, text: str, start: int, end: int, source: str = "") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chunks (text, start_index, end_index, file_source) VALUES (?, ?, ?, ?)",
                (text, start, end, source)
            )
            return cursor.lastrowid

    def add_summary(self, text: str, level: int, parent_id: Optional[int] = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO summaries (summary_text, level, parent_id) VALUES (?, ?, ?)",
                (text, level, parent_id)
            )
            return cursor.lastrowid

    def link_summary_to_chunk(self, summary_id: int, chunk_id: int):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO summary_chunks (summary_id, chunk_id) VALUES (?, ?)",
                (summary_id, chunk_id)
            )

    def update_summary_parent(self, summary_id: int, parent_id: int):
        with self._get_connection() as conn:
            conn.execute("UPDATE summaries SET parent_id = ? WHERE id = ?", (parent_id, summary_id))

    # --- Read Operations ---

    def get_document_structure(self) -> str:
        """Returns the top-level nodes of the tree."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Find the highest level
            cursor.execute("SELECT MAX(level) FROM summaries")
            res = cursor.fetchone()
            max_level = res[0] if res[0] is not None else -1

            if max_level == -1:
                return "Index is empty."

            cursor.execute("SELECT id, summary_text FROM summaries WHERE level = ?", (max_level,))
            roots = cursor.fetchall()
            
            output = f"Document Structure (Root Level: {max_level}):\n"
            for r in roots:
                output += f"- [Node {r[0]}] {r[1][:150]}...\n"
            return output

    def get_summary_children(self, summary_id: int) -> str:
        """
        Intelligently fetches children. 
        If Node is Level > 0: Returns child Summaries.
        If Node is Level 0: Returns linked raw Chunks.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Get current node details
            cursor.execute("SELECT level, summary_text FROM summaries WHERE id = ?", (summary_id,))
            current = cursor.fetchone()
            if not current:
                return f"Node {summary_id} not found."
            
            level, text = current
            
            output = f"Node {summary_id} (Level {level}):\n{text[:100]}...\n\nContains:\n"

            # 2. Logic based on level
            if level > 0:
                # Fetch child summaries
                cursor.execute("SELECT id, summary_text FROM summaries WHERE parent_id = ?", (summary_id,))
                children = cursor.fetchall()
                if not children:
                    return output + "No child summaries found."
                for c in children:
                    output += f"- [Node {c[0]}] {c[1][:150]}...\n"
            
            else:
                # Fetch linked raw chunks via junction table
                query = """
                    SELECT c.id, c.text 
                    FROM chunks c
                    JOIN summary_chunks sc ON c.id = sc.chunk_id
                    WHERE sc.summary_id = ?
                """
                cursor.execute(query, (summary_id,))
                chunks = cursor.fetchall()
                if not chunks:
                    return output + "No linked raw chunks found."
                for c in chunks:
                    output += f"- [Chunk {c[0]}] {c[1][:150]}...\n"

            return output

    def search_summaries(self, query: str) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, level, summary_text FROM summaries WHERE summary_text LIKE ? LIMIT 10", 
                (f"%{query}%",)
            )
            matches = cursor.fetchall()
            
            if not matches:
                return "No matches found."
                
            output = f"Search Results for '{query}':\n"
            for m in matches:
                output += f"- [Node {m[0]} L{m[1]}] {m[2][:150]}...\n"
            return output
    
    def get_chunk_text(self, chunk_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
            res = cursor.fetchone()
            return res[0] if res else None

    def get_chunk_texts(self, chunk_ids: List[int]) -> List[Optional[str]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(chunk_ids))
            cursor.execute(f"SELECT text FROM chunks WHERE id IN ({placeholders})", chunk_ids)
            return [r[0] for r in cursor.fetchall()]