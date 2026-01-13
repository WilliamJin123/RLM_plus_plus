import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

class StorageEngine:
    def __init__(self, db_path: str = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        
        if db_path is None:
            self.db_path = str(project_root / "data" / "rlm_storage.db")
        else:
            # If a path is provided, ensure it's handled correctly relative to root if it's not absolute
            # However, for this project's convention, we anchor it to project_root if it looks like a filename
            # or keep it if the user provided a full path.
            # Simple approach matching previous intent: force anchor to project root for consistency
            self.db_path = str(project_root / db_path)
            
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

    def get_root_summaries(self) -> List[Tuple[int, str]]:
        """Returns list of (id, text) for the highest level nodes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(level) FROM summaries")
            res = cursor.fetchone()
            max_level = res[0] if res[0] is not None else -1
            
            if max_level == -1:
                return []
            
            cursor.execute("SELECT id, summary_text FROM summaries WHERE level = ?", (max_level,))
            return cursor.fetchall()
        
    def get_node_metadata(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """
        Lightweight lookup to check a node's level before deciding how to handle it.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, level, summary_text FROM summaries WHERE id = ?", (summary_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {"id": row[0], "level": row[1], "text": row[2]}
    
    def get_child_summaries(self, parent_id: int) -> List[Tuple[int, str]]:
        """Returns child summaries (id, text) for navigation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, summary_text FROM summaries WHERE parent_id = ?", (parent_id,))
            return cursor.fetchall()

    def get_linked_chunk_id(self, summary_id: int) -> Optional[int]:
        """
        Returns the raw chunk ID associated with a leaf summary.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id FROM summary_chunks WHERE summary_id = ? LIMIT 1", (summary_id,))
            res = cursor.fetchone()
            return res[0] if res else None
    
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
        
    def search_summaries(self, query: str) -> List[Tuple[int, int, str]]:
        """Returns (id, level, text) matches."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, level, summary_text FROM summaries WHERE summary_text LIKE ? LIMIT 10", 
                (f"%{query}%",)
            )
            return cursor.fetchall()