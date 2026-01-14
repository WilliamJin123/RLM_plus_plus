import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def clean_summary_text(text: str) -> str:
    """
    Clean common artifacts from summary text.
    - Removes <think>...</think> blocks
    - Removes ```markdown prefix/suffix
    """
    if not text:
        return text

    # Remove think blocks (handles multi-line)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    text = re.sub(r'```(?:\w+)?\n?', '', text)               
    text = text.replace("###", "")                           
    text = text.strip()            

    return text.strip()


class StorageEngine:
    def __init__(self, db_path: Optional[str] = None):
        project_root = Path(__file__).resolve().parent.parent.parent

        if db_path is None:
            self.db_path = str(project_root / "data" / "rlm_storage.db")
        elif Path(db_path).is_absolute():
            self.db_path = db_path
        else:
            self.db_path = str(project_root / db_path)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    start_index INTEGER,
                    end_index INTEGER,
                    file_source TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_text TEXT,
                    level INTEGER,
                    parent_id INTEGER,
                    sequence_index INTEGER,
                    chunk_id INTEGER,
                    FOREIGN KEY(parent_id) REFERENCES summaries(id),
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id)
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_seq ON summaries(parent_id, sequence_index)"
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_level ON summaries(level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON summaries(chunk_id)")
            conn.commit()

        # Migrate existing DBs that have old schema
        self._migrate_schema_if_needed()

    def _migrate_schema_if_needed(self) -> None:
        """Auto-migrate from summary_chunks junction table to direct chunk_id column."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if summaries table has chunk_id column
            cursor.execute("PRAGMA table_info(summaries)")
            columns = [row[1] for row in cursor.fetchall()]

            if "chunk_id" not in columns:
                # Old schema detected - add chunk_id column
                cursor.execute(
                    "ALTER TABLE summaries ADD COLUMN chunk_id INTEGER REFERENCES chunks(id)"
                )

                # Migrate data from junction table if it exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='summary_chunks'"
                )
                if cursor.fetchone():
                    cursor.execute("""
                        UPDATE summaries
                        SET chunk_id = (
                            SELECT chunk_id FROM summary_chunks
                            WHERE summary_chunks.summary_id = summaries.id
                        )
                        WHERE level = 0
                    """)

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON summaries(chunk_id)")
                conn.commit()

            # Drop the old junction table if it exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='summary_chunks'"
            )
            if cursor.fetchone():
                cursor.execute("DROP TABLE summary_chunks")
                conn.commit()

    def add_chunk(self, text: str, start: int, end: int, source: str = "") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chunks (text, start_index, end_index, file_source) VALUES (?, ?, ?, ?)",
                (text, start, end, source),
            )
            return cursor.lastrowid

    def add_summary(
        self,
        text: str,
        level: int,
        parent_id: Optional[int] = None,
        sequence_index: int = 0,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO summaries (summary_text, level, parent_id, sequence_index) VALUES (?, ?, ?, ?)",
                (text, level, parent_id, sequence_index),
            )
            return cursor.lastrowid

    def link_summary_to_chunk(self, summary_id: int, chunk_id: int) -> None:
        """Links a summary to its source chunk using direct column."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE summaries SET chunk_id = ? WHERE id = ?",
                (chunk_id, summary_id),
            )

    def update_summary_parent(self, summary_id: int, parent_id: int) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE summaries SET parent_id = ? WHERE id = ?",
                (parent_id, summary_id),
            )

    def get_root_summaries(self) -> List[Tuple[int, str]]:
        """Returns list of (id, text) for the highest level nodes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(level) FROM summaries")
            res = cursor.fetchone()
            max_level = res[0] if res and res[0] is not None else -1

            if max_level < 0:
                return []

            cursor.execute(
                "SELECT id, summary_text FROM summaries WHERE level = ? ORDER BY sequence_index ASC",
                (max_level,),
            )
            return cursor.fetchall()

    def get_node_metadata(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """Lightweight lookup to check a node's level before deciding how to handle it."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, level, summary_text FROM summaries WHERE id = ?",
                (summary_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {"id": row[0], "level": row[1], "text": row[2]}

    def get_child_summaries(self, parent_id: int) -> List[Tuple[int, str]]:
        """Returns child summaries (id, text) for navigation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, summary_text FROM summaries WHERE parent_id = ? ORDER BY sequence_index ASC",
                (parent_id,),
            )
            return cursor.fetchall()

    def get_linked_chunk_id(self, summary_id: int) -> Optional[int]:
        """Returns the raw chunk ID associated with a leaf summary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT chunk_id FROM summaries WHERE id = ?",
                (summary_id,),
            )
            res = cursor.fetchone()
            return res[0] if res else None

    def get_adjacent_nodes(self, summary_id: int) -> Dict[str, Optional[int]]:
        """
        Returns adjacent nodes for navigation.
        Returns {'prev': id, 'next': id, 'parent': id}
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT parent_id, sequence_index, level FROM summaries WHERE id = ?",
                (summary_id,),
            )
            row = cursor.fetchone()
            if not row:
                return {"prev": None, "next": None, "parent": None}

            parent_id, current_seq, level = row
            result: Dict[str, Optional[int]] = {"parent": parent_id}

            # Find Previous Sibling
            if parent_id is None:
                cursor.execute(
                    """
                    SELECT id FROM summaries
                    WHERE parent_id IS NULL AND level = ? AND sequence_index < ?
                    ORDER BY sequence_index DESC LIMIT 1
                    """,
                    (level, current_seq),
                )
            else:
                cursor.execute(
                    """
                    SELECT id FROM summaries
                    WHERE parent_id = ? AND level = ? AND sequence_index < ?
                    ORDER BY sequence_index DESC LIMIT 1
                    """,
                    (parent_id, level, current_seq),
                )
            prev_row = cursor.fetchone()
            result["prev"] = prev_row[0] if prev_row else None

            # Find Next Sibling
            if parent_id is None:
                cursor.execute(
                    """
                    SELECT id FROM summaries
                    WHERE parent_id IS NULL AND level = ? AND sequence_index > ?
                    ORDER BY sequence_index ASC LIMIT 1
                    """,
                    (level, current_seq),
                )
            else:
                cursor.execute(
                    """
                    SELECT id FROM summaries
                    WHERE parent_id = ? AND level = ? AND sequence_index > ?
                    ORDER BY sequence_index ASC LIMIT 1
                    """,
                    (parent_id, level, current_seq),
                )
            next_row = cursor.fetchone()
            result["next"] = next_row[0] if next_row else None

            return result

    def get_chunk_text(self, chunk_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
            res = cursor.fetchone()
            return res[0] if res else None

    def get_chunk_texts(self, chunk_ids: List[int]) -> List[Optional[str]]:
        if not chunk_ids:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"SELECT id, text FROM chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            results = {row[0]: row[1] for row in cursor.fetchall()}
            return [results.get(cid) for cid in chunk_ids]

    def get_summary(self, summary_id: int) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT summary_text FROM summaries WHERE id = ?",
                (summary_id,),
            )
            res = cursor.fetchone()
            return res[0] if res else None

    def get_summaries(self, summary_ids: List[int]) -> List[Optional[str]]:
        if not summary_ids:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(summary_ids))
            cursor.execute(
                f"SELECT id, summary_text FROM summaries WHERE id IN ({placeholders})",
                summary_ids,
            )
            results = {row[0]: row[1] for row in cursor.fetchall()}
            return [results.get(sid) for sid in summary_ids]

    def search_summaries(self, query: str, limit: int = 10) -> List[Tuple[int, int, str]]:
        """Returns (id, level, text) matches."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, level, summary_text FROM summaries WHERE summary_text LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            )
            return cursor.fetchall()

    # -------------------------------------------------------------------------
    # Validation and Repair Methods
    # -------------------------------------------------------------------------

    def get_broken_summaries(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Scans all summaries and returns categorized broken ones.
        Returns: {
            'provider_error': [(id, text), ...],
            'think_blocks': [(id, text), ...],
            'markdown_prefix': [(id, text), ...]
        }
        """
        broken: Dict[str, List[Tuple[int, str]]] = {
            "provider_error": [],
            "think_blocks": [],
            "markdown_prefix": [],
        }

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, summary_text FROM summaries")

            for row in cursor.fetchall():
                summary_id, text = row
                if not text:
                    continue

                # Check for provider errors
                if "Provider returned error" in text:
                    broken["provider_error"].append((summary_id, text))
                # Check for think blocks
                elif "<think>" in text or "</think>" in text:
                    broken["think_blocks"].append((summary_id, text))
                # Check for markdown code block prefix
                elif text.strip().startswith("```markdown"):
                    broken["markdown_prefix"].append((summary_id, text))

        return broken

    def get_summary_with_context(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """
        Get summary with its associated chunk text (for regeneration).
        Returns: {'id': int, 'level': int, 'chunk_id': int, 'chunk_text': str, 'child_texts': List[str]}
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get summary metadata
            cursor.execute(
                "SELECT id, level, parent_id, sequence_index, chunk_id FROM summaries WHERE id = ?",
                (summary_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            result: Dict[str, Any] = {
                "id": row[0],
                "level": row[1],
                "parent_id": row[2],
                "sequence_index": row[3],
                "chunk_id": row[4],
                "chunk_text": None,
                "child_texts": [],
            }

            if row[1] == 0 and row[4]:  # Level 0 - get chunk text
                cursor.execute("SELECT text FROM chunks WHERE id = ?", (row[4],))
                chunk_row = cursor.fetchone()
                if chunk_row:
                    result["chunk_text"] = chunk_row[0]
            else:  # Higher level - get child summary texts
                cursor.execute(
                    "SELECT summary_text FROM summaries WHERE parent_id = ? ORDER BY sequence_index",
                    (summary_id,),
                )
                result["child_texts"] = [r[0] for r in cursor.fetchall()]

            return result

    def update_summary_text(self, summary_id: int, new_text: str) -> None:
        """Update the text of an existing summary."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE summaries SET summary_text = ? WHERE id = ?",
                (new_text, summary_id),
            )
