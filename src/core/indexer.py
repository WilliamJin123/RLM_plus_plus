import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from src.chunking.base import BaseChunker
from src.chunking.fixed import FixedTokenChunker
from src.chunking.llm import SemanticBoundaryChunker
from src.config.config import CONFIG
from src.core.factory import AgentFactory, ModelRotator
from src.core.storage import StorageEngine, clean_summary_text
from src.utils.token_buffer import TokenBuffer

logger = logging.getLogger(__name__)

VALID_STRATEGIES = {"fixed", "llm"}


class Indexer:
    def __init__(
        self,
        db_path: Optional[str] = None,
        max_chunk_tokens: int = 4000,
        strategy: str = "fixed",
        num_keys: int = 20,
    ):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown chunking strategy: {strategy}. Valid: {VALID_STRATEGIES}")

        self.db_path = db_path
        self.storage = StorageEngine(self.db_path)
        self.token_buffer = TokenBuffer(model_name="gpt-4o")
        self.max_chunk_tokens = max_chunk_tokens

        self.key_queue: queue.Queue[int] = queue.Queue()
        for i in range(num_keys):
            self.key_queue.put(i)

        self.db_lock = threading.Lock()
        self.max_workers = num_keys

        self.chunker: BaseChunker
        if strategy == "llm":
            self.chunker = SemanticBoundaryChunker(max_chunk_tokens, self.token_buffer)
        else:
            self.chunker = FixedTokenChunker(max_chunk_tokens, self.token_buffer)

        # Create model rotator for summarization agent if configured
        summary_config = CONFIG.get_agent("summarization-agent")
        if summary_config and summary_config.model_pool:
            self.summary_rotator: Optional[ModelRotator] = ModelRotator(
                configs=summary_config.model_pool.models,
                calls_per_model=summary_config.model_pool.calls_per_model,
            )
            logger.info(
                "Initialized ModelRotator for summarization with %d models",
                len(self.summary_rotator),
            )
        else:
            self.summary_rotator = None

    def _get_summary_from_llm(self, prompt: str) -> str:
        """Thread-safe wrapper to checkout a key, run the agent, and return the key."""
        key_index = self.key_queue.get()

        try:
            agent = AgentFactory.create_agent("summarization-agent", key_index=key_index)

            # Rotate model if configured
            if self.summary_rotator:
                model_config = self.summary_rotator.get_next_config()
                agent.model = AgentFactory.create_model(model_config)
                logger.debug(
                    "Using model: %s/%s", model_config.provider, model_config.model_id
                )

            response = agent.run(prompt)
            cleaned_content = clean_summary_text(response.content)
            return cleaned_content
        except Exception as e:
            logger.error("Error in LLM thread: %s", e)
            return "Error generating summary."
        finally:
            self.key_queue.put(key_index)

    def ingest_file(self, file_path: str, group_size: int = 5, max_depth: int = 1) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        logger.info("Indexing %s using %d threads", file_path, self.max_workers)

        with path.open("r", encoding="utf-8") as f:
            full_text = f.read()

        if not full_text.strip():
            logger.warning("File is empty: %s", file_path)
            return

        level_0_ids = self._process_chunks_parallel(full_text, path.name)

        if max_depth > 0 and len(level_0_ids) > 1:
            self._build_hierarchy_parallel(level_0_ids, group_size=group_size, max_depth=max_depth)

        logger.info("Indexing complete for %s", file_path)

    def _process_chunks_parallel(self, full_text: str, filename: str) -> List[int]:
        """
        1. Chunks text (Main Thread).
        2. Saves chunks to DB (Main Thread - fast).
        3. Generates summaries (Parallel Threads).
        4. Saves summaries/links (Main Thread via Lock).
        """
        logger.info("Chunking text...")
        chunks = list(self.chunker.chunk_text(full_text))

        chunk_ids = []
        with self.db_lock:
            for chunk_res in chunks:
                chunk_id = self.storage.add_chunk(
                    text=chunk_res.text,
                    start=chunk_res.start_index,
                    end=chunk_res.end_index,
                    source=filename,
                )
                chunk_ids.append(chunk_id)

        logger.info("Generated %d chunks. Starting parallel summarization...", len(chunk_ids))

        prompts = [
            f"Summarize the following document segment. "
            f"Identify key topics, entities, and events:\n\n{chunk_res.text}"
            for chunk_res in chunks
        ]

        summary_ids = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            summary_iterator = executor.map(self._get_summary_from_llm, prompts)
            for sequence_index, summary_text in enumerate(summary_iterator):
                with self.db_lock:
                    sum_id = self.storage.add_summary(
                        text=summary_text,
                        level=0,
                        parent_id=None,
                        sequence_index=sequence_index,
                    )
                    self.storage.link_summary_to_chunk(sum_id, chunk_ids[sequence_index])
                    summary_ids.append(sum_id)

        return summary_ids

    def _build_hierarchy_parallel(
        self,
        child_ids: List[int],
        group_size: int,
        max_depth: int,
    ) -> None:
        """
        Parallelizes the batch processing within each level.
        Levels must still be processed sequentially (L0 -> L1 -> L2).
        """
        current_ids = child_ids
        current_level = 0

        while current_level < max_depth and len(current_ids) > 1:
            logger.info("Building Level %d from %d nodes...", current_level + 1, len(current_ids))

            batches_ids = []
            prompts = []

            for i in range(0, len(current_ids), group_size):
                batch_ids = current_ids[i : i + group_size]
                batches_ids.append(batch_ids)

                with self.db_lock:
                    batch_texts = self.storage.get_summaries(batch_ids)

                combined_text = "\n\n".join(t for t in batch_texts if t)
                prompts.append(
                    f"Synthesize the following summaries into a cohesive higher-level summary:\n\n"
                    f"{combined_text}"
                )

            next_level_ids = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                summary_iterator = executor.map(self._get_summary_from_llm, prompts)

                for sequence_index, summary_text in enumerate(summary_iterator):
                    batch_ids = batches_ids[sequence_index]

                    with self.db_lock:
                        parent_id = self.storage.add_summary(
                            text=summary_text,
                            level=current_level + 1,
                            sequence_index=sequence_index,
                        )
                        for child_id in batch_ids:
                            self.storage.update_summary_parent(child_id, parent_id)

                        next_level_ids.append(parent_id)

            current_ids = next_level_ids
            current_level += 1

        if len(current_ids) == 1:
            logger.info("Tree converged to a single root node.")
