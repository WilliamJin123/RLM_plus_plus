"""Database validation and repair for RLM storage."""

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from src.config.config import CONFIG
from src.core.factory import AgentFactory, ModelRotator
from src.core.storage import StorageEngine, clean_summary_text

logger = logging.getLogger(__name__)


class DatabaseValidator:
    """Validates and repairs RLM database files."""

    def __init__(self, db_path: str, num_keys: int = 20):
        self.db_path = db_path
        self.storage = StorageEngine(db_path)

        self.key_queue: queue.Queue[int] = queue.Queue()
        for i in range(num_keys):
            self.key_queue.put(i)

        self.db_lock = threading.Lock()
        self.max_workers = num_keys

        # Setup model rotator
        summary_config = CONFIG.get_agent("summarization-agent")
        if summary_config and summary_config.model_pool:
            self.summary_rotator: Optional[ModelRotator] = ModelRotator(
                configs=summary_config.model_pool.models,
                calls_per_model=summary_config.model_pool.calls_per_model,
            )
        else:
            self.summary_rotator = None

    def validate(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Validates database and returns categorized issues.
        Returns dict with keys: 'provider_error', 'think_blocks', 'markdown_prefix'
        """
        return self.storage.get_broken_summaries()

    def repair(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Repairs all broken summaries.

        Args:
            dry_run: If True, only report what would be fixed without making changes.

        Returns:
            Dict with counts: {'cleaned': int, 'regenerated': int, 'failed': int}
        """
        broken = self.validate()
        stats = {"cleaned": 0, "regenerated": 0, "failed": 0}

        # Phase 1: Clean fixable summaries (think blocks, markdown prefix)
        cleanable = broken["think_blocks"] + broken["markdown_prefix"]
        for summary_id, text in cleanable:
            cleaned = clean_summary_text(text)
            if cleaned and cleaned != text:
                if not dry_run:
                    self.storage.update_summary_text(summary_id, cleaned)
                stats["cleaned"] += 1
                logger.info("Cleaned summary %d", summary_id)

        # Phase 2: Regenerate provider errors (requires LLM calls)
        if broken["provider_error"]:
            regenerate_results = self._regenerate_summaries_parallel(
                [sid for sid, _ in broken["provider_error"]], dry_run=dry_run
            )
            stats["regenerated"] = regenerate_results["success"]
            stats["failed"] = regenerate_results["failed"]

        return stats

    def _regenerate_summaries_parallel(
        self, summary_ids: List[int], dry_run: bool = False
    ) -> Dict[str, int]:
        """Regenerate multiple summaries in parallel."""
        results = {"success": 0, "failed": 0}

        if dry_run:
            results["success"] = len(summary_ids)
            return results

        # Separate by level (level 0 needs chunk text, higher levels need child texts)
        level_0_items: List[Tuple[int, Dict]] = []
        higher_level_items: List[Tuple[int, Dict]] = []

        for sid in summary_ids:
            context = self.storage.get_summary_with_context(sid)
            if context and context["level"] == 0:
                level_0_items.append((sid, context))
            elif context:
                higher_level_items.append((sid, context))

        # Process level 0 summaries (from chunk text)
        if level_0_items:
            logger.info("Regenerating %d level-0 summaries...", len(level_0_items))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for sid, context in level_0_items:
                    if context["chunk_text"]:
                        prompt = (
                            f"Summarize the following document segment. "
                            f"Identify key topics, entities, and events:\n\n{context['chunk_text']}"
                        )
                        futures.append((sid, executor.submit(self._get_summary_from_llm, prompt)))

                for sid, future in futures:
                    try:
                        new_text = future.result()
                        if new_text and "Error" not in new_text:
                            self.storage.update_summary_text(sid, new_text)
                            results["success"] += 1
                            logger.info("Regenerated summary %d", sid)
                        else:
                            results["failed"] += 1
                            logger.warning("Failed to regenerate summary %d: %s", sid, new_text)
                    except Exception as e:
                        logger.error("Failed to regenerate summary %d: %s", sid, e)
                        results["failed"] += 1

        # Process higher level summaries (from child texts)
        if higher_level_items:
            logger.info("Regenerating %d higher-level summaries...", len(higher_level_items))
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for sid, context in higher_level_items:
                    if context["child_texts"]:
                        combined = "\n\n".join(context["child_texts"])
                        prompt = (
                            f"Synthesize the following summaries into a cohesive higher-level summary:\n\n"
                            f"{combined}"
                        )
                        futures.append((sid, executor.submit(self._get_summary_from_llm, prompt)))

                for sid, future in futures:
                    try:
                        new_text = future.result()
                        if new_text and "Error" not in new_text:
                            self.storage.update_summary_text(sid, new_text)
                            results["success"] += 1
                            logger.info("Regenerated summary %d", sid)
                        else:
                            results["failed"] += 1
                            logger.warning("Failed to regenerate summary %d: %s", sid, new_text)
                    except Exception as e:
                        logger.error("Failed to regenerate summary %d: %s", sid, e)
                        results["failed"] += 1

        return results

    def _get_summary_from_llm(self, prompt: str) -> str:
        """Thread-safe LLM call (same pattern as Indexer)."""
        key_index = self.key_queue.get()

        try:
            agent = AgentFactory.create_agent("summarization-agent", key_index=key_index)

            if self.summary_rotator:
                model_config = self.summary_rotator.get_next_config()
                agent.model = AgentFactory.create_model(model_config)

            response = agent.run(prompt)
            cleaned = response.content.replace("###", "")
            return clean_summary_text(cleaned)  # Apply cleaning to new summaries too
        except Exception as e:
            logger.error("Error in LLM thread: %s", e)
            return "Error generating summary."
        finally:
            self.key_queue.put(key_index)
