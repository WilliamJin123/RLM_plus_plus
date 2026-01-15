"""Database validation and repair for RLM storage."""

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

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

    def validate(self) -> Dict[str, Any]:
        """
        Validates database and returns categorized issues.
        Returns dict with keys:
            - 'provider_error': [(id, text), ...]
            - 'think_blocks': [(id, text), ...]
            - 'markdown_prefix': [(id, text), ...]
            - 'incomplete_summaries': {
                'missing_level_0': [(chunk_id, chunk_text), ...],
                'orphan_summary_ids': [id, ...],
                'current_max_level': int,
              }
        """
        issues = self.storage.get_broken_summaries()

        # Check for incomplete summary generation
        missing_level_0 = self.storage.get_chunks_without_summaries()
        orphan_ids = self.storage.get_orphan_summaries()
        max_level = self.storage.get_max_summary_level()

        issues["incomplete_summaries"] = {
            "missing_level_0": missing_level_0,
            "orphan_summary_ids": orphan_ids,
            "current_max_level": max_level,
        }

        return issues

    def repair(
        self,
        dry_run: bool = False,
        issues: Optional[Dict[str, Any]] = None,
        group_size: int = 5,
        max_depth: int = 1,
    ) -> Dict[str, int]:
        """
        Repairs all broken summaries and completes incomplete summary generation.

        Args:
            dry_run: If True, only report what would be fixed without making changes.
            issues: Pre-validated issues dict. If None, validate() will be called.
            group_size: Number of summaries to group when building hierarchy.
            max_depth: Maximum hierarchy depth to build.

        Returns:
            Dict with counts: {
                'cleaned': int,
                'regenerated': int,
                'failed': int,
                'generated_level_0': int,
                'generated_hierarchy': int,
            }
        """
        if issues is None:
            issues = self.validate()

        stats = {
            "cleaned": 0,
            "regenerated": 0,
            "failed": 0,
            "generated_level_0": 0,
            "generated_hierarchy": 0,
        }

        logger.info("=== DB REPAIR START ===")

        # Phase 1: Clean fixable summaries (think blocks, markdown prefix)
        cleanable = issues["think_blocks"] + issues["markdown_prefix"]
        if cleanable:
            logger.info(
                "Phase 1: Cleaning %d think_blocks, %d markdown_prefix issues...",
                len(issues["think_blocks"]),
                len(issues["markdown_prefix"]),
            )
            for summary_id, text in cleanable:
                cleaned = clean_summary_text(text)
                if cleaned and cleaned != text:
                    if not dry_run:
                        self.storage.update_summary_text(summary_id, cleaned)
                    stats["cleaned"] += 1
                    logger.debug("Cleaned summary %d", summary_id)
            logger.info("Phase 1 complete: %d summaries cleaned", stats["cleaned"])

        # Phase 2: Regenerate provider errors (requires LLM calls)
        if issues["provider_error"]:
            logger.info(
                "Phase 2: Regenerating %d provider_error summaries via LLM...",
                len(issues["provider_error"]),
            )
            regenerate_results = self._regenerate_summaries_parallel(
                [sid for sid, _ in issues["provider_error"]], dry_run=dry_run
            )
            stats["regenerated"] = regenerate_results["success"]
            stats["failed"] = regenerate_results["failed"]
            logger.info(
                "Phase 2 complete: %d regenerated, %d failed",
                stats["regenerated"],
                stats["failed"],
            )

        # Phase 3: Generate missing level-0 summaries
        incomplete = issues.get("incomplete_summaries", {})
        missing_level_0 = incomplete.get("missing_level_0", [])
        if missing_level_0:
            logger.info(
                "Phase 3: Generating %d missing level-0 summaries...",
                len(missing_level_0),
            )
            level_0_results = self._generate_missing_level_0_summaries_parallel(
                missing_level_0, dry_run=dry_run
            )
            stats["generated_level_0"] = level_0_results["success"]
            stats["failed"] += level_0_results["failed"]
            logger.info(
                "Phase 3 complete: %d generated, %d failed",
                level_0_results["success"],
                level_0_results["failed"],
            )

        # Phase 4: Complete hierarchy if incomplete
        # Re-check orphans after potentially generating new level-0 summaries
        if not dry_run:
            orphan_ids = self.storage.get_orphan_summaries()
        else:
            orphan_ids = incomplete.get("orphan_summary_ids", [])
            # In dry_run, estimate new orphans from missing_level_0
            orphan_ids = list(orphan_ids) + [None] * len(missing_level_0)

        current_max_level = self.storage.get_max_summary_level() if not dry_run else incomplete.get("current_max_level", 0)

        if len(orphan_ids) > 1 and current_max_level < max_depth:
            logger.info(
                "Phase 4: Completing hierarchy from %d orphan summaries (level %d -> %d)...",
                len(orphan_ids),
                current_max_level,
                max_depth,
            )
            hierarchy_results = self._complete_hierarchy_parallel(
                group_size=group_size,
                max_depth=max_depth,
                dry_run=dry_run,
                estimated_orphan_count=len(orphan_ids) if dry_run else None,
            )
            stats["generated_hierarchy"] = hierarchy_results["success"]
            stats["failed"] += hierarchy_results["failed"]
            logger.info(
                "Phase 4 complete: %d hierarchy summaries generated, %d failed",
                hierarchy_results["success"],
                hierarchy_results["failed"],
            )

        logger.info("=== DB REPAIR COMPLETE ===")

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
            logger.debug("Regenerating %d level-0 summaries...", len(level_0_items))
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
                            logger.debug("Regenerated summary %d", sid)
                        else:
                            results["failed"] += 1
                            logger.warning("Failed to regenerate summary %d: %s", sid, new_text)
                    except Exception as e:
                        logger.error("Failed to regenerate summary %d: %s", sid, e)
                        results["failed"] += 1

        # Process higher level summaries (from child texts)
        if higher_level_items:
            logger.debug("Regenerating %d higher-level summaries...", len(higher_level_items))
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
                            logger.debug("Regenerated summary %d", sid)
                        else:
                            results["failed"] += 1
                            logger.warning("Failed to regenerate summary %d: %s", sid, new_text)
                    except Exception as e:
                        logger.error("Failed to regenerate summary %d: %s", sid, e)
                        results["failed"] += 1

        return results

    def _get_summary_from_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Thread-safe LLM call with retry and force rotation on failure."""
        key_index = self.key_queue.get()

        try:
            agent = AgentFactory.create_agent("summarization-agent", key_index=key_index)

            for attempt in range(max_retries):
                try:
                    if self.summary_rotator:
                        model_config = self.summary_rotator.get_next_config()
                        agent.model = AgentFactory.create_model(model_config)

                    response = agent.run(prompt)
                    content = response.content

                    # Check for provider error in response
                    if "Provider returned error" in content or "No endpoints found" in content:
                        logger.warning(
                            "Provider error on attempt %d/%d", attempt + 1, max_retries
                        )
                        if self.summary_rotator:
                            self.summary_rotator.force_rotate()
                        continue

                    cleaned = content.replace("###", "")
                    return clean_summary_text(cleaned)

                except Exception as e:
                    logger.warning(
                        "LLM call failed attempt %d/%d: %s", attempt + 1, max_retries, e
                    )
                    if self.summary_rotator:
                        self.summary_rotator.force_rotate()
                    if attempt == max_retries - 1:
                        return "Error generating summary."

            return "Error generating summary."
        finally:
            self.key_queue.put(key_index)

    def _generate_missing_level_0_summaries_parallel(
        self,
        missing_chunks: List[Tuple[int, str]],
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        Generate level-0 summaries for chunks that are missing them.

        Args:
            missing_chunks: List of (chunk_id, chunk_text) tuples.
            dry_run: If True, only report what would be generated.

        Returns:
            Dict with 'success' and 'failed' counts.
        """
        results = {"success": 0, "failed": 0}

        if dry_run:
            results["success"] = len(missing_chunks)
            return results

        if not missing_chunks:
            return results

        # Generate summaries in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for chunk_id, chunk_text in missing_chunks:
                prompt = (
                    f"Summarize the following document segment. "
                    f"Identify key topics, entities, and events:\n\n{chunk_text}"
                )
                futures.append((chunk_id, executor.submit(self._get_summary_from_llm, prompt)))

            for chunk_id, future in futures:
                try:
                    summary_text = future.result()
                    if summary_text and "Error" not in summary_text:
                        with self.db_lock:
                            # Get the next sequence index for level 0
                            seq_idx = self.storage.get_next_sequence_index(level=0)
                            sum_id = self.storage.add_summary(
                                text=summary_text,
                                level=0,
                                parent_id=None,
                                sequence_index=seq_idx,
                            )
                            self.storage.link_summary_to_chunk(sum_id, chunk_id)
                        results["success"] += 1
                        logger.debug("Generated level-0 summary for chunk %d", chunk_id)
                    else:
                        results["failed"] += 1
                        logger.warning(
                            "Failed to generate summary for chunk %d: %s", chunk_id, summary_text
                        )
                except Exception as e:
                    logger.error("Failed to generate summary for chunk %d: %s", chunk_id, e)
                    results["failed"] += 1

        return results

    def _complete_hierarchy_parallel(
        self,
        group_size: int = 5,
        max_depth: int = 1,
        dry_run: bool = False,
        estimated_orphan_count: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Complete the hierarchical summarization from where it was interrupted.

        Finds orphan summaries (those at max level without parents) and
        continues building hierarchy until converged or max_depth reached.

        Args:
            group_size: Number of summaries to group into each parent.
            max_depth: Maximum hierarchy depth to build.
            dry_run: If True, only estimate what would be generated.
            estimated_orphan_count: For dry_run, use this count instead of querying DB.

        Returns:
            Dict with 'success' and 'failed' counts.
        """
        results = {"success": 0, "failed": 0}

        if dry_run:
            # Estimate number of hierarchy summaries needed
            orphan_count = estimated_orphan_count if estimated_orphan_count is not None else len(self.storage.get_orphan_summaries())
            current_level = self.storage.get_max_summary_level()
            estimated = 0
            while orphan_count > 1 and current_level < max_depth:
                # Each level reduces count by group_size factor
                new_count = (orphan_count + group_size - 1) // group_size
                estimated += new_count
                orphan_count = new_count
                current_level += 1
            results["success"] = estimated
            return results

        current_level = self.storage.get_max_summary_level()

        while current_level < max_depth:
            # Get current orphans (summaries at max level with no parent)
            orphan_ids = self.storage.get_orphan_summaries()

            if len(orphan_ids) <= 1:
                # Already converged to single root or empty
                break

            logger.info(
                "Building level %d from %d orphan summaries...",
                current_level + 1,
                len(orphan_ids),
            )

            # Group orphans and create parent summaries
            batches_ids = []
            prompts = []

            for i in range(0, len(orphan_ids), group_size):
                batch_ids = orphan_ids[i : i + group_size]
                batches_ids.append(batch_ids)

                with self.db_lock:
                    batch_texts = self.storage.get_summaries(batch_ids)

                combined_text = "\n\n".join(t for t in batch_texts if t)
                prompts.append(
                    f"Synthesize the following summaries into a cohesive higher-level summary:\n\n"
                    f"{combined_text}"
                )

            # Generate parent summaries in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                summary_futures = list(executor.map(self._get_summary_from_llm, prompts))

            # Save results and link children
            for seq_idx, (batch_ids, summary_text) in enumerate(zip(batches_ids, summary_futures)):
                if summary_text and "Error" not in summary_text:
                    with self.db_lock:
                        parent_id = self.storage.add_summary(
                            text=summary_text,
                            level=current_level + 1,
                            sequence_index=seq_idx,
                        )
                        for child_id in batch_ids:
                            self.storage.update_summary_parent(child_id, parent_id)
                    results["success"] += 1
                    logger.debug("Created level-%d summary %d", current_level + 1, parent_id)
                else:
                    results["failed"] += 1
                    logger.warning(
                        "Failed to create level-%d summary for batch: %s",
                        current_level + 1,
                        summary_text,
                    )

            current_level += 1

        return results
