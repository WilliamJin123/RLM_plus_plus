import json
import logging
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.factory import AgentFactory
from src.core.indexer import Indexer
from src.core.validator import DatabaseValidator

logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class BenchmarkLogic(ABC):
    """Defines the custom logic for a specific dataset (LongBench, Oolong, etc)."""

    @abstractmethod
    def load_data(self, subset: str, limit: int = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_context(self, item: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def create_prompt(self, item: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def evaluate(self, agent_response: str, item: Dict[str, Any]) -> Tuple[bool, str]:
        """Returns (is_correct, expected_answer_string)"""
        pass


class BenchmarkEngine:
    def __init__(
        self,
        name: str,
        subset: str,
        strategy: BenchmarkLogic,
        max_chunk_tokens: int = 50000,
        db_output_dir: str = None,
    ):
        self.name = name
        self.subset = subset
        self.strategy = strategy
        self.output_file = RESULTS_DIR / f"{name}_{subset}_results.jsonl"
        self.max_chunk_tokens = max_chunk_tokens

        if db_output_dir:
            self.db_storage_dir = Path(db_output_dir)
        else:
            self.db_storage_dir = BASE_DIR / "benchmark_dbs" / name / subset

        self.db_storage_dir.mkdir(parents=True, exist_ok=True)

    def _ingest_context(self, context: str, db_path: Path) -> None:
        """Ingests context into a temporary DB."""
        indexer = Indexer(db_path=str(db_path), max_chunk_tokens=self.max_chunk_tokens)

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmp:
            tmp.write(context)
            tmp_path = tmp.name

        try:
            indexer.ingest_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_processed_indices(self) -> Set[int]:
        """Returns set of already processed item indices."""
        processed: Set[int] = set()
        if not self.output_file.exists():
            return processed

        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed.add(record["index"])
                except (json.JSONDecodeError, KeyError):
                    continue

        return processed

    def _load_existing_stats(self) -> Tuple[int, int]:
        """Returns (correct_count, total_count) from existing results."""
        correct_count = 0
        total_count = 0

        if not self.output_file.exists():
            return correct_count, total_count

        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("is_correct"):
                        correct_count += 1
                    total_count += 1
                except json.JSONDecodeError:
                    continue

        return correct_count, total_count

    def _save_result(
        self,
        index: int,
        question: str,
        response: str,
        expected: str,
        is_correct: bool,
    ) -> None:
        """Saves result to JSONL (does NOT save full context to save space)."""
        record = {
            "index": index,
            "dataset": self.name,
            "subset": self.subset,
            "question": question,
            "agent_response": response,
            "expected_answer": expected,
            "is_correct": is_correct,
        }
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write("\n"+ json.dumps(record) + "\n")

    def run(self, limit: int = None, questions: List[int] = None) -> None:
        data = self.strategy.load_data(self.subset, limit)
        processed = self._get_processed_indices()
        correct_count, total_processed_count = self._load_existing_stats()

        # Convert questions to a set for O(1) lookup (already 0-based from CLI parsing)
        questions_set = set(questions) if questions else None

        # Validate question indices
        if questions_set:
            max_idx = len(data) - 1
            invalid = [q + 1 for q in questions_set if q < 0 or q > max_idx]
            if invalid:
                logger.warning(
                    "Question indices out of range (1-%d): %s", len(data), invalid
                )
            questions_set = {q for q in questions_set if 0 <= q <= max_idx}

        logger.info(
            "Resuming %s/%s. %d items already done.",
            self.name,
            self.subset,
            len(processed),
        )

        for i, item in enumerate(data):
            # Skip if not in specified questions list
            if questions_set is not None and i not in questions_set:
                continue
            # Skip if already processed
            if i in processed:
                continue

            logger.info(
                "[%s] Item %d (Total Done: %d)",
                self.name,
                i,
                total_processed_count,
            )

            db_filename = f"q_{i}.db"
            item_db_path = self.db_storage_dir / db_filename

            context = self.strategy.get_context(item)
            if not context:
                logger.warning("Skipping empty context for item %d", i)
                continue

            # Validate and repair existing DB, or ingest new one
            if item_db_path.exists():
                logger.info("Found existing DB at %s, validating...", item_db_path)
                validator = DatabaseValidator(str(item_db_path))
                issues = validator.validate()

                # Count issues properly (incomplete_summaries is a dict, not a list)
                missing_level_0 = len(issues["incomplete_summaries"]["missing_level_0"])
                orphan_summaries = len(issues["incomplete_summaries"]["orphan_summary_ids"])
                total_issues = (
                    len(issues["provider_error"])
                    + len(issues["think_blocks"])
                    + len(issues["markdown_prefix"])
                    + missing_level_0
                    + orphan_summaries
                )

                if total_issues > 0:
                    logger.info(
                        "Found %d issues: %d provider_error, %d think_blocks, %d markdown_prefix, %d missing_level_0, %d orphan_summaries",
                        total_issues,
                        len(issues["provider_error"]),
                        len(issues["think_blocks"]),
                        len(issues["markdown_prefix"]),
                        missing_level_0,
                        orphan_summaries,
                    )
                    stats = validator.repair(dry_run=False, issues=issues)
                    logger.info(
                        "Repair complete: cleaned=%d, regenerated=%d, failed=%d, generated_level_0=%d, generated_hierarchy=%d",
                        stats["cleaned"],
                        stats["regenerated"],
                        stats["failed"],
                        stats.get("generated_level_0", 0),
                        stats.get("generated_hierarchy", 0),
                    )
                else:
                    logger.info("Database validation passed.")
            else:
                logger.info("Ingesting %d chars...", len(context))
                self._ingest_context(context, item_db_path)

            # Create agent with unique session ID
            session_id = f"bench_{self.name}_{self.subset}_{i}"
            agent = AgentFactory.create_agent(
                "rlm-agent",
                content_db_path=str(item_db_path),
                session_id=session_id,
            )

            prompt = self.strategy.create_prompt(item)
            logger.info("Asking Agent...")

            response_obj = agent.run(prompt)
            response_text = str(response_obj.content)
            logger.info("Agent response: %s", response_text)

            is_correct, expected = self.strategy.evaluate(response_text, item)

            status = "CORRECT" if is_correct else f"WRONG (Exp: {expected})"
            logger.info("Result: %s", status)

            if is_correct:
                correct_count += 1
            total_processed_count += 1

            self._save_result(
                i,
                item.get("question", ""),
                response_text,
                expected,
                is_correct,
            )

        logger.info("Benchmarking Complete")
        if total_processed_count > 0:
            acc = (correct_count / total_processed_count) * 100
            logger.info(
                "Final Accuracy: %d/%d (%.2f%%)",
                correct_count,
                total_processed_count,
                acc,
            )
