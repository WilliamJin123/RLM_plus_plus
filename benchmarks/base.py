import os
import json
import tempfile
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR.as_posix())

from src.core.indexer import Indexer
from src.core.factory import AgentFactory

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TEMP_DB_PATH = BASE_DIR / "data" / "benchmark_temp.db"

class BenchmarkLogic(ABC):
    """
    Defines the custom logic for a specific dataset (LongBench, Oolong, etc).
    """
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
    def __init__(self, name: str, subset: str, strategy: BenchmarkLogic):
        self.name = name
        self.subset = subset
        self.strategy = strategy
        self.output_file = RESULTS_DIR / f"{name}_{subset}_results.jsonl"
    
    def _ingest_context(self, context: str):
        """Ingests context into a temporary DB."""
        if os.path.exists(TEMP_DB_PATH):
            try:
                os.remove(TEMP_DB_PATH)
            except PermissionError:
                print("Warning: Could not remove DB file.")

        indexer = Indexer(db_path = TEMP_DB_PATH)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp:
            tmp.write(context)
            tmp_path = tmp.name
        
        try:
            # Ensure we use max_chunk_tokens to prevent context window overflow
            indexer.ingest_file(tmp_path, max_chunk_tokens=40000)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_processed_indices(self) -> Set[int]:
        processed = set()
        if not self.output_file.exists():
            return processed
        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed.add(json.loads(line)['index'])
                except: continue
        return processed

    def _save_result(self, index: int, question: str, response: str, expected: str, is_correct: bool):
        """
        Saves result to JSONL. 
        CRITICAL: Does NOT save the full context text to save space.
        """
        record = {
            "index": index,
            "dataset": self.name,
            "subset": self.subset,
            "question": question,
            "agent_response": response,
            "expected_answer": expected,
            "is_correct": is_correct
        }
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

    def run(self, limit: int = None):
        data = self.strategy.load_data(self.subset, limit)
        processed = self._get_processed_indices()
        print(f"Resuming {self.name}/{self.subset}. {len(processed)} items already done.")

        # Calculate initial stats
        correct_count = 0
        total_processed_count = 0
        
        # Pre-scan file for stats
        if self.output_file.exists():
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if json.loads(line)['is_correct']: correct_count += 1
                    total_processed_count += 1

        for i, item in enumerate(data):
            if i in processed:
                continue
            
            print(f"\n--- [{self.name}] Item {i} (Total Done: {total_processed_count}) ---")
            
            # 1. Get Context
            context = self.strategy.get_context(item)
            if not context:
                print("Skipping empty context.")
                continue

            # 2. Ingest
            print(f"Ingesting {len(context)} chars...")
            self._ingest_context(context)

            # Unique session ID avoids cache collision in `agno`
            session_id = f"bench_{self.name}_{self.subset}_{i}"
            agent = AgentFactory.create_agent(
                "rlm-agent",
                content_db_path=str(TEMP_DB_PATH),
                session_id=session_id
            )

            prompt = self.strategy.create_prompt(item)
            print("Asking Agent...")
            
            response_obj = agent.run(prompt)
            response_text = str(response_obj.content)
            print(f"Agent: {response_text}")

            is_correct, expected = self.strategy.evaluate(response_text, item)
            
            status = "CORRECT" if is_correct else f"WRONG (Exp: {expected})"
            print(f"Result: {status}")

            if is_correct: correct_count += 1
            total_processed_count += 1

            # 5. Persist
            self._save_result(i, item.get('question', ''), response_text, expected, is_correct)


        print(f"\n--- Benchmarking Complete ---")
        if total_processed_count > 0:
            acc = (correct_count / total_processed_count) * 100
            print(f"Final Accuracy: {correct_count}/{total_processed_count} ({acc:.2f}%)")
        
        # Cleanup
        if os.path.exists(TEMP_DB_PATH):
            os.remove(TEMP_DB_PATH)