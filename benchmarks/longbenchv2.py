import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from base import BASE_DIR, BenchmarkLogic


class LongBenchLogic(BenchmarkLogic):
    def load_data(self, subset: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        path = BASE_DIR / "datasets" / "longbenchv2" / f"{subset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data[:limit] if limit else data

    def get_context(self, item: Dict[str, Any]) -> str:
        return item.get("context", "")

    def create_prompt(self, item: Dict[str, Any]) -> str:
        choices = [
            f"A: {item.get('choice_A', '')}",
            f"B: {item.get('choice_B', '')}",
            f"C: {item.get('choice_C', '')}",
            f"D: {item.get('choice_D', '')}",
        ]
        choices_text = "\n".join(choices)
        return (
            f"Question: {item['question']}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Search the indexed context using your tools.\n"
            "2. Provide your final answer in the format: 'ANSWER: <Letter>'.\n"
            "   Example: 'ANSWER: B'\n"
            "   Do not provide any other text in the final line."
        )

    def evaluate(self, agent_response: str, item: Dict[str, Any]) -> Tuple[bool, str]:
        correct_answer = item["answer"].upper()
        resp = agent_response.strip().upper()

        # Try explicit ANSWER pattern first
        match = re.search(r"ANSWER:?\s*([A-D])", resp)
        if match:
            pred = match.group(1)
        else:
            # Fallback for "The answer is B" or "OPTION B"
            match = re.search(r"(?:OPTION|CHOICE)\s*([A-D])", resp)
            if match:
                pred = match.group(1)
            elif resp in ["A", "B", "C", "D"]:
                pred = resp
            else:
                pred = None

        return pred == correct_answer, correct_answer
