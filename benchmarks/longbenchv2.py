
import json
import re
from typing import Dict, List, Any, Tuple

from base import BASE_DIR, BenchmarkLogic


class LongBenchLogic(BenchmarkLogic):
    def load_data(self, subset: str, limit: int = None) -> List[Dict[str, Any]]:
        print(f"Loading LongBench ({subset})...")
        path = BASE_DIR / "datasets" / "longbenchv2" / f"{subset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:limit] if limit else data

    def get_context(self, item: Dict[str, Any]) -> str:
        return item.get('context', '')

    def create_prompt(self, item: Dict[str, Any]) -> str:
        choices = [
            f"A: {item.get('choice_A', '')}",
            f"B: {item.get('choice_B', '')}",
            f"C: {item.get('choice_C', '')}",
            f"D: {item.get('choice_D', '')}"
        ]
        return (
            f"Question: {item['question']}\n\n"
            f"Choices:\n{'\n'.join(choices)}\n\n"
            "INSTRUCTIONS:\n"
            "1. Search the indexed context using your tools.\n"
            "2. Provide your final answer in the format: 'ANSWER: <Letter>'.\n"
            "   Example: 'ANSWER: B'\n"
            "   Do not provide any other text in the final line."
        )

    def evaluate(self, agent_response: str, item: Dict[str, Any]) -> Tuple[bool, str]:
        correct_answer = item['answer'].upper()
        resp = agent_response.strip().upper()
        
        # Robust Regex for "Answer: A" or just "A"
        match = re.search(r"ANSWER:?\s*([A-D])", resp)
        if match:
            pred = match.group(1)
        else:
            # Fallback for "The answer is B"
            match = re.search(r"(?:OPTION|CHOICE)\s*([A-D])", resp)
            pred = match.group(1) if match else (resp if resp in ["A","B","C","D"] else None)
            
        return (pred == correct_answer, correct_answer)
