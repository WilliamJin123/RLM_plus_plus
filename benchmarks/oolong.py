import ast
import re
from typing import Dict, List, Any, Tuple
from datasets import load_dataset
from base import BASE_DIR, BenchmarkLogic


    
class OolongLogic(BenchmarkLogic):
    def load_data(self, subset: str, limit: int = None) -> List[Dict[str, Any]]:
        print(f"Loading Oolong ({subset})...")
        data_dir = BASE_DIR / "datasets" / "oolong" / "filtered_oolong_parquet"
        
        filename = f"{subset}_1024000_plus.parquet"
        file_path = data_dir / filename
        
        if not file_path.exists():
             raise FileNotFoundError(f"Dataset not found: {file_path}")

        ds = load_dataset("parquet", data_files=str(file_path), split="train")
        if limit:
            ds = ds.select(range(limit))
        return ds

    def get_context(self, item: Dict[str, Any]) -> str:
        return item.get('context_window_text', '')

    def create_prompt(self, item: Dict[str, Any]) -> str:
        return (
            f"Question: {item.get('question', '')}\n\n"
            "INSTRUCTIONS:\n"
            "1. Search the indexed context using your tools.\n"
            "2. Follow the format requested in the question exactly (e.g. 'Label: answer').\n"
            "   Do not provide any other text in the final line."
        )

    def evaluate(self, agent_response: str, item: Dict[str, Any]) -> Tuple[bool, str]:
        raw_ans = item.get('answer', '')
        # Oolong answers are often stringified lists "['correct']"
        try:
            answers = ast.literal_eval(raw_ans)
            target = str(answers[0]).lower().strip() if isinstance(answers, list) else str(raw_ans).lower().strip()
        except:
            target = str(raw_ans).lower().strip()

        resp = agent_response.strip()
        
        # Look for "Label: answer"
        match = re.search(r"Label:\s*([a-zA-Z]+)", resp, re.IGNORECASE)
        if match:
            prediction = match.group(1).lower()
        else:
            # Fallback: check last word
            last_word = re.split(r'\s+', resp)[-1].lower()
            prediction = re.sub(r'[^\w]', '', last_word)
            
        return (prediction == target, target)

