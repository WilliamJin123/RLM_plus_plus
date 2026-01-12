import os
import json
import argparse
import tempfile
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure src is in path if running from root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR.as_posix())

from src.core.indexer import Indexer
from src.core.factory import AgentFactory
# from src.core.overseer import overseer # Overseer was deleted previously

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    print(f"Loading dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def ingest_context(indexer: Indexer, context: str):
    # Create a temp file to ingest
    # Using utf-8 explicitly
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp:
        tmp.write(context)
        tmp_path = tmp.name
    
    try:
        # Ingest the file
        indexer.ingest_file(tmp_path, target_chunk_tokens=30000, group_size=2)
    except Exception as e:
        print(f"Error during ingestion: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def evaluate_answer(agent_response: str, correct_answer: str) -> bool:
    resp = agent_response.strip().upper()
    
    # Regex for "ANSWER: [ABCD]" or "ANSWER: [ABCD]" or just the letter
    match = re.search(r"ANSWER:?\s*([A-D])", resp)
    if match:
        prediction = match.group(1)
    else:
        # Fallback: check if the response is just a single letter or surrounded by non-word chars
        # e.g. "The answer is B." -> match B
        # But be careful of false positives.
        # Let's try finding the last occurrence of a single letter A-D?
        # Or look for "Option B", "Choice C"
        match = re.search(r"(?:OPTION|CHOICE)\s*([A-D])", resp)
        if match:
            prediction = match.group(1)
        else:
            # Last resort: if the whole string is just one letter
            if resp in ["A", "B", "C", "D"]:
                prediction = resp
            else:
                prediction = None
            
    if prediction == correct_answer.upper():
        return True
    return False

def run_benchmark(dataset_name: str, limit: int = None):
    # Paths
    if dataset_name == "code_qa":
        file_path = BASE_DIR / "datasets" / "longbenchv2" / "code_qa.json"
    elif dataset_name == "history_qa":
        file_path = BASE_DIR / "datasets" / "longbenchv2" / "history_qa.json"
    else:
        raise ValueError("Invalid dataset name. Choose 'code_qa' or 'history_qa'.")
        
    if not file_path.exists():
        print(f"Error: Dataset file not found at {file_path}")
        return

    data = load_dataset(str(file_path))
    
    if limit:
        data = data[:limit]
        
    correct_count = 0
    total_count = 0
    
    db_path = str(BASE_DIR / "data" / "longbench_temp.db")
    
    for item in data:
        print(f"\n--- Processing Item {total_count + 1}/{len(data)} ---")
        
        # 1. Reset DB
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                print("Warning: Could not remove DB file. It might be in use.")
            
        # 2. Initialize Indexer (this also inits the global DB engine)
        indexer = Indexer(db_path)
        
        # 3. Ingest Context
        context = item.get('context', '')
        if not context:
            print("Empty context, skipping.")
            continue
            
        print(f"Ingesting context ({len(context)} chars)...")
        ingest_context(indexer, context)
        
        # 4. Prepare Agent
        # Use a unique session ID
        session_id = f"bench_{dataset_name}_{total_count}"
        try:
            agent = AgentFactory.create_agent(
                "rlm-agent", 
                session_id=session_id, 
                add_history_to_context=False, 
                read_chat_history=False
            )
        except Exception as e:
            print(f"Error creating agent: {e}")
            continue
        
        # 5. Formulate Query
        question = item['question']
        choices = [
            f"A: {item.get('choice_A', '')}",
            f"B: {item.get('choice_B', '')}",
            f"C: {item.get('choice_C', '')}",
            f"D: {item.get('choice_D', '')}"
        ]
        choices_str = "\n".join(choices)
        
        prompt = (
            f"Question: {question}\n\n"
            f"Choices:\n{choices_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the provided choices carefully.\n"
            "2. Search the indexed context (using your tools) to find the answer.\n"
            "3. Provide your final answer in the format: 'ANSWER: <Letter>'.\n"
            "   Example: 'ANSWER: B'\n"
            "   Do not provide any other text in the final line."
        )
        
        print("Asking Agent...")
        try:
            response = agent.run(prompt)
            content = str(response.content)
            print(f"Agent Response: {content}") 
            
            answer = item['answer']
            is_correct = evaluate_answer(content, answer)
            
            if is_correct:
                print("Result: CORRECT")
                correct_count += 1
            else:
                print(f"Result: WRONG (Expected {answer})")
                
        except Exception as e:
            print(f"Error running agent: {e}")
            
        total_count += 1
        
    print(f"\n--- Benchmark Complete ---")
    print(f"Dataset: {dataset_name}")
    if total_count > 0:
        print(f"Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)")
    else:
        print("No items processed.")
    
    # Cleanup final DB
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongBenchV2 Benchmark")
    parser.add_argument("dataset", choices=["code_qa", "history_qa"], help="Which dataset to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    
    args = parser.parse_args()
    
    run_benchmark(args.dataset, args.limit)