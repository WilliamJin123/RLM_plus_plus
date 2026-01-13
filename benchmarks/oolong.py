import os
import argparse
import sys
import re
import ast
import tempfile
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any

# Ensure src is in path if running from root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR.as_posix())

from src.core.indexer import Indexer
from src.core.factory import AgentFactory
from src.tools.rlm_tools import RLMTools

def load_oolong_dataset(subset: str = "metaphors", limit: int = None) -> List[Dict[str, Any]]:
    print(f"Loading Oolong dataset ({subset})...")
    
    # Define paths relative to the project root or this file
    data_dir = BASE_DIR / "datasets" / "oolong" / "filtered_oolong_parquet"
    
    if subset == "metaphors":
        file_path = data_dir / "metaphors_1024000_plus.parquet"
    elif subset == "negation":
        file_path = data_dir / "negation_1024000_plus.parquet"
    else:
        raise ValueError("Invalid subset. Choose 'metaphors' or 'negation'.")
        
    if not file_path.exists():
        print(f"Error: Dataset file not found at {file_path}")
        return []

    try:
        # Load using datasets library
        ds = load_dataset("parquet", data_files=str(file_path), split="train")
        
        if limit:
            ds = ds.select(range(limit))
            
        print(f"Loaded {len(ds)} items.")
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def ingest_context(indexer: Indexer, context: str):
    # Create a temp file to ingest
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp:
        tmp.write(context)
        tmp_path = tmp.name
    
    try:
        # Ingest the file
        # target_chunk_tokens=25000 is used in longbenchv2, keeping it similar
        # Updated to max_chunk_tokens
        indexer.ingest_file(tmp_path, max_chunk_tokens=40000)
    except Exception as e:
        print(f"Error during ingestion: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def evaluate_answer(agent_response: str, correct_answer_str: str) -> bool:
    # correct_answer_str usually looks like "['correct']" or "['incorrect']"
    try:
        # Parse the string list
        answers = ast.literal_eval(correct_answer_str)
        if isinstance(answers, list) and len(answers) > 0:
            target = str(answers[0]).lower().strip()
        else:
            target = str(correct_answer_str).lower().strip()
    except:
        target = str(correct_answer_str).lower().strip()
        
    resp = agent_response.strip()
    
    # The prompt usually asks for "Label: answer"
    # We look for "Label: <word>"
    match = re.search(r"Label:\s*([a-zA-Z]+)", resp, re.IGNORECASE)
    if match:
        prediction = match.group(1).lower()
        if prediction == target:
            return True
    
    # Fallback: Check if the target is the very last word (common in "Answer: correct")
    last_word = re.split(r'\s+', resp)[-1].lower()
    # Remove punctuation
    last_word = re.sub(r'[^\w]', '', last_word)
    if last_word == target:
        return True
        
    return False

def run_benchmark(subset: str, limit: int = None):
    data = load_oolong_dataset(subset, limit)
    
    if not data:
        print("No data to process.")
        return

    correct_count = 0
    total_count = 0
    
    db_path = str(BASE_DIR / "data" / "oolong_temp.db")
    
    for i, item in enumerate(data):
        print(f"\n--- Processing Item {total_count + 1}/{len(data)} ---")
        
        # 1. Reset DB
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                print("Warning: Could not remove DB file. It might be in use.")
            
        # 2. Initialize Indexer
        indexer = Indexer(db_path)
        
        # 3. Ingest Context
        context = item.get('context_window_text', '')
        if not context:
            print("Empty context, skipping.")
            continue
            
        print(f"Ingesting context ({len(context)} chars)...")
        ingest_context(indexer, context)
        
        # 4. Prepare Agent
        session_id = f"bench_oolong_{subset}_{total_count}"
        try:
            agent = AgentFactory.create_agent("rlm-agent")
            
            # Manually override tools to use the benchmark DB
            new_tools = [t for t in agent.tools if not isinstance(t, RLMTools)]
            new_tools.append(RLMTools(db_path=db_path))
            agent.tools = new_tools
            
            agent.session_id = session_id

        except Exception as e:
            print(f"Error creating agent: {e}")
            continue
        
        # 5. Formulate Query
        question = item.get('question', '')
        # question usually contains instructions like "Give your final answer in the form 'Label: answer'..."
        
        prompt = (
            f"Question: {question}\n\n"
            "INSTRUCTIONS:\n"
            "1. Search the indexed context using your tools to find the answer.\n"
            "2. Follow the format requested in the question exactly.\n"
            "   Do not provide any other text in the final line."
        )
        
        print("Asking Agent...")
        try:
            response = agent.run(prompt)
            content = str(response.content)
            print(f"Agent Response: {content}") 
            
            answer_str = item.get('answer', '')
            is_correct = evaluate_answer(content, answer_str)
            
            if is_correct:
                print(f"Result: CORRECT (Expected {answer_str})")
                correct_count += 1
            else:
                print(f"Result: WRONG (Expected {answer_str})")
                
        except Exception as e:
            print(f"Error running agent: {e}")
            
        total_count += 1
        
    print(f"\n--- Benchmark Complete ---")
    print(f"Dataset: Oolong ({subset})")
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
    parser = argparse.ArgumentParser(description="Run Oolong Benchmark")
    parser.add_argument("subset", choices=["metaphors", "negation"], help="Which subset to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process")
    
    args = parser.parse_args()
    
    run_benchmark(args.subset, args.limit)
