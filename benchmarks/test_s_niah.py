import os
import random
import uuid
import time

import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.as_posix())

from src.core.indexer import Indexer
from src.core.factory import AgentFactory
from src.core.db import init_db

def generate_haystack(file_path: str, size_in_tokens: int = 100000):
    """
    Generates a large text file with a 'needle' (a UUID) hidden in it.
    """
    needle = f"The secret code is: {uuid.uuid4()}"
    
    # Approx 5 chars per token
    filler_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Knowledge is power.",
        "Data is the new oil.",
        "Artificial Intelligence is transforming the world."
    ]
    
    total_chars = size_in_tokens * 5
    needle_pos = random.randint(0, total_chars)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        current_chars = 0
        needle_inserted = False
        while current_chars < total_chars:
            if not needle_inserted and current_chars >= needle_pos:
                f.write(needle + " ")
                current_chars += len(needle) + 1
                needle_inserted = True
            
            sentence = random.choice(filler_sentences)
            f.write(sentence + " ")
            current_chars += len(sentence) + 1
            
    print(f"Generated haystack with needle at position approx {needle_pos}")
    return needle

def run_test():
    haystack_file = "data/haystack.txt"
    # Generate 150k tokens to test 50k chunking (should create ~3 chunks)
    needle = generate_haystack(haystack_file, size_in_tokens=150000) 
    
    print("--- Starting Indexing ---")
    start_time = time.time()
    db_path = "data/s_niah.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    indexer = Indexer(db_path)
    
    # Use Large Scale settings: 50k tokens per chunk, group 2 chunks for summary (100k context)
    indexer.ingest_file(haystack_file, target_chunk_tokens=25000, group_size=2)
    print(f"Indexing took {time.time() - start_time:.2f}s")
    
    print("--- Starting Agent Search ---")
    # Use factory to create agent with NO HISTORY
    agent = AgentFactory.create_agent("rlm-agent", session_id="test_niah", add_history_to_context=False, read_chat_history=False)
    response = agent.run("Find the 'secret code' mentioned in the document and return it.")
    
    print("\nAgent Response:")
    print(response.content)
    
    if str(needle).split(": ")[1] in response.content:
        print("\nSUCCESS: Needle found!")
    else:
        print("\nFAILURE: Needle not found.")

if __name__ == "__main__":
    run_test()
