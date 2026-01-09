import os
import time
from src.core.indexer import Indexer
from src.core.agent import RLMAgent
from src.core.db import init_db

def ingest_codebase(indexer: Indexer, root_dir: str):
    for root, dirs, files in os.walk(root_dir):
        if ".git" in root or "__pycache__" in root or ".venv" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Indexing {file_path}...")
                indexer.ingest_file(file_path)

def run_test():
    print("--- Starting Indexing ---")
    indexer = Indexer()
    # Reset DB for test
    if os.path.exists("data/rlm.db"):
        os.remove("data/rlm.db")
    init_db()
    
    # Ingest the src directory of this project
    ingest_codebase(indexer, "src")
    
    print("--- Starting Agent Search ---")
    agent = RLMAgent()
    query = "Explain how the 'Summary' table in the database is used to build a hierarchical index. Which file defines this?"
    response = agent.run(query)
    
    print("\nAgent Response:")
    print(response.content)

if __name__ == "__main__":
    run_test()
