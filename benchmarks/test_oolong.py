import os
import random
import time
from src.core.indexer import Indexer
from src.core.agent import RLMAgent
from src.core.db import init_db

def generate_oolong_data(file_path: str, num_entries: int = 100):
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
    locations = ["Paris", "London", "New York", "Tokyo", "Berlin", "Rome", "Madrid"]
    
    # We want at least one pair in Paris at the same time
    # Alice and Bob in Paris on 2024-05-10
    entries = [
        "Alice was in Paris on 2024-05-10.",
        "Bob was in Paris on 2024-05-10.",
        "Charlie was in London on 2024-05-10.",
        "David was in Paris on 2024-06-15.",
        "Eve was in Rome on 2024-05-10."
    ]
    
    # Add random filler
    for _ in range(num_entries - len(entries)):
        name = random.choice(names)
        loc = random.choice(locations)
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        entries.append(f"{name} was in {loc} on 2024-{month:02d}-{day:02d}.")
        
    random.shuffle(entries)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(entries))
    
    print(f"Generated OOLONG data with {num_entries} entries.")

def run_test():
    data_file = "data/oolong.txt"
    generate_oolong_data(data_file, num_entries=50)
    
    print("---" + " Starting Indexing" + " ---")
    indexer = Indexer()
    # Reset DB for test
    if os.path.exists("data/rlm.db"):
        os.remove("data/rlm.db")
    init_db()
    
    indexer.ingest_file(data_file, chunk_size=500, overlap=100)
    
    print("---" + " Starting Agent Search" + " ---")
    agent = RLMAgent()
    query = "Find all pairs of people who were in Paris at the same time. Check the dates carefully."
    response = agent.run(query)
    
    print("\nAgent Response:")
    print(response.content)

if __name__ == "__main__":
    run_test()
