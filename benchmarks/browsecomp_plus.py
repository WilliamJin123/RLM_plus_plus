import os
import time
import sys
from pathlib import Path

# Ensure src is in path if running from root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR.as_posix())

from src.core.indexer import Indexer
from src.core.factory import AgentFactory
from src.tools.rlm_tools import RLMTools

def generate_browsecomp_data(file_path: Path):
    content = """
    <html>
    <body>
    <h1>Product Comparison: Smartphones 2024</h1>
    
    <div class="product">
        <h2>Phone A (Model X)</h2>
        <p>Price: $799</p>
        <p>Specs: 6.5 inch screen, 128GB storage</p>
        <p>Camera Rating: 8.5/10</p>
        <p>Battery: 4000mAh</p>
    </div>

    <div class="product">
        <h2>Phone B (Model Y)</h2>
        <p>Price: $999</p>
        <p>Specs: 6.7 inch screen, 256GB storage</p>
        <p>Camera Rating: 9.5/10</p>
        <p>Battery: 5000mAh</p>
    </div>

    <div class="product">
        <h2>Phone C (Model Z)</h2>
        <p>Price: $650</p>
        <p>Specs: 6.1 inch screen, 128GB storage</p>
        <p>Camera Rating: 7.0/10</p>
        <p>Battery: 3500mAh</p>
    </div>
    
    <div class="product">
        <h2>Phone D (Model Ultra)</h2>
        <p>Price: $750</p>
        <p>Specs: 6.4 inch screen, 256GB storage</p>
        <p>Camera Rating: 9.0/10</p>
        <p>Battery: 4500mAh</p>
    </div>

    </body>
    </html>
    """
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Generated BrowseComp data at {file_path}")

def run_test():
    data_file = BASE_DIR / "data" / "browsecomp.html" # Using html extension just for flavor, indexer reads as text
    generate_browsecomp_data(data_file)
    
    print("--- Starting Indexing ---")
    db_path = BASE_DIR / "data" / "browsecomp_plus.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    indexer = Indexer(str(db_path))
    # Use Large Scale settings (mocking a large file scenario)
    # Changed target_chunk_tokens to max_chunk_tokens to match current implementation
    indexer.ingest_file(str(data_file), max_chunk_tokens=25000, group_size=2)
    
    print("--- Starting Agent Search ---")
    # Updated: removed session_id and history args from create_agent as they are not supported in factory.py
    agent = AgentFactory.create_agent("rlm-agent")
    
    # Manually override tools to use the benchmark DB
    # Filter out existing RLMTools and add new one pointing to our DB
    new_tools = [t for t in agent.tools if not isinstance(t, RLMTools)]
    new_tools.append(RLMTools(db_path=str(db_path)))
    agent.tools = new_tools
    
    # Set session ID explicitly on the agent instance
    agent.session_id = "test_browsecomp"
    
    query = "Which phone has the best camera rating for under $800?"
    print(f"Query: {query}")
    response = agent.run(query)
    
    print("\nAgent Response:")
    print(response.content)

if __name__ == "__main__":
    run_test()