import pytest
import os
from pathlib import Path
from unittest.mock import patch
from src.core.indexer import Indexer
from src.core.storage import storage
from src.config.yaml_config import load_agents_config, AgentConfig

# Define the models requested
GROQ_MODEL = "llama-3.1-8b-instant"
CEREBRAS_MODEL = "llama3.1-8b"

def get_patched_agent_config(agent_id: str):
    """
    Custom side_effect for get_agent_config that overrides models
    for specific agents to use the requested toy models.
    """
    # Load original configs first
    configs = load_agents_config()
    config = configs.get(agent_id)
    
    if not config:
        return None

    # Override for Smart Ingest Agent (Groq)
    if agent_id == "smart-ingest-agent":
        print(f"DEBUG: Patching {agent_id} to use Groq/{GROQ_MODEL}")
        config.model_settings["provider"] = "groq"
        config.model_settings["model_id"] = GROQ_MODEL
        config.storage_settings = None # Ensure no DB requirement for this helper agent

    # Override for Summarization Agent (Cerebras)
    elif agent_id == "summarization-agent":
        print(f"DEBUG: Patching {agent_id} to use Cerebras/{CEREBRAS_MODEL}")
        config.model_settings["provider"] = "cerebras"
        config.model_settings["model_id"] = CEREBRAS_MODEL
        # Use a memory-based DB or temp file for the agent's history if needed, 
        # but the indexer manages the main DB. 
        # The agent config "storage" is for chat history. 
        # We can disable history for this test to be safe/clean.
        config.storage_settings = None

    return config

@pytest.mark.integration
@patch('src.config.yaml_config.get_agent_config', side_effect=get_patched_agent_config)
def test_smart_ingestion_toy_models(mock_get_config, tmp_path):
    """
    Verifies that smart ingestion works using Groq and Cerebras models.
    """
    # 1. Setup Environment
    # Ensure keys are present (will fail if not, which is expected for verify task)
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not found in environment.")
    if not os.getenv("CEREBRAS_API_KEY"):
        pytest.skip("CEREBRAS_API_KEY not found in environment.")

    # 2. Setup DB and File
    # Indexer uses a global or passed DB path. 
    # We can pass db_path to Indexer.
    db_path = tmp_path / "test_ingestion.db"
    
    # Create a dummy text file
    # We want enough text to trigger chunking. 
    # Target chunk is small (e.g. 50 tokens ~ 200 chars).
    # We'll create ~1000 chars.
    text_file = tmp_path / "toy_data.txt"
    content = (
        "The history of artificial intelligence is a history of fantasies, possibilities, "
        "demonstrations, and promises. Ever since Homer wrote of mechanical 'tripods' waiting "
        "on the gods at dinner, imaginary mechanical assistants have been part of our culture. "
        "However, only in the last half century have we, the AI community, been able to build "
        "experimental machines that test these hypotheses.\n\n"
        "AI has followed a path of distinct eras. The first era, from the 1950s to the 1970s, "
        "was the era of reasoning and search. It was believed that intelligence could be "
        "captured by symbolic manipulation and logical deduction. Programs like the Logic Theorist "
        "and the General Problem Solver were born.\n\n"
        "The second era, starting in the 1980s, was the era of knowledge. Expert systems "
        "attempted to capture the specific knowledge of human experts in rules. This led to "
        "commercial success but also the 'AI Winter' when promises were not met.\n\n"
        "The third era, the era of learning, began in earnest in the 1990s and 2000s, "
        "driven by statistical methods and later, deep learning. This is the era we are "
        "currently in, where data is king and neural networks rule.\n\n"
        "Future eras may bring us to Artificial General Intelligence, or they may lead "
        "us down new paths we cannot yet imagine. One thing is certain: the quest to "
        "understand intelligence is one of the most profound endeavors of science."
    )
    text_file.write_text(content, encoding='utf-8')

    # 3. Run Indexer
    # We patch the DB path used by SessionLocal if possible, 
    # but Indexer.__init__ calls init_db(db_path).
    # AND SessionLocal is imported from src.core.db.
    # src.core.indexer.init_db configures the engine that SessionLocal uses.
    # So passing db_path to Indexer should work.
    
    print("\n--- Starting Ingestion Test ---")
    indexer = Indexer(db_path=str(db_path))
    
    # Target ~50 tokens per chunk to force multiple chunks (text is ~200 tokens)
    # Group size 2 to trigger summarization quickly.
    indexer.ingest_file(str(text_file), target_chunk_tokens=50, group_size=2)
    
    # 4. Verification
    # Connect to DB and check results via sqlite3 directly
    import sqlite3
    import pandas as pd
    
    conn = sqlite3.connect(str(db_path))
    try:
        chunks = pd.read_sql_query("SELECT * FROM chunks", conn)
        summaries = pd.read_sql_query("SELECT * FROM summaries", conn)
        
        print(f"\nChunks Found: {len(chunks)}")
        for _, c in chunks.iterrows():
            print(f"- Chunk [{c['start_index']}:{c['end_index']}] ({len(c['text'])} chars)")
            
        print(f"\nSummaries Found: {len(summaries)}")
        for _, s in summaries.iterrows():
            print(f"- Level {s['level']} Summary: {s['summary_text'][:50]}...")

        # Assertions
        assert len(chunks) > 1, "Should have created multiple chunks"
        assert len(summaries) > 0, "Should have created summaries"
        
        # Verify Smart Ingest Logic
        chunks = chunks.sort_values("start_index").to_dict('records')
        
        assert chunks[0]['start_index'] == 0
        # Check continuity
        for i in range(len(chunks) - 1):
            assert chunks[i+1]['start_index'] <= chunks[i]['end_index']
            
    finally:
        conn.close()

if __name__ == "__main__":
    # Manually run the test function if executed directly
    # But usually via pytest
    pass
