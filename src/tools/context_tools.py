from typing import List, Dict, Any, cast
from pathlib import Path
from sqlalchemy import create_engine, text
from agno.agent import Agent
from src.core.factory import AgentFactory
from src.config.yaml_config import get_agent_config

_engine_cache = {}

def _get_engine(db_path: str):
    if db_path not in _engine_cache:
        # Check if path is valid or needs resolution
        # We assume caller has resolved it or we resolve it here.
        # But for caching key, we should use resolved path.
        _engine_cache[db_path] = create_engine(f"sqlite:///{db_path}")
    return _engine_cache[db_path]

def _get_raw_agent_history(agent_id: str, last_n: int = 10) -> List[Dict[str, Any]]:
    """
    Internal helper: Retrieves the last n interactions for a given agent_id.
    """
    # 1. Get storage settings for the agent using YAML config
    config = get_agent_config(agent_id)
    if not config:
        return [{"error": f"Agent {agent_id} not found in config."}]
    
    storage_settings = config.storage_settings
    if not storage_settings:
        return [{"error": f"Storage not configured for {agent_id}."}]

    db_path = storage_settings.get("db_path")
    table_name = storage_settings.get("session_table")
    
    if not db_path or not table_name:
        return [{"error": f"Storage not configured for {agent_id}."}]

    # 2. Connect to the agent's history DB
    # Ensure path is relative to project root if it's relative
    db_file = Path(db_path)
    if not db_file.is_absolute():
        db_file = Path.cwd() / db_path
        
    if not db_file.exists():
        # Even if it doesn't exist, create_engine won't fail immediately, but connect will if file is missing?
        # Actually sqlite will create file if not exists usually, but here we are reading history.
        return [{"error": f"History DB file not found at {db_file}"}]

    try:
        engine = _get_engine(str(db_file))
        with engine.connect() as conn:
            # Check if table exists first to avoid error?
            # Or just try-catch.
            query = text(f"SELECT * FROM {table_name} ORDER BY updated_at DESC LIMIT :limit")
            result = conn.execute(query, {"limit": last_n})
            rows = [dict(row._mapping) for row in result]
            return rows
            
    except Exception as e:
        return [{"error": f"Failed to read history: {e}"}]

def analyze_agent_history(agent_id: str, query: str, last_n: int = 10) -> str:
    """
    Analyzes the recent history of an agent using a sub-agent to prevent context overflow.
    
    Args:
        agent_id: The ID of the agent to analyze (e.g., 'rlm-agent').
        query: The specific question about the agent's history (e.g., "Why did it fail?", "Summarize the last run").
        last_n: Number of recent interactions to check. Defaults to 10.
    """
    history_rows = _get_raw_agent_history(agent_id, last_n)
    
    if not history_rows:
        return f"No history found for {agent_id}."
    
    if "error" in history_rows[0]:
        return f"Error retrieving history: {history_rows[0]['error']}"
        
    # Convert history rows to a text block
    history_text = ""
    for row in reversed(history_rows): # Chronological order for reading
        history_text += f"--- Interaction ---\n{str(row)}\n"

    try:
        # Spawn a lightweight sub-agent
        sub_agent = AgentFactory.create_agent("history-analyzer-agent")
        
        prompt = f"Agent History ({agent_id}):\n{history_text}\n\nQuestion: {query}"
        response = sub_agent.run(prompt)
        return str(response.content)
    except Exception as e:
        return f"Error analyzing history: {str(e)}"
