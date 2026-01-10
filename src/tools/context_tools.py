from typing import List, Dict, Any, cast
from pathlib import Path
from sqlalchemy import create_engine, text
from agno.agent import Agent
from src.core.factory import AgentFactory
from src.config.yaml_config import get_agent_config

def _get_raw_agent_history(agent_id: str, last_n: int = 10) -> List[Dict[str, Any]]:
    """
    Internal helper: Retrieves the last n interactions for a given agent_id.
    """
    # 1. Get storage settings for the agent using YAML config
    config = get_agent_config(agent_id)
    if not config:
        return [{"error": f"Agent {agent_id} not found in config."}]
    
    storage_settings = config.storage_settings
    db_path = storage_settings.get("db_path")
    table_name = storage_settings.get("session_table")
    
    if not db_path or not table_name:
        return [{"error": f"Storage not configured for {agent_id}."}]

    # 2. Connect to the agent's history DB
    # Ensure path is relative to project root if it's relative
    db_file = Path(db_path)
    if not db_file.is_absolute():
        # Assuming we are running from root, but we can't be sure.
        # However, AgentFactory uses standard logic.
        # Let's resolve against project root (assuming CWD is root or close to it)
        db_file = Path.cwd() / db_path
        
    if not db_file.exists():
        # Try relative to this file? No, usually relative to CWD.
        return [{"error": f"History DB file not found at {db_file}"}]

    try:
        engine = create_engine(f"sqlite:///{db_file}")
        with engine.connect() as conn:
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
        sub_agent = Agent(
            model=AgentFactory.create_model(), 
            description="You are an expert system analyst.",
            instructions="Analyze the provided agent interaction logs and answer the user's question.",
            markdown=True
        )
        
        prompt = f"Agent History ({agent_id}):\n{history_text}\n\nQuestion: {query}"
        response = sub_agent.run(prompt)
        return str(response.content)
    except Exception as e:
        return f"Error analyzing history: {str(e)}"
