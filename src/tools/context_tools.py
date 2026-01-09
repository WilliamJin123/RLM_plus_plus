from typing import List, Dict, Any, cast
from pathlib import Path
from sqlalchemy import create_engine, text
from src.core.config_store import SessionLocal, AgentConfig, StorageSettings

def get_agent_history(agent_id: str, last_n: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves the last n interactions for a given agent_id.
    Useful for the Architect to analyze agent performance.
    """
    # 1. Get storage settings for the agent
    db_session = SessionLocal()
    try:
        config = db_session.query(AgentConfig).filter_by(agent_id=agent_id).first()
        if not config:
            return [{"error": f"Agent {agent_id} not found in config."}]
        
        storage_settings = cast(StorageSettings, config.storage_settings)
        db_path = storage_settings.get("db_path")
        table_name = storage_settings.get("session_table")
        
        if not db_path or not table_name:
            return [{"error": f"Storage not configured for {agent_id}."}]
            
    finally:
        db_session.close()

    # 2. Connect to the agent's history DB
    # Ensure path is relative to project root if it's relative
    db_file = Path(db_path)
    if not db_file.is_absolute():
        db_file = Path.cwd() / db_path
        
    if not db_file.exists():
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