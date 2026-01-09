from typing import TypedDict, List, Dict, Any, Optional
from sqlalchemy import Column, String, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path
from src.config import config

# Create a separate database for configuration
CONFIG_DB_PATH = Path(config.CONFIG_DB_PATH)
CONFIG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

Base = declarative_base()

class ModelSettings(TypedDict, total=False):
    provider: str
    model_id: str
    temperature: float
    max_tokens: Optional[int]

class StorageSettings(TypedDict, total=False):
    db_path: str
    session_table: str
    add_history_to_context: bool
    num_history_runs: int
    read_chat_history: bool

class AgentConfig(Base):
    __tablename__ = 'agent_configs'
    
    agent_id = Column(String, primary_key=True)
    instructions = Column(JSON, default=list) # List[str]
    tools = Column(JSON, default=list) # List[str]
    model_settings = Column(JSON, default=dict) # ModelSettings
    storage_settings = Column(JSON, default=dict) # StorageSettings

engine = create_engine(f"sqlite:///{CONFIG_DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_config_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_config_db()
    print(f"Config Database initialized at {CONFIG_DB_PATH}")