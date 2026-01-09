from sqlalchemy import Column, String, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path

# Create a separate database for configuration
CONFIG_DB_PATH = Path("data/config.db")
CONFIG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

Base = declarative_base()

class AgentConfig(Base):
    __tablename__ = 'agent_configs'
    
    agent_id = Column(String, primary_key=True)
    instructions = Column(JSON, default=list) # List of strings
    tools = Column(JSON, default=list) # List of tool names (strings)
    model_settings = Column(JSON, default=dict) # Provider, model name, params
    storage_settings = Column(JSON, default=dict) # History settings

engine = create_engine(f"sqlite:///{CONFIG_DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_config_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_config_db()
    print(f"Config Database initialized at {CONFIG_DB_PATH}")
