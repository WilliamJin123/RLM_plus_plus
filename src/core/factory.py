from pathlib import Path
from typing import Optional, Dict, Any, cast
from sqlalchemy.orm import Session
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.core.config_store import AgentConfig, SessionLocal, ModelSettings, StorageSettings
from src.tools.registry import registry

# Try importing MultiProviderWrapper, handle if missing
try:
    from keycycle import MultiProviderWrapper
except ImportError:
    MultiProviderWrapper = None # type: ignore

class AgentFactory:
    @staticmethod
    def _create_model(model_settings: ModelSettings):
        provider = model_settings.get("provider", "Cerebras")
        model_id = model_settings.get("model_id", "zai-glm-4.7")
        temperature = model_settings.get("temperature", 0.0)
        
        wrapper = MultiProviderWrapper.from_env(
            provider=provider,
            default_model_id=model_id,
            env_file=Path(__file__).resolve().parent.parent / ".env",
            temperature=temperature
        )
        model = wrapper.get_model()
                    
        return model
    
    @staticmethod
    def create_agent(agent_id: str, session_id: str = None, add_history_to_context: bool = None, read_chat_history: bool = None) -> Agent:
        db: Session = SessionLocal()
        try:
            config_record = db.query(AgentConfig).filter_by(agent_id=agent_id).first()
            if not config_record:
                raise ValueError(f"No configuration found for agent_id: {agent_id}")

            # Resolve Tools
            tool_map = registry.get_tool_map()
            agent_tools = []
            tool_names = cast(list, config_record.tools)
            for tool_name in tool_names:
                if tool_name in tool_map:
                    agent_tools.append(tool_map[tool_name])
                else:
                    print(f"Warning: Tool '{tool_name}' not found in registry.")

            # Resolve Model
            model_settings = cast(ModelSettings, config_record.model_settings)
            model = AgentFactory._create_model(model_settings)

            # Resolve Storage
            storage_settings = cast(StorageSettings, config_record.storage_settings)
            
            db_path = storage_settings.get("db_path", "data/history.db")
            session_table = storage_settings.get("session_table", "sessions")
            
            # Determine history settings (override > config > default)
            use_history = add_history_to_context if add_history_to_context is not None else storage_settings.get("add_history_to_context", True)
            read_history = read_chat_history if read_chat_history is not None else storage_settings.get("read_chat_history", False)

            agent_db = SqliteDb(
                db_file=db_path,
                session_table=session_table
            )

            # Create Agent
            agent = Agent(
                id=agent_id,
                name=agent_id.replace("-", " ").title(),
                model=model,
                tools=agent_tools,
                instructions=config_record.instructions,
                db=agent_db,
                add_history_to_context=use_history,
                num_history_runs=storage_settings.get("num_history_runs", 5),
                read_chat_history=read_history,
                session_id=session_id,
                markdown=True
            )

            # Setup Tracing
            setup_tracing(db=agent_db, batch_processing=True)

            return agent
            
        finally:
            db.close()
