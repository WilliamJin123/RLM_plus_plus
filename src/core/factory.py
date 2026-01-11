from pathlib import Path
from typing import Optional, Dict, Any, cast
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.tools.registry import registry
from src.core.monitor_bus import monitored_tool
from src.config.config import config
import inspect

from keycycle import MultiProviderWrapper

ModelSettings = Dict[str, Any]
StorageSettings = Dict[str, Any]

class AgentFactory:
    @staticmethod
    def create_model(model_settings: Optional[ModelSettings] = None):
        if model_settings:
            provider = model_settings.get("provider", config.FAST_MODEL_PROVIDER)
            model_id = model_settings.get("model_id", config.FAST_MODEL_NAME)
            temperature = model_settings.get("temperature", 0.0)
        else:
            provider = config.FAST_MODEL_PROVIDER
            model_id = config.FAST_MODEL_NAME
            temperature = 0.0
        
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
        from src.config.yaml_config import get_agent_config
        
        config_record = get_agent_config(agent_id)
        if not config_record:
            raise ValueError(f"No configuration found for agent_id: {agent_id}")

        # Resolve Tools
        tool_map = registry.get_tool_map()
        agent_tools = []
        tool_names = cast(list, config_record.tools)
        for tool_name in tool_names:
            if tool_name in tool_map:
                tool_obj = tool_map[tool_name]
                # Wrap callable tools (functions) with monitoring
                if inspect.isfunction(tool_obj):
                    # monitor_bus.monitored_tool uses functools.wraps, so metadata is preserved.
                    wrapped_tool = monitored_tool(tool_obj)
                    agent_tools.append(wrapped_tool)
                else:
                    agent_tools.append(tool_obj)
            else:
                print(f"Warning: Tool '{tool_name}' not found in registry.")

        # Resolve Model
        model_settings = cast(ModelSettings, config_record.model_settings)
        model = AgentFactory.create_model(model_settings)

        # Resolve Storage
        storage_settings = cast(Optional[StorageSettings], config_record.storage_settings)
        
        agent_db = None
        use_history = False
        read_history = False
        num_history = 0

        if storage_settings:
            db_path = storage_settings.get("db_path", "data/history.db")
            session_table = storage_settings.get("session_table", "sessions")
            
            # Determine history settings (override > config > default)
            use_history = add_history_to_context if add_history_to_context is not None else storage_settings.get("add_history_to_context", True)
            read_history = read_chat_history if read_chat_history is not None else storage_settings.get("read_chat_history", False)
            num_history = storage_settings.get("num_history_runs", 5)

            if db_path and session_table:
                agent_db = SqliteDb(
                    db_file=db_path,
                    session_table=session_table
                )
        
        # If manual overrides provided but no storage config, we might want to respect them if possible,
        # but Agno usually needs a DB for history.
        if add_history_to_context is not None:
            use_history = add_history_to_context
        if read_chat_history is not None:
            read_history = read_chat_history

        # Create Agent
        agent = Agent(
            id=agent_id,
            name=agent_id.replace("-", " ").title(),
            model=model,
            tools=agent_tools,
            instructions=config_record.instructions,
            db=agent_db,
            add_history_to_context=use_history,
            num_history_runs=num_history,
            read_chat_history=read_history,
            session_id=session_id,
            markdown=True
        )

        # Setup Tracing
        if agent_db:
            setup_tracing(db=agent_db, batch_processing=True)

        return agent