from pathlib import Path
from typing import Optional, Dict, Any, cast
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.config.config import ModelConfig, config
from src.tools.registry import registry

import inspect

from keycycle import MultiProviderWrapper


class AgentFactory:

    _wrapper_cache: Dict[str, MultiProviderWrapper] = {}

    @classmethod
    def _get_cached_wrapper(cls, provider: str) -> MultiProviderWrapper:
        """
        Retrieves a wrapper from cache or creates a new one if it doesn't exist.
        """

        if provider not in cls._wrapper_cache:
            print(f"Initializing new MultiProviderWrapper for {provider}")
            cls._wrapper_cache[provider] = MultiProviderWrapper.from_env(
                provider=provider,
                default_model_id=None, 
                env_file=Path(__file__).resolve().parents[2] / ".env",
            )
        
        return cls._wrapper_cache[provider]


    @staticmethod
    def create_model(model_settings: ModelConfig):
        wrapper = AgentFactory._get_cached_wrapper(
            provider=model_settings.provider,
        )

        # Agno model kwargs        
        return wrapper.get_model(
            id=model_settings.model_id,
            temperature=model_settings.temperature,
        )
    
    @staticmethod
    def create_agent(agent_id: str, session_id: str = None) -> Agent:
        
        
        config_record = config.get_agent(agent_id)
        if not config_record:
            raise ValueError(f"No configuration found for agent_id: {agent_id}")

        # Resolve Tools
        tool_map = registry.get_tool_map()
        agent_tools = []
        tool_names = cast(list, config_record.tools)
        for tool_name in tool_names:
            if tool_name in tool_map:
                tool_obj = tool_map[tool_name]
                agent_tools.append(tool_obj)
            else:
                print(f"Warning: Tool '{tool_name}' not found in registry.")

        model = AgentFactory.create_model(config_record.model_settings)

        project_root = Path(__file__).resolve().parent.parent.parent
        default_db_path = str(project_root / "data" / "history.db")

        storage_settings = config_record.storage_settings
        
        if storage_settings:
            db_path=storage_settings.db_path
            session_table=storage_settings.session_table
            add_history_to_context = storage_settings.add_history_to_context
            num_history_runs = storage_settings.num_history_runs
            read_chat_history = storage_settings.read_chat_history
        else:
            db_path = default_db_path
            session_table = "sessions"
            add_history_to_context = False
            num_history_runs = 0
            read_chat_history = False

        agent_db = SqliteDb(
            db_path= project_root / db_path, 
            session_table=session_table
        )
        setup_tracing(db=agent_db, batch_processing=True)

        # Create Agent
        agent = Agent(
            id=agent_id,
            name=agent_id.replace("-", " ").title(),
            model=model,
            tools=agent_tools,
            instructions=config_record.instructions,
            db=agent_db,
            add_history_to_context=add_history_to_context,
            num_history_runs=num_history_runs,
            read_chat_history=read_chat_history,
            markdown=True,
            **({"session_id": session_id} if session_id else {})
        )

        return agent