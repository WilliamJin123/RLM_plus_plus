from pathlib import Path
from typing import List, Optional, Dict, Any, cast
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing

from src.config.config import ModelConfig, CONFIG
from src.tools.rlm_tools import TOOL_REGISTRY

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
    def _hydrate_tools(tool_names: List[str], content_db_path: str = None) -> list:
        """
        Converts a list of string tool names into initialized Toolkit objects.
        """
        hydrated_tools = []
        
        if not tool_names:
            return hydrated_tools

        for name in tool_names:
            tool_cls = TOOL_REGISTRY.get(name)
            if not tool_cls:
                print(f"Warning: Tool '{name}' not found in registry. Skipping.")
                continue

            # --- Logic to initialize specific tools ---
            if name == "RLMTools":
                # RLMTools requires the path to the ingest DB (content)
                hydrated_tools.append(tool_cls(db_path=content_db_path))
            elif name == "PythonTools":
                # PythonTools usually requires no args, or specific permissions
                hydrated_tools.append(tool_cls())
            else:
                # Generic fallback for tools with no args
                hydrated_tools.append(tool_cls())
        
        return hydrated_tools

    @staticmethod
    def create_agent(agent_id: str, session_id: str = None, content_db_path: str = None) -> Agent:
        
        config_record = CONFIG.get_agent(agent_id)
        if not config_record:
            raise ValueError(f"No configuration found for agent_id: {agent_id}")

        model = AgentFactory.create_model(config_record.model_settings)
        tools = config_record.tools

        project_root = Path(__file__).resolve().parent.parent.parent
        default_db_path = "data/history.db"

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

        tools = AgentFactory._hydrate_tools(config_record.tools, content_db_path)

        agent_db = SqliteDb(
            db_file=str(project_root / db_path),
            session_table=session_table
        )
        setup_tracing(db=agent_db, batch_processing=True)

        # Create Agent
        agent = Agent(
            id=agent_id,
            name=agent_id.replace("-", " ").title(),
            model=model,
            tools=tools,
            instructions=config_record.instructions,
            db=agent_db,
            add_history_to_context=add_history_to_context,
            num_history_runs=num_history_runs,
            read_chat_history=read_chat_history,
            markdown=True,
            **({'session_id': session_id} if session_id else {})
        )

        return agent