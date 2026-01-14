import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing
from keycycle import MultiProviderWrapper

from src.config.config import ModelConfig, ModelPoolConfig, CONFIG
from src.tools.rlm_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class ModelRotator:
    """Thread-safe round-robin model rotator with configurable calls per model."""

    def __init__(self, configs: List[ModelConfig], calls_per_model: int = 3):
        self._configs = configs
        self._calls_per_model = calls_per_model
        self._index = 0
        self._call_count = 0
        self._lock = threading.Lock()

    def get_next_config(self) -> ModelConfig:
        """Get the next model config, rotating after calls_per_model calls."""
        with self._lock:
            config = self._configs[self._index]
            self._call_count += 1
            if self._call_count >= self._calls_per_model:
                self._call_count = 0
                self._index = (self._index + 1) % len(self._configs)
                logger.debug(
                    "Rotating to model %s/%s",
                    self._index + 1,
                    len(self._configs),
                )
            return config

    def force_rotate(self) -> None:
        """Force immediate rotation to next model (used on failure)."""
        with self._lock:
            self._call_count = 0
            self._index = (self._index + 1) % len(self._configs)
            next_config = self._configs[self._index]
            logger.warning(
                "Forced rotation due to failure. Now on model %d/%d: %s/%s",
                self._index + 1,
                len(self._configs),
                next_config.provider,
                next_config.model_id,
            )

    def __len__(self) -> int:
        return len(self._configs)


class AgentFactory:
    _wrapper_cache: Dict[str, MultiProviderWrapper] = {}
    _cache_lock = threading.Lock()

    @classmethod
    def _get_cached_wrapper(cls, provider: str) -> MultiProviderWrapper:
        """Retrieves a wrapper from cache or creates a new one if it doesn't exist."""
        with cls._cache_lock:
            if provider not in cls._wrapper_cache:
                logger.info("Initializing new MultiProviderWrapper for %s", provider)
                cls._wrapper_cache[provider] = MultiProviderWrapper.from_env(
                    provider=provider,
                    default_model_id=None,
                    env_file=Path(__file__).resolve().parents[2] / ".env",
                )
        return cls._wrapper_cache[provider]

    @staticmethod
    def create_model(
        model_settings: ModelConfig,
        estimated_tokens: int = 4000,
        key_index: Optional[int] = None,
    ):
        wrapper = AgentFactory._get_cached_wrapper(provider=model_settings.provider)

        kwargs = {
            "estimated_tokens": estimated_tokens,
            "id": model_settings.model_id,
            "temperature": model_settings.temperature,
            "pin_key": False,
        }
        if key_index is not None:
            kwargs["key_id"] = key_index

        return wrapper.get_model(**kwargs)

    @staticmethod
    def _hydrate_tools(tool_names: List[str], content_db_path: Optional[str] = None) -> list:
        """Converts a list of string tool names into initialized Toolkit objects."""
        if not tool_names:
            return []

        hydrated_tools = []
        for name in tool_names:
            tool_cls = TOOL_REGISTRY.get(name)
            if not tool_cls:
                logger.warning("Tool '%s' not found in registry. Skipping.", name)
                continue

            if name == "RLMTools":
                if content_db_path:
                    hydrated_tools.append(tool_cls(db_path=content_db_path))
                else:
                    logger.warning("RLMTools requires content_db_path. Skipping.")
            else:
                hydrated_tools.append(tool_cls())

        return hydrated_tools

    @staticmethod
    def create_agent(
        agent_id: str,
        session_id: Optional[str] = None,
        content_db_path: Optional[str] = None,
        estimated_tokens: int = 1000,
        key_index: Optional[int] = None,
    ) -> Agent:
        config_record = CONFIG.get_agent(agent_id)
        if not config_record:
            raise ValueError(f"No configuration found for agent_id: {agent_id}")

        # Handle single model or pick first from rotation pool
        if config_record.model_settings:
            model = AgentFactory.create_model(
                config_record.model_settings,
                estimated_tokens,
                key_index,
            )
        elif config_record.model_pool:
            # Use first model from pool as default
            model = AgentFactory.create_model(
                config_record.model_pool.models[0],
                estimated_tokens,
                key_index,
            )
        else:
            raise ValueError(f"No model configuration found for agent_id: {agent_id}")

        project_root = Path(__file__).resolve().parent.parent.parent
        storage_settings = config_record.storage_settings

        agent_db = None
        add_history_to_context = False
        num_history_runs = None
        read_chat_history = False

        if storage_settings and storage_settings.db_path:
            agent_db = SqliteDb(
                db_file=str(project_root / storage_settings.db_path),
                session_table=storage_settings.session_table,
            )
            setup_tracing(db=agent_db, batch_processing=True)
            add_history_to_context = storage_settings.add_history_to_context
            num_history_runs = storage_settings.num_history_runs
            read_chat_history = storage_settings.read_chat_history

        tools = AgentFactory._hydrate_tools(config_record.tools, content_db_path)

        agent_kwargs = {
            "id": agent_id,
            "name": agent_id.replace("-", " ").title(),
            "model": model,
            "tools": tools,
            "instructions": config_record.instructions,
            "add_history_to_context": add_history_to_context,
            "read_chat_history": read_chat_history,
            "markdown": True,
        }

        if agent_db:
            agent_kwargs["db"] = agent_db
        if session_id:
            agent_kwargs["session_id"] = session_id
        if num_history_runs:
            agent_kwargs["num_history_runs"] = num_history_runs

        return Agent(**agent_kwargs)

    @staticmethod
    def create_rotating_agent(
        agent_id: str,
        session_id: Optional[str] = None,
        content_db_path: Optional[str] = None,
        estimated_tokens: int = 1000,
    ) -> Tuple[Agent, ModelRotator]:
        """
        Create an agent with a ModelRotator for model rotation.

        Returns a tuple of (Agent, ModelRotator). The caller should use the rotator
        to get new model configs and update agent.model before each call.
        """
        config_record = CONFIG.get_agent(agent_id)
        if not config_record:
            raise ValueError(f"No configuration found for agent_id: {agent_id}")

        if not config_record.model_pool:
            raise ValueError(
                f"Agent '{agent_id}' does not have model rotation configured. "
                "Use 'models' list in config instead of 'model'."
            )

        rotator = ModelRotator(
            configs=config_record.model_pool.models,
            calls_per_model=config_record.model_pool.calls_per_model,
        )

        logger.info(
            "Created ModelRotator for '%s' with %d models, %d calls per model",
            agent_id,
            len(rotator),
            config_record.model_pool.calls_per_model,
        )

        # Create agent with first model
        agent = AgentFactory.create_agent(
            agent_id=agent_id,
            session_id=session_id,
            content_db_path=content_db_path,
            estimated_tokens=estimated_tokens,
        )

        return agent, rotator
