from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    temperature: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
        }


@dataclass
class StorageConfig:
    db_path: str
    session_table: str
    add_history_to_context: bool = True
    num_history_runs: int = 5
    read_chat_history: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "session_table": self.session_table,
            "add_history_to_context": self.add_history_to_context,
            "num_history_runs": self.num_history_runs,
            "read_chat_history": self.read_chat_history,
        }


@dataclass
class AgentConfig:
    agent_id: str
    instructions: List[str]
    tools: List[str]
    model_settings: ModelConfig
    storage_settings: Optional[StorageConfig]


class AgentConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self._config_cache: Dict[str, AgentConfig] = {}
        self._last_mtime: float = 0.0

        if config_path:
            self.file_path = Path(config_path)
        else:
            self.file_path = Path(__file__).resolve().parent / "agents.yaml"

    def _load_if_needed(self) -> None:
        if not self.file_path.exists():
            self._config_cache = {}
            self._last_mtime = 0
            raise FileNotFoundError(f"Configuration file {self.file_path} not found.")

        current_mtime = self.file_path.stat().st_mtime

        if self._config_cache and current_mtime == self._last_mtime:
            return

        with open(self.file_path, "r", encoding="utf-8") as f:
            data: Dict[str, Dict[str, Any]] = yaml.safe_load(f) or {}

        configs: Dict[str, AgentConfig] = {}
        for agent_id, agent_data in data.items():
            model_data = ModelConfig(**agent_data["model"])

            storage_data = None
            storage_dict = agent_data.get("storage")
            if storage_dict:
                storage_data = StorageConfig(**storage_dict)

            configs[agent_id] = AgentConfig(
                agent_id=agent_id,
                instructions=agent_data.get("instructions", []),
                tools=agent_data.get("tools", []),
                model_settings=model_data,
                storage_settings=storage_data,
            )

        self._config_cache = configs
        self._last_mtime = current_mtime

    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get a specific agent config (auto-reloads if needed)."""
        self._load_if_needed()
        return self._config_cache.get(agent_id)

    def get_all_agents(self) -> Dict[str, AgentConfig]:
        """Get all configs."""
        self._load_if_needed()
        return self._config_cache

    def save(self) -> None:
        """Save current config cache to file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Dict[str, Any]] = {}
        for agent_id, config in self._config_cache.items():
            agent_dict: Dict[str, Any] = {
                "instructions": config.instructions,
                "tools": config.tools,
                "model": config.model_settings.to_dict(),
            }
            if config.storage_settings:
                agent_dict["storage"] = config.storage_settings.to_dict()
            data[agent_id] = agent_dict

        with open(self.file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False)

        self._last_mtime = self.file_path.stat().st_mtime


CONFIG = AgentConfigLoader()
