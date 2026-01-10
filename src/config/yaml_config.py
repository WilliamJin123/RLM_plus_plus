from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    provider: str
    model_id: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
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
            "read_chat_history": self.read_chat_history
        }

@dataclass
class AgentConfig:
    agent_id: str
    instructions: List[str]
    tools: List[str]
    model_settings: Dict[str, Any] # Keeping as dict for compatibility
    storage_settings: Dict[str, Any] # Keeping as dict for compatibility

_cached_config: Optional[Dict[str, AgentConfig]] = None

def load_agents_config(path: str = "agents.yaml") -> Dict[str, AgentConfig]:
    global _cached_config
    # Refresh logic could be added here if we want to support hot-reloading from disk
    # For now, let's allow re-reading if explicit, but caching is fine for read-heavy.
    # However, architect tools need to write and read back.
    # So we should probably invalidate cache on write.
    
    yaml_path = Path(path)
    if not yaml_path.exists():
        # Try finding it in project root if we are running from elsewhere
        # Assuming current working directory might be different or src/...
        # But for now, let's assume CWD or root.
        root_path = Path(__file__).parent.parent.parent / "agents.yaml"
        if root_path.exists():
            yaml_path = root_path
        else:
             raise FileNotFoundError(f"Configuration file {path} not found.")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    configs = {}
    for agent_id, agent_data in data.items():
        model_data = agent_data.get("model", {})
        storage_data = agent_data.get("storage", {})
        
        # We store them as dicts to match what AgentFactory expects (it expects TypedDict/JSON)
        # But we could also use objects if we updated Factory.
        # Given the Plan, let's keep it compatible.
        
        configs[agent_id] = AgentConfig(
            agent_id=agent_id,
            instructions=agent_data.get("instructions", []),
            tools=agent_data.get("tools", []),
            model_settings=model_data,
            storage_settings=storage_data
        )
    
    _cached_config = configs
    return configs

def save_agents_config(configs: Dict[str, AgentConfig], path: str = "agents.yaml"):
    global _cached_config
    yaml_path = Path(path)
    if not yaml_path.exists():
         root_path = Path(__file__).parent.parent.parent / "agents.yaml"
         if root_path.exists():
             yaml_path = root_path
    
    data = {}
    for agent_id, config in configs.items():
        data[agent_id] = {
            "instructions": config.instructions,
            "tools": config.tools,
            "model": config.model_settings,
            "storage": config.storage_settings
        }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    _cached_config = configs

def get_agent_config(agent_id: str) -> Optional[AgentConfig]:
    configs = load_agents_config()
    return configs.get(agent_id)
