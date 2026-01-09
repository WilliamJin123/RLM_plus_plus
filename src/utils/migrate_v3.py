import json
import yaml
from pathlib import Path
from src.core.config_store import init_config_db, SessionLocal, AgentConfig, ModelSettings, StorageSettings
from src.config import config

def load_yaml_instructions():
    try:
        path = Path("src/prompts/agent_prompt.yaml")
        if path.exists():
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("instructions", [])
    except Exception as e:
        print(f"Error loading prompts: {e}")
    return ["You are a helpful agent."]

def migrate():
    print("Initializing Config DB...")
    init_config_db()
    session = SessionLocal()

    # RLM Agent Defaults
    rlm_instructions = load_yaml_instructions()
    rlm_tools = [
        "get_document_structure",
        "get_summary_children",
        "analyze_chunk",
        "search_summaries",
        "PythonTools"
    ]
    rlm_model: ModelSettings = {
        "provider": config.FAST_MODEL_PROVIDER,
        "model_id": config.FAST_MODEL_NAME,
        "temperature": 0.0
    }
    rlm_storage: StorageSettings = {
        "db_path": "data/rlm_agent.db",
        "session_table": "rlm_agent_sessions",
        "add_history_to_context": False,
        "num_history_runs": 5,
        "read_chat_history": False
    }

    # Overseer Defaults
    overseer_instructions = [
        "You are the Overseer. Your job is to monitor the RLM Agent's thoughts and actions.",
        "Ensure the agent is making progress and not looping.",
        "If the agent is stuck, provide a hint or direction.",
        "Do not interfere if the agent is proceeding logically."
    ]
    overseer_tools = [] 
    overseer_model: ModelSettings = {
        "provider": config.FAST_MODEL_PROVIDER,
        "model_id": config.FAST_MODEL_NAME,
        "temperature": 0.1
    }
    overseer_storage: StorageSettings = {
        "db_path": "data/overseer.db",
        "session_table": "overseer_sessions",
        "add_history_to_context": True,
        "num_history_runs": 10,
        "read_chat_history": True
    }
    
    # Architect Defaults
    architect_instructions = [
        "You are the Architect. You have the power to modify the behavior of other agents.",
        "You optimize the system by analyzing performance logs and updating agent configurations.",
        "You can update instructions, tools, and model parameters."
    ]
    architect_tools = [
        "update_instructions",
        "add_tool",
        "remove_tool",
        "update_model_params",
        "get_agent_history"
    ]
    architect_model: ModelSettings = {
        "provider": config.FAST_MODEL_PROVIDER,
        "model_id": config.FAST_MODEL_NAME,
        "temperature": 0.0
    }
    architect_storage: StorageSettings = {
        "db_path": "data/architect.db",
        "session_table": "architect_sessions",
        "add_history_to_context": True,
        "num_history_runs": 5,
        "read_chat_history": True
    }

    defaults = [
        AgentConfig(
            agent_id="rlm-agent",
            instructions=rlm_instructions,
            tools=rlm_tools,
            model_settings=rlm_model,
            storage_settings=rlm_storage
        ),
        AgentConfig(
            agent_id="overseer",
            instructions=overseer_instructions,
            tools=overseer_tools,
            model_settings=overseer_model,
            storage_settings=overseer_storage
        ),
        AgentConfig(
            agent_id="architect",
            instructions=architect_instructions,
            tools=architect_tools,
            model_settings=architect_model,
            storage_settings=architect_storage
        )
    ]

    for default_config in defaults:
        existing = session.query(AgentConfig).filter_by(agent_id=default_config.agent_id).first()
        if not existing:
            print(f"Creating config for {default_config.agent_id}")
            session.add(default_config)
        else:
            # Optionally update if exists, to ensure new schema fields (like read_chat_history) are present?
            # For now, let's just skip as per "immutable identity" somewhat, or we could update settings.
            # Let's update settings just in case schema evolved.
            print(f"Updating config for {default_config.agent_id}")
            existing.storage_settings = default_config.storage_settings
            existing.model_settings = default_config.model_settings
            # Don't overwrite instructions/tools as they might have been "evolved"
    
    session.commit()
    session.close()
    print("Migration/Update complete.")

if __name__ == "__main__":
    migrate()