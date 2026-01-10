from src.core.config_store import SessionLocal, AgentConfig
from src.tools.registry import registry

def check_rlm_agent_config():
    db = SessionLocal()
    try:
        agent = db.query(AgentConfig).filter_by(agent_id="rlm-agent").first()
        if not agent:
            print("Agent 'rlm-agent' NOT FOUND in config DB.")
            return

        print(f"Agent ID: {agent.agent_id}")
        print(f"Tools: {agent.tools}")
        
        architect_tools = ["update_instructions", "add_tool", "remove_tool", "update_model_params"]
        missing_tools = [t for t in architect_tools if t not in agent.tools]
        
        if not missing_tools:
            print("SUCCESS: All architect (self-editing) tools are enabled.")
        else:
            print(f"WARNING: The following architect tools are MISSING: {missing_tools}")
            print("Self-editing capabilities are NOT fully active for this agent.")

    finally:
        db.close()

if __name__ == "__main__":
    check_rlm_agent_config()
