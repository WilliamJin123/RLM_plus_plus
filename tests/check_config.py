from src.config.yaml_config import load_agents_config
from src.tools.registry import registry

def check_rlm_agent_config():
    try:
        configs = load_agents_config()
        agent = configs.get("rlm-agent")
        
        if not agent:
            print("Agent 'rlm-agent' NOT FOUND in config YAML.")
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

    except Exception as e:
        print(f"Error checking config: {e}")

if __name__ == "__main__":
    check_rlm_agent_config()