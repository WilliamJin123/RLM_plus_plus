from typing import List, Dict, Any, cast
from src.config.yaml_config import load_agents_config, save_agents_config

def update_instructions(agent_id: str, new_instructions: List[str]) -> str:
    """Replaces the instructions for a given agent."""
    try:
        configs = load_agents_config()
        if agent_id not in configs:
            return f"Agent {agent_id} not found."
        
        configs[agent_id].instructions = new_instructions
        save_agents_config(configs)
        return f"Updated instructions for {agent_id}."
    except Exception as e:
        return f"Error: {e}"

def add_tool(agent_id: str, tool_name: str) -> str:
    """Adds a tool to the agent's configuration."""
    try:
        configs = load_agents_config()
        if agent_id not in configs:
            return f"Agent {agent_id} not found."
        
        agent = configs[agent_id]
        if tool_name in agent.tools:
            return f"Tool {tool_name} already exists for {agent_id}."
        
        agent.tools.append(tool_name)
        save_agents_config(configs)
        return f"Added tool {tool_name} to {agent_id}."
    except Exception as e:
        return f"Error: {e}"

def remove_tool(agent_id: str, tool_name: str) -> str:
    """Removes a tool from the agent's configuration."""
    try:
        configs = load_agents_config()
        if agent_id not in configs:
            return f"Agent {agent_id} not found."
        
        agent = configs[agent_id]
        if tool_name not in agent.tools:
            return f"Tool {tool_name} not found for {agent_id}."
        
        agent.tools.remove(tool_name)
        save_agents_config(configs)
        return f"Removed tool {tool_name} from {agent_id}."
    except Exception as e:
        return f"Error: {e}"

def update_model_params(agent_id: str, params: Dict[str, Any]) -> str:
    """Updates model settings (provider, model_id, temperature, etc.)."""
    try:
        configs = load_agents_config()
        if agent_id not in configs:
            return f"Agent {agent_id} not found."
        
        agent = configs[agent_id]
        
        if agent.readonly_model:
            return f"Modification not allowed: Agent {agent_id} has a read-only model configuration."

        agent.model_settings.update(params)
        save_agents_config(configs)
        return f"Updated model settings for {agent_id}."
    except Exception as e:
        return f"Error: {e}"