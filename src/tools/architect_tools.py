from typing import List, Dict, Any, cast
from src.core.config_store import SessionLocal, AgentConfig, ModelSettings
from sqlalchemy.orm.attributes import flag_modified

def update_instructions(agent_id: str, new_instructions: List[str]) -> str:
    """Replaces the instructions for a given agent."""
    db = SessionLocal()
    try:
        agent = db.query(AgentConfig).filter_by(agent_id=agent_id).first()
        if not agent:
            return f"Agent {agent_id} not found."
        
        agent.instructions = new_instructions
        flag_modified(agent, "instructions") # Ensure SQLAlchemy notices JSON change
        db.commit()
        return f"Updated instructions for {agent_id}."
    except Exception as e:
        return f"Error: {e}"
    finally:
        db.close()

def add_tool(agent_id: str, tool_name: str) -> str:
    """Adds a tool to the agent's configuration."""
    db = SessionLocal()
    try:
        agent = db.query(AgentConfig).filter_by(agent_id=agent_id).first()
        if not agent:
            return f"Agent {agent_id} not found."
        
        current_tools = list(agent.tools)
        if tool_name in current_tools:
            return f"Tool {tool_name} already exists for {agent_id}."
        
        current_tools.append(tool_name)
        agent.tools = current_tools
        flag_modified(agent, "tools")
        db.commit()
        return f"Added tool {tool_name} to {agent_id}."
    except Exception as e:
        return f"Error: {e}"
    finally:
        db.close()

def remove_tool(agent_id: str, tool_name: str) -> str:
    """Removes a tool from the agent's configuration."""
    db = SessionLocal()
    try:
        agent = db.query(AgentConfig).filter_by(agent_id=agent_id).first()
        if not agent:
            return f"Agent {agent_id} not found."
        
        current_tools = list(agent.tools)
        if tool_name not in current_tools:
            return f"Tool {tool_name} not found for {agent_id}."
        
        current_tools.remove(tool_name)
        agent.tools = current_tools
        flag_modified(agent, "tools")
        db.commit()
        return f"Removed tool {tool_name} from {agent_id}."
    except Exception as e:
        return f"Error: {e}"
    finally:
        db.close()

def update_model_params(agent_id: str, params: Dict[str, Any]) -> str:
    """Updates model settings (provider, model_id, temperature, etc.)."""
    db = SessionLocal()
    try:
        agent = db.query(AgentConfig).filter_by(agent_id=agent_id).first()
        if not agent:
            return f"Agent {agent_id} not found."
        
        # Cast to dict to ensure update method is recognized
        current_settings = cast(dict, agent.model_settings)
        current_settings.update(params)
        agent.model_settings = current_settings
        flag_modified(agent, "model_settings")
        db.commit()
        return f"Updated model settings for {agent_id}."
    except Exception as e:
        return f"Error: {e}"
    finally:
        db.close()
