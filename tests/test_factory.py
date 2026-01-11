import pytest
from unittest.mock import MagicMock, patch, ANY
from src.core.factory import AgentFactory
from src.config.config import config
from src.config.yaml_config import load_agents_config

# =================================================================================================
# UNIT TESTS (MOCKED)
# These tests ensure the factory logic is correct without making external calls.
# =================================================================================================

@patch('src.core.factory.MultiProviderWrapper')
def test_create_model_logic(mock_wrapper):
    mock_instance = MagicMock()
    mock_wrapper.from_env.return_value = mock_instance
    mock_instance.get_model.return_value = "mock_model"
    
    # Default
    res = AgentFactory.create_model(None)
    # Note: env_file path might vary, so we use ANY
    mock_wrapper.from_env.assert_called_with(
        provider=config.FAST_MODEL_PROVIDER,
        default_model_id=config.FAST_MODEL_NAME,
        env_file=ANY,
        temperature=0.0
    )
    assert res == "mock_model"
    
    # Custom
    custom_settings = {"provider": "custom", "model_id": "custom-id", "temperature": 0.7}
    res = AgentFactory.create_model(custom_settings)
    mock_wrapper.from_env.assert_called_with(
        provider="custom",
        default_model_id="custom-id",
        env_file=ANY,
        temperature=0.7
    )

@patch('src.config.yaml_config.get_agent_config')
@patch('src.core.factory.registry')
@patch('src.core.factory.Agent')
@patch('src.core.factory.AgentFactory.create_model')
def test_create_agent_mock(mock_create_model, mock_agent_cls, mock_registry, mock_get_config):
    # Setup mocks
    mock_config_record = MagicMock()
    mock_config_record.tools = ["tool1"]
    mock_config_record.model_settings = {}
    mock_config_record.instructions = ["instruction1"]
    mock_config_record.storage_settings = None
    mock_get_config.return_value = mock_config_record
    
    mock_registry.get_tool_map.return_value = {"tool1": lambda x: x}
    
    mock_create_model.return_value = "mock_model"
    
    mock_agent_instance = MagicMock()
    mock_agent_cls.return_value = mock_agent_instance
    
    # Run
    agent = AgentFactory.create_agent("test-agent")
    
    # Assert
    mock_get_config.assert_called_with("test-agent")
    mock_create_model.assert_called()
    mock_agent_cls.assert_called()
    
    # Check if tools were resolved
    call_args = mock_agent_cls.call_args
    assert call_args is not None
    _, kwargs = call_args
    assert "tools" in kwargs
    assert len(kwargs["tools"]) == 1
    assert kwargs["id"] == "test-agent"


# =================================================================================================
# INTEGRATION TESTS (LIVE)
# These tests verify the actual configuration in agents.yaml and attempt to connect to providers.
# WARNING: These tests consume tokens and require valid API keys in .env
# =================================================================================================

# Load all agent IDs from the actual configuration file
try:
    live_configs = load_agents_config()
    agent_ids = list(live_configs.keys())
except Exception as e:
    print(f"Could not load agent configs: {e}")
    agent_ids = []

@pytest.mark.integration
@pytest.mark.parametrize("agent_id", agent_ids)
def test_agent_connectivity(agent_id):
    """
    Iterates through all configured agents, initializes them, and attempts a minimal generation
    to verify provider and model ID correctness.
    """
    print(f"\n[Integration] Testing Agent: {agent_id}")
    
    try:
        # 1. Instantiate the agent
        # This checks if tools can be loaded and model wrapper can be initialized
        agent = AgentFactory.create_agent(agent_id)
        assert agent is not None, f"Failed to create agent {agent_id}"
        
        # 2. Check Model Configuration
        print(f"  Provider: {agent.model.provider if hasattr(agent.model, 'provider') else 'Unknown'}")
        print(f"  Model ID: {agent.model.model_id if hasattr(agent.model, 'model_id') else 'Unknown'}")

        # 3. Attempt a lightweight generation
        # We use a very simple prompt and try to limit tokens to avoid costs/time
        # Note: We wrap this in a try/except to catch provider errors (404, 401, etc.)
        
        # Using a direct print request to the agent to avoid history complications if possible,
        # but run() is the standard entry point.
        response = agent.run("Hello. Reply with 'OK'.", stream=False)
        
        # Extract content
        content = ""
        if hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'messages'):
            content = response.messages[-1].content
        else:
            content = str(response)
            
        print(f"  Response: {content[:50]}...") # Print first 50 chars
        
        # Basic assertion that we got *something* back
        assert content, "Agent returned empty content"
        
    except Exception as e:
        error_msg = str(e)
        print(f"  FAILED: {error_msg}")
        
        # Specific help for common errors
        if "404" in error_msg and "Route" in error_msg:
            pytest.fail(f"Agent '{agent_id}' failed with 404 (Route not found). "
                        f"Check if model ID '{live_configs[agent_id].model_settings.get('model_id')}' is valid on {live_configs[agent_id].model_settings.get('provider')}.")
        elif "401" in error_msg:
            pytest.fail(f"Agent '{agent_id}' failed with 401 (Unauthorized). Check API Key.")
        else:
            pytest.fail(f"Agent '{agent_id}' encountered an error: {error_msg}")

if __name__ == "__main__":
    # Allow running this file directly to test
    pytest.main(["-v", __file__])