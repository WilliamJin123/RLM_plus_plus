import pytest
from unittest.mock import MagicMock, patch, ANY
from src.core.factory import AgentFactory
from src.config.config import config

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
def test_create_agent(mock_create_model, mock_agent_cls, mock_registry, mock_get_config):
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
