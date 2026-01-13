import unittest
from unittest.mock import patch, MagicMock, call
from src.core.factory import AgentFactory
from src.config.config import ModelConfig, AgentConfig, StorageConfig

class TestAgentFactory(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test to ensure isolation
        AgentFactory._wrapper_cache = {}

    @patch('src.core.factory.MultiProviderWrapper')
    def test_get_cached_wrapper(self, MockWrapper):
        # Setup
        mock_instance = MagicMock()
        MockWrapper.from_env.return_value = mock_instance
        
        # Act
        wrapper1 = AgentFactory._get_cached_wrapper("openai")
        wrapper2 = AgentFactory._get_cached_wrapper("openai")
        
        # Assert
        self.assertEqual(wrapper1, mock_instance)
        self.assertEqual(wrapper2, mock_instance)
        # Should only be initialized once
        MockWrapper.from_env.assert_called_once()
        self.assertEqual(MockWrapper.from_env.call_args[1]['provider'], "openai")

    @patch('src.core.factory.AgentFactory._get_cached_wrapper')
    def test_create_model(self, mock_get_wrapper):
        # Setup
        mock_wrapper = MagicMock()
        mock_get_wrapper.return_value = mock_wrapper
        model_config = ModelConfig(provider="openai", model_id="gpt-4", temperature=0.7)
        
        # Act
        AgentFactory.create_model(model_config)
        
        # Assert
        mock_get_wrapper.assert_called_with(provider="openai")
        mock_wrapper.get_model.assert_called_with(id="gpt-4", temperature=0.7)

    @patch('src.core.factory.TOOL_REGISTRY')
    def test_hydrate_tools(self, mock_registry):
        # Setup
        mock_rlm_tool_cls = MagicMock()
        mock_python_tool_cls = MagicMock()
        mock_registry.get.side_effect = lambda name: {
            "RLMTools": mock_rlm_tool_cls,
            "PythonTools": mock_python_tool_cls
        }.get(name)
        
        tool_names = ["RLMTools", "PythonTools", "UnknownTool"]
        db_path = "/tmp/test.db"
        
        # Act
        tools = AgentFactory._hydrate_tools(tool_names, db_path)
        
        # Assert
        self.assertEqual(len(tools), 2)
        mock_rlm_tool_cls.assert_called_with(db_path=db_path)
        mock_python_tool_cls.assert_called_with()

    @patch('src.core.factory.CONFIG')
    @patch('src.core.factory.AgentFactory.create_model')
    @patch('src.core.factory.AgentFactory._hydrate_tools')
    @patch('src.core.factory.SqliteDb')
    @patch('src.core.factory.setup_tracing')
    @patch('src.core.factory.Agent')
    def test_create_agent(self, MockAgent, mock_setup_tracing, MockSqliteDb, mock_hydrate, mock_create_model, mock_config):
        # Setup
        agent_id = "test-agent"
        model_settings = ModelConfig(provider="openai", model_id="gpt-4", temperature=0.5)
        storage_settings = StorageConfig(db_path="custom.db", session_table="test_sessions", add_history_to_context=True, num_history_runs=3, read_chat_history=True)
        
        agent_config = AgentConfig(
            agent_id=agent_id,
            instructions=["Do this"],
            tools=["RLMTools"],
            model_settings=model_settings,
            storage_settings=storage_settings
        )
        mock_config.get_agent.return_value = agent_config
        
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_tools = [MagicMock()]
        mock_hydrate.return_value = mock_tools
        
        mock_db = MagicMock()
        MockSqliteDb.return_value = mock_db
        
        # Act
        agent = AgentFactory.create_agent(agent_id)
        
        # Assert
        mock_config.get_agent.assert_called_with(agent_id)
        mock_create_model.assert_called_with(model_settings)
        # Check hydrate tools call - verify db_path is passed from storage settings
        mock_hydrate.assert_called()
        args, _ = mock_hydrate.call_args
        self.assertEqual(args[0], ["RLMTools"])
        self.assertEqual(args[1], "custom.db") # Should use custom db path
        
        MockSqliteDb.assert_called() # Check DB initialization
        mock_setup_tracing.assert_called_with(db=mock_db, batch_processing=True)
        
        MockAgent.assert_called_with(
            id=agent_id,
            name="Test Agent",
            model=mock_model,
            tools=mock_tools,
            instructions=["Do this"],
            db=mock_db,
            add_history_to_context=True,
            num_history_runs=3,
            read_chat_history=True,
            markdown=True
        )

if __name__ == '__main__':
    unittest.main()
