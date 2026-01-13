import unittest
from unittest.mock import patch, MagicMock
from src.core.smart_ingest import SmartIngestor

class TestSmartIngestor(unittest.TestCase):

    @patch('src.core.smart_ingest.AgentFactory')
    def test_analyze_segment_valid(self, MockAgentFactory):
        # Setup
        mock_agent = MagicMock()
        MockAgentFactory.create_agent.return_value = mock_agent
        
        # Valid JSON response
        mock_response = MagicMock()
        mock_response.content = """
        ```json
        {
            "cut_index": 50,
            "next_chunk_start_index": 45,
            "summary": "This is a summary.",
            "reasoning": "Good cut point."
        }
        ```
        """
        mock_agent.run.return_value = mock_response
        
        ingestor = SmartIngestor()
        text = "x" * 100
        
        # Act
        result = ingestor.analyze_segment(text)
        
        # Assert
        self.assertEqual(result['cut_index'], 50)
        self.assertEqual(result['next_chunk_start_index'], 45)
        self.assertEqual(result['summary'], "This is a summary.")
        self.assertEqual(result['reasoning'], "Good cut point.")

    @patch('src.core.smart_ingest.AgentFactory')
    def test_analyze_segment_malformed(self, MockAgentFactory):
        # Setup
        mock_agent = MagicMock()
        MockAgentFactory.create_agent.return_value = mock_agent
        
        # Malformed JSON
        mock_response = MagicMock()
        mock_response.content = "Not JSON"
        mock_agent.run.return_value = mock_response
        
        ingestor = SmartIngestor()
        text = "x" * 100
        
        # Act
        result = ingestor.analyze_segment(text)
        
        # Assert - should fallback
        self.assertEqual(result['cut_index'], 100) # Length of text
        self.assertEqual(result['next_chunk_start_index'], 90) # 90%
        self.assertEqual(result['summary'], "Automatic fallback summary.")
        self.assertEqual(result['reasoning'], "Error in LLM processing.")

    @patch('src.core.smart_ingest.AgentFactory')
    def test_analyze_segment_out_of_bounds(self, MockAgentFactory):
        # Setup
        mock_agent = MagicMock()
        MockAgentFactory.create_agent.return_value = mock_agent
        
        # JSON with indices out of bounds
        mock_response = MagicMock()
        mock_response.content = """
        {
            "cut_index": 200,
            "next_chunk_start_index": 250,
            "summary": "Summary",
            "reasoning": "Reasoning"
        }
        """
        mock_agent.run.return_value = mock_response
        
        ingestor = SmartIngestor()
        text = "x" * 100 # len 100
        
        # Act
        result = ingestor.analyze_segment(text)
        
        # Assert - should clamp
        self.assertEqual(result['cut_index'], 100) # Clamped to len
        # next_start was 250 -> clamped to 100.
        # Logic: if next_start >= cut (100 >= 100), next_start = max(0, cut - 50) = 50
        self.assertEqual(result['next_chunk_start_index'], 50) 

if __name__ == '__main__':
    unittest.main()
