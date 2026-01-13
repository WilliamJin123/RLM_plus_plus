import unittest
from unittest.mock import patch, MagicMock, mock_open
from src.core.indexer import Indexer

class TestIndexer(unittest.TestCase):

    @patch('src.core.indexer.AgentFactory')
    @patch('src.core.indexer.StorageEngine')
    @patch('src.core.indexer.SmartIngestor')
    @patch('src.core.indexer.TokenBuffer')
    def setUp(self, MockTokenBuffer, MockSmartIngestor, MockStorageEngine, MockAgentFactory):
        self.mock_agent_factory = MockAgentFactory
        self.mock_storage = MockStorageEngine.return_value
        self.mock_smart_ingestor = MockSmartIngestor.return_value
        self.mock_token_buffer = MockTokenBuffer.return_value
        self.indexer = Indexer(db_path=":memory:")

    @patch('src.core.indexer.Path')
    def test_ingest_file(self, MockPath):
        # Setup
        mock_path_obj = MagicMock()
        MockPath.return_value = mock_path_obj
        mock_path_obj.name = "test.txt"
        
        # Mock file opening
        m = mock_open(read_data="Full text content")
        mock_path_obj.open = m
        
        # Mock internal methods
        self.indexer._process_chunks = MagicMock(return_value=[1, 2, 3])
        self.indexer._build_hierarchy = MagicMock()
        
        # Act
        self.indexer.ingest_file("path/to/test.txt", max_chunk_tokens=100, group_size=2)
        
        # Assert
        MockPath.assert_called_with("path/to/test.txt")
        mock_path_obj.open.assert_called_with('r', encoding='utf-8')
        self.indexer._process_chunks.assert_called_with("Full text content", 100, "test.txt")
        self.indexer._build_hierarchy.assert_called_with([1, 2, 3], group_size=2)

    def test_process_chunks(self):
        # Setup
        full_text = "Chunk1Chunk2"
        max_chunk_tokens = 10
        filename = "test.txt"
        
        # TokenBuffer behavior
        self.mock_token_buffer.get_chunk_at.side_effect = ["Chunk1", "Chunk2"]
        
        # SmartIngestor behavior
        # First iteration: cut after "Chunk1", next starts at end of "Chunk1"
        # Second iteration: cut after "Chunk2" (end of text)
        self.mock_smart_ingestor.analyze_segment.side_effect = [
            {'cut_index': 6, 'next_chunk_start_index': 6, 'summary': 'Sum1'},
            {'cut_index': 6, 'next_chunk_start_index': 6, 'summary': 'Sum2'}
        ]
        
        # Storage returns IDs
        self.mock_storage.add_chunk.side_effect = [101, 102]
        self.mock_storage.add_summary.side_effect = [201, 202]
        
        # Act
        summary_ids = self.indexer._process_chunks(full_text, max_chunk_tokens, filename)
        
        # Assert
        self.assertEqual(summary_ids, [201, 202])
        self.assertEqual(self.mock_storage.add_chunk.call_count, 2)
        self.assertEqual(self.mock_storage.add_summary.call_count, 2)
        self.mock_storage.link_summary_to_chunk.assert_any_call(201, 101)
        self.mock_storage.link_summary_to_chunk.assert_any_call(202, 102)

    @patch('time.sleep') # speed up test
    def test_build_hierarchy(self, mock_sleep):
        # Setup
        level_0_ids = [1, 2, 3, 4]
        group_size = 2
        
        # Mock getting texts for summarization
        self.mock_storage.get_chunk_texts.side_effect = [
            ["Text1", "Text2"], # First batch
            ["Text3", "Text4"], # Second batch
            ["SumText1", "SumText2"] # Higher level batch
        ]
        
        # Mock summarization
        self.indexer._summarize_text = MagicMock(side_effect=["SummaryA", "SummaryB", "RootSummary"])
        
        # Mock adding summaries
        # Level 1: IDs 10, 11
        # Level 2: ID 100
        self.mock_storage.add_summary.side_effect = [10, 11, 100]
        
        # Act
        self.indexer._build_hierarchy(level_0_ids, group_size=group_size)
        
        # Assert
        # Check parents created
        self.mock_storage.add_summary.assert_any_call(text="SummaryA", level=1)
        self.mock_storage.add_summary.assert_any_call(text="SummaryB", level=1)
        self.mock_storage.add_summary.assert_any_call(text="RootSummary", level=2)
        
        # Check linking
        self.mock_storage.update_summary_parent.assert_any_call(1, 10)
        self.mock_storage.update_summary_parent.assert_any_call(2, 10)
        self.mock_storage.update_summary_parent.assert_any_call(3, 11)
        self.mock_storage.update_summary_parent.assert_any_call(4, 11)
        self.mock_storage.update_summary_parent.assert_any_call(10, 100)
        self.mock_storage.update_summary_parent.assert_any_call(11, 100)

if __name__ == '__main__':
    unittest.main()
