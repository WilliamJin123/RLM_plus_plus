import pytest
from unittest.mock import MagicMock, patch, mock_open
from src.core.indexer import Indexer

@patch('src.core.indexer.SmartIngestor')
@patch('src.core.indexer.SessionLocal')
@patch('src.core.indexer.AgentFactory')
@patch('src.core.indexer.init_db')
def test_ingest_file(mock_init_db, mock_factory, mock_session_cls, mock_ingestor_cls, tmp_path):
    # Setup File
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "dummy.txt"
    p.write_text("Mock content " * 10, encoding='utf-8')
    
    # Setup Mocks
    mock_ingestor = mock_ingestor_cls.return_value
    
    # Mock find_cut_point to always return a cut half way through segment
    def side_effect_find_cut(segment):
        # simple logic: cut at len/2
        cut = len(segment) // 2
        return {
            "cut_index": cut,
            "next_chunk_start_index": cut,
            "reasoning": "split"
        }
    mock_ingestor.find_cut_point.side_effect = side_effect_find_cut
    
    mock_session = mock_session_cls.return_value
    
    mock_agent = MagicMock()
    mock_factory.create_agent.return_value = mock_agent
    mock_agent.run.return_value.content = "Summary text"
    
    # Run
    indexer = Indexer()
    # 130 chars. target 10 tokens ~ 50 chars lookahead.
    indexer.ingest_file(str(p), target_chunk_tokens=10)
    
    # Assertions
    # We expect some chunks to be added
    assert mock_session.add.called
    assert mock_session.commit.called
    
    # Check if summarization happened
    # We should have Level 0 summaries at least
    # group_size default is 2.
    # If we created > 2 chunks, we should have summaries.
    # 130 chars. chunks ~ 25 chars. ~5-6 chunks.
    # So summarization should trigger.
    assert mock_agent.run.called
    
    # Check hierarchy
    # We can't easily check DB state on a mock session unless we used an in-memory DB or sophisticated mock.
    # But ensuring 'add' was called with Summary objects is enough for unit test.
    
    # Filter calls to add
    added_objects = [call[0][0] for call in mock_session.add.call_args_list]
    summaries = [obj for obj in added_objects if hasattr(obj, 'summary_text')]
    chunks = [obj for obj in added_objects if hasattr(obj, 'text') and not hasattr(obj, 'summary_text')]
    
    assert len(chunks) > 0
    assert len(summaries) > 0
