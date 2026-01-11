import pytest
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock, patch, mock_open
from src.core.indexer import Indexer

@patch('src.core.indexer.SmartIngestor')
@patch('src.core.indexer.storage')
@patch('src.core.indexer.AgentFactory')
def test_ingest_file(mock_factory, mock_storage, mock_ingestor_cls, tmp_path):
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
    
    # Mock storage behavior
    mock_storage.db.open_table.return_value = MagicMock()
    
    # Mock add_summaries to return dummy IDs
    # It takes a list of summaries. We need to return a list of IDs of same length.
    def side_effect_add_summaries(summaries):
        return list(range(1000, 1000 + len(summaries)))
    mock_storage.add_summaries.side_effect = side_effect_add_summaries
    
    mock_agent = MagicMock()
    mock_factory.create_agent.return_value = mock_agent
    mock_agent.run.return_value.content = "Summary text"
    
    # Run
    indexer = Indexer()
    # 130 chars. target 10 tokens ~ 50 chars lookahead.
    indexer.ingest_file(str(p), target_chunk_tokens=10)
    
    # Assertions
    # We expect chunks to be added via storage.add_chunks
    assert mock_storage.add_chunks.called
    chunks_args = mock_storage.add_chunks.call_args[0][0]
    assert len(chunks_args) > 0
    assert "text" in chunks_args[0]
    
    # Check if summarization happened (Level 0 and maybe Level 1)
    assert mock_agent.run.called
    
    # Check storage calls for summaries
    assert mock_storage.add_summaries.called
    
    # Check that update was called (for parent_id linking in recursive step)
    # The new implementation calls storage.db.open_table("summaries").update(...)
    # We can check if db.open_table was called
    assert mock_storage.db.open_table.called