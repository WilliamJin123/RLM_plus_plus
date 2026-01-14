import unittest
from unittest.mock import MagicMock, mock_open, patch

from src.core.indexer import Indexer


class TestIndexer(unittest.TestCase):
    @patch("src.core.indexer.AgentFactory")
    @patch("src.core.indexer.StorageEngine")
    @patch("src.core.indexer.FixedTokenChunker")
    @patch("src.core.indexer.TokenBuffer")
    def test_init_fixed_strategy(
        self, MockTokenBuffer, MockChunker, MockStorageEngine, MockAgentFactory
    ):
        indexer = Indexer(db_path=":memory:", strategy="fixed")
        MockChunker.assert_called_once()
        self.assertEqual(indexer.max_chunk_tokens, 4000)

    @patch("src.core.indexer.AgentFactory")
    @patch("src.core.indexer.StorageEngine")
    @patch("src.core.indexer.SemanticBoundaryChunker")
    @patch("src.core.indexer.TokenBuffer")
    def test_init_llm_strategy(
        self, MockTokenBuffer, MockChunker, MockStorageEngine, MockAgentFactory
    ):
        indexer = Indexer(db_path=":memory:", strategy="llm")
        MockChunker.assert_called_once()

    def test_init_invalid_strategy(self):
        with self.assertRaises(ValueError) as context:
            Indexer(db_path=":memory:", strategy="invalid")
        self.assertIn("Unknown chunking strategy", str(context.exception))

    @patch("src.core.indexer.Path")
    @patch("src.core.indexer.AgentFactory")
    @patch("src.core.indexer.StorageEngine")
    @patch("src.core.indexer.FixedTokenChunker")
    @patch("src.core.indexer.TokenBuffer")
    def test_ingest_file_not_found(
        self,
        MockTokenBuffer,
        MockChunker,
        MockStorageEngine,
        MockAgentFactory,
        MockPath,
    ):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        MockPath.return_value = mock_path

        indexer = Indexer(db_path=":memory:")
        with self.assertRaises(FileNotFoundError):
            indexer.ingest_file("nonexistent.txt")

    @patch("src.core.indexer.Path")
    @patch("src.core.indexer.AgentFactory")
    @patch("src.core.indexer.StorageEngine")
    @patch("src.core.indexer.FixedTokenChunker")
    @patch("src.core.indexer.TokenBuffer")
    def test_ingest_file_not_a_file(
        self,
        MockTokenBuffer,
        MockChunker,
        MockStorageEngine,
        MockAgentFactory,
        MockPath,
    ):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False
        MockPath.return_value = mock_path

        indexer = Indexer(db_path=":memory:")
        with self.assertRaises(ValueError) as context:
            indexer.ingest_file("directory/")
        self.assertIn("not a file", str(context.exception))


if __name__ == "__main__":
    unittest.main()
