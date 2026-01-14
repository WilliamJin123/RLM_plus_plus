from abc import ABC, abstractmethod
from typing import Generator, NamedTuple

from src.utils.token_buffer import TokenBuffer


class ChunkResult(NamedTuple):
    text: str
    start_index: int
    end_index: int


class BaseChunker(ABC):
    def __init__(self, max_tokens: int, token_buffer: TokenBuffer):
        self.max_tokens = max_tokens
        self.token_buffer = token_buffer

    @abstractmethod
    def chunk_text(self, text: str) -> Generator[ChunkResult, None, None]:
        """Yields text chunks based on the specific strategy."""
        pass
