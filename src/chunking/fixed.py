from typing import Generator

from src.chunking.base import BaseChunker, ChunkResult
from src.utils.token_buffer import TokenBuffer

# Average characters per token varies by language; 4 is typical for English
# Using 6 as safety factor to ensure we capture enough text
CHARS_PER_TOKEN_ESTIMATE = 6


class FixedTokenChunker(BaseChunker):
    def __init__(
        self,
        max_tokens: int,
        token_buffer: TokenBuffer,
        overlap_ratio: float = 0.1,
    ):
        super().__init__(max_tokens, token_buffer)
        if not 0 <= overlap_ratio < 1:
            raise ValueError("overlap_ratio must be between 0 and 1")
        self.overlap_ratio = overlap_ratio

    def chunk_text(self, text: str) -> Generator[ChunkResult, None, None]:
        if not text:
            return

        current_idx = 0
        text_len = len(text)

        while current_idx < text_len:
            # Grab a window of characters large enough to cover max_tokens
            char_window = self.max_tokens * CHARS_PER_TOKEN_ESTIMATE
            raw_end = min(current_idx + char_window, text_len)
            window_text = text[current_idx:raw_end]

            # Use TokenBuffer to find the precise character cut for max_tokens
            chunk_text = self.token_buffer.get_chunk_at(self.max_tokens, text=window_text)
            chunk_len = len(chunk_text)
            abs_end = current_idx + chunk_len

            yield ChunkResult(
                text=chunk_text,
                start_index=current_idx,
                end_index=abs_end,
            )

            # If we've reached the end, break
            if abs_end >= text_len:
                break

            # Calculate overlap based on character count
            overlap_chars = int(chunk_len * self.overlap_ratio)
            current_idx = abs_end - overlap_chars
