import pytest
from src.utils.token_buffer import TokenBuffer


def test_token_buffer_initialization():
    tb = TokenBuffer()
    assert tb.encoding is not None


def test_token_buffer_initialization_with_model():
    tb = TokenBuffer(model_name="gpt-4o")
    assert tb.encoding is not None


def test_token_buffer_count_tokens():
    tb = TokenBuffer()
    count = tb.count_tokens("Hello world")
    assert count > 0


def test_token_buffer_count_tokens_empty():
    tb = TokenBuffer()
    count = tb.count_tokens("")
    assert count == 0


def test_token_buffer_count_tokens_none():
    tb = TokenBuffer()
    count = tb.count_tokens(None)
    assert count == 0


def test_token_buffer_get_chunk_at():
    tb = TokenBuffer()
    # "Hello world" is typically 2-3 tokens
    chunk = tb.get_chunk_at(100, text="Hello world")
    assert chunk == "Hello world"


def test_token_buffer_get_chunk_at_truncates():
    tb = TokenBuffer()
    long_text = "word " * 100  # Many tokens
    chunk = tb.get_chunk_at(10, text=long_text)
    # Verify truncation worked
    assert len(tb.encoding.encode(chunk)) <= 10


def test_token_buffer_get_chunk_at_empty():
    tb = TokenBuffer()
    chunk = tb.get_chunk_at(100, text="")
    assert chunk == ""


def test_token_buffer_get_chunk_at_none():
    tb = TokenBuffer()
    chunk = tb.get_chunk_at(100, text=None)
    assert chunk == ""
