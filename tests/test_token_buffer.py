import pytest
from src.utils.token_buffer import TokenBuffer

def test_token_buffer_initialization():
    tb = TokenBuffer()
    assert tb.token_count == 0
    assert tb.text == ""

def test_token_buffer_add_text():
    tb = TokenBuffer()
    text = "Hello world"
    tb.add_text(text)
    assert tb.text == text
    assert tb.token_count > 0
    
    # Check incremental update
    initial_count = tb.token_count
    tb.add_text(" again")
    assert tb.token_count > initial_count
    assert tb.text == "Hello world again"

def test_token_buffer_clear():
    tb = TokenBuffer()
    tb.add_text("Test")
    tb.clear()
    assert tb.token_count == 0
    assert tb.text == ""

def test_token_buffer_get_chunk_at():
    tb = TokenBuffer()
    # "Hello world" is usually 2 tokens ("Hello", " world") or similar
    tb.add_text("Hello world") 
    
    # We don't know exact tokenization without checking implementation, 
    # but we can check constraints.
    chunk = tb.get_chunk_at(100)
    assert chunk == "Hello world"
    
    # Test truncation
    # Force a long string
    long_text = "word " * 100
    tb.clear()
    tb.add_text(long_text)
    
    chunk_small = tb.get_chunk_at(10)
    # Re-encode check
    encoding = tb.encoding
    assert len(encoding.encode(chunk_small)) <= 10
