import re
from typing import List, Dict

def sliding_window_chunker(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Chunks text into overlapping segments.
    
    Args:
        text: The input text to chunk.
        chunk_size: Target size of each chunk in characters (approximate).
        overlap: Number of characters to overlap between chunks.
        
    Returns:
        A list of dictionaries containing chunk text and indices.
    """
    chunks = []
    if not text:
        return chunks

    start = 0
    while start < len(text):
        end = start + chunk_size
        
        # If we're not at the end of the text, try to find a good breaking point (e.g., newline or period)
        if end < len(text):
            # Look for last newline or period in the last 100 chars of the chunk
            search_window = text[max(start, end-100):end]
            break_point = -1
            for char in ['\n\n', '\n', '. ', ' ']:
                idx = search_window.rfind(char)
                if idx != -1:
                    break_point = max(start, end-100) + idx + len(char)
                    break
            
            if break_point != -1:
                end = break_point

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_index": start,
                "end_index": end
            })
            
        # Move start forward by chunk_size - overlap
        if end >= len(text):
            break
            
        start = end - overlap
        if start < 0:
            start = 0
            
    return chunks

if __name__ == "__main__":
    sample_text = "This is a test. " * 100
    test_chunks = sliding_window_chunker(sample_text, chunk_size=100, overlap=20)
    for i, c in enumerate(test_chunks[:5]):
        print(f"Chunk {i} ({c['start_index']}-{c['end_index']}): {c['text'][:50]}...")
