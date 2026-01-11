from typing import List, Optional
from agno.agent import Agent
from src.core.factory import AgentFactory
from src.core.storage import storage

def get_document_structure() -> str:
    """
    Returns the top-level summaries (the 'root' or level N nodes) 
    to give an overview of the document.
    """
    return storage.get_document_structure()

def get_summary_children(summary_id: int) -> str:
    """
    Returns the child summaries or chunks for a given summary ID.
    Allows the agent to 'drill down' into the document.
    """
    return storage.get_summary_children(summary_id)

def analyze_chunk(chunk_id: int, query: str) -> str:
    """
    Spawns a sub-agent to read the full text of a specific chunk and answer a query.
    This prevents the main agent's context from being flooded with raw text.
    
    Args:
        chunk_id: The ID of the chunk to analyze.
        query: The specific question or instruction for the sub-agent regarding this chunk.
    """
    text = storage.get_chunk_text(chunk_id)
    if not text:
        return f"Chunk with ID {chunk_id} not found."
    
    try:
        # Spawn a lightweight sub-agent to read the chunk
        sub_agent = AgentFactory.create_agent("chunk-analyzer-agent")
        
        prompt = f"Context:\n{text}\n\nQuestion: {query}"
        response = sub_agent.run(prompt)
        return str(response.content)
    except Exception as e:
        return f"Error analyzing chunk: {str(e)}"

def search_summaries(query: str) -> str:
    """
    Searches summary text for a keyword or phrase.
    """
    return storage.search_summaries(query)
