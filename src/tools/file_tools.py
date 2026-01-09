from typing import List, Optional
from sqlalchemy.orm import Session
from src.core.db import SessionLocal, Chunk, Summary

def get_document_structure() -> str:
    """
    Returns the top-level summaries (the 'root' or level N nodes) 
    to give an overview of the document.
    """
    session = SessionLocal()
    # Find the highest level
    highest_level = session.query(Summary.level).order_by(Summary.level.desc()).first()
    if highest_level is None:
        session.close()
        return "Document index is empty."
    
    level = highest_level[0]
    roots = session.query(Summary).filter(Summary.level == level).all()
    
    structure = f"Document Structure (Highest Level: {level}):\n"
    for r in roots:
        structure += f"- [ID: {r.id}] {r.summary_text[:200]}...\n"
    
    session.close()
    return structure

def get_summary_children(summary_id: int) -> str:
    """
    Returns the child summaries or chunks for a given summary ID.
    Allows the agent to 'drill down' into the document.
    """
    session = SessionLocal()
    summary = session.query(Summary).filter(Summary.id == summary_id).first()
    if not summary:
        session.close()
        return f"Summary with ID {summary_id} not found."
    
    children = session.query(Summary).filter(Summary.parent_id == summary_id).all()
    
    result = f"Children of Summary {summary_id} (Level {summary.level}):\n"
    if children:
        for c in children:
            result += f"- [Summary ID: {c.id}] {c.summary_text[:200]}...\n"
    elif summary.level == 0:
        # Level 0 summaries cover chunks
        chunk_ids = summary.get_chunk_id_list()
        result += f"This is a leaf summary covering chunks: {chunk_ids}\n"
        chunks = session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
        for ch in chunks:
            result += f"- [Chunk ID: {ch.id}] {ch.text[:100]}...\n"
    
    session.close()
    return result

def read_chunk(chunk_id: int) -> str:
    """
    Returns the full text of a specific chunk.
    """
    session = SessionLocal()
    chunk = session.query(Chunk).filter(Chunk.id == chunk_id).first()
    if not chunk:
        session.close()
        return f"Chunk with ID {chunk_id} not found."
    
    text = chunk.text
    session.close()
    return text

def search_summaries(query: str) -> str:
    """
    Searches summary text for a keyword or phrase.
    """
    session = SessionLocal()
    results = session.query(Summary).filter(Summary.summary_text.contains(query)).limit(10).all()
    
    if not results:
        session.close()
        return f"No summaries found matching '{query}'."
    
    output = f"Search results for '{query}':\n"
    for r in results:
        output += f"- [Summary ID: {r.id}, Level: {r.level}] {r.summary_text[:200]}...\n"
    
    session.close()
    return output
