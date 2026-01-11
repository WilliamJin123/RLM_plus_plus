import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM providers and models."""
    
    # Provider API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    CEREBRAS_API_KEY: Optional[str] = os.getenv("CEREBRAS_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    # Model Selection
    # Fast models for summarization/indexing (Groq/Cerebras/OpenRouter)
    FAST_MODEL_PROVIDER: str = "groq" # Options: groq, cerebras, openrouter, openai, anthropic
    FAST_MODEL_NAME: str = "llama-3.3-70b-versatile" 
    
    # Reasoning models for the Agent (OpenAI/Anthropic/OpenRouter)
    REASONING_MODEL_PROVIDER: str = "groq" # Options: openai, anthropic, openrouter
    REASONING_MODEL_NAME: str = "llama-3.3-70b-versatile"

    # OpenRouter defaults if used
    OPENROUTER_MODEL_NAME: str = "meta-llama/llama-3.1-70b-instruct"

config = LLMConfig()
