import json
from typing import Dict, Any
from agno.agent import Agent
from src.core.get_model import get_model

class SmartIngestor:
    def __init__(self):
        self.model = get_model()
        self.agent = Agent(
            model=self.model,
            description="You are an expert text segmenter.",
            instructions=[
                "Analyze the provided text buffer.",
                "Identify the best semantic stopping point (e.g., end of a paragraph, section, or topic) near the end of the text.",
                "Determine how many lines of overlap are needed for the NEXT chunk to maintain context.",
                "Return your response in strict JSON format: { 'cut_index': int, 'next_chunk_start_index': int, 'reasoning': str }",
                "The 'cut_index' is the character index relative to the start of the text where the current chunk should end.",
                "The 'next_chunk_start_index' is the character index where the next chunk should begin (usually < cut_index to allow overlap).",
                "Do NOT return markdown blocks, just the JSON string."
            ],
            markdown=True
        )

    def find_cut_point(self, text: str) -> Dict[str, Any]:
        """
        Asks the LLM to find the best cut point in the text.
        Returns indices relative to the start of 'text'.
        """
        # We only send the last part of the buffer to save tokens, 
        # as the cut point must be near the end.
        window_size = 2000
        offset = 0
        text_to_analyze = text
        
        if len(text) > window_size:
            offset = len(text) - window_size
            text_to_analyze = text[offset:]

        prompt = f"Analyze this text segment (which is the end of a larger buffer):\n\n---\n{text_to_analyze}\n---\n\nFind the best cut point relative to the start of THIS segment. Return 'cut_index' and 'next_chunk_start_index' as integers relative to the start of this provided text snippet."
        
        try:
            response = self.agent.run(prompt)
            content = response.content
            # Cleanup code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            data = json.loads(content)
            
            # The LLM gives indices relative to text_to_analyze.
            # We need to add offset to make them relative to 'text'.
            data['cut_index'] = data['cut_index'] + offset
            data['next_chunk_start_index'] = data['next_chunk_start_index'] + offset
            
            # Safety checks
            if data['cut_index'] > len(text):
                data['cut_index'] = len(text)
            if data['next_chunk_start_index'] < 0:
                data['next_chunk_start_index'] = 0
                
            return data
            
        except Exception as e:
            print(f"SmartIngest Error: {e}. Fallback to static overlap.")
            # Fallback
            cut_idx = len(text)
            overlap = 200
            return {
                "cut_index": cut_idx,
                "next_chunk_start_index": max(0, cut_idx - overlap),
                "reasoning": "Fallback due to error."
            }
