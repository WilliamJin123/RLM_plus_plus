import json
import re
from typing import Dict, Any, TypedDict
from src.core.factory import AgentFactory
from src.utils.token_buffer import TokenBuffer

class SegmentAnalysisResult(TypedDict):
    cut_index: int
    next_chunk_start_index: int
    summary: str
    reasoning: str

class SmartIngestor:
    def __init__(self, estimated_tokens: int = 4000):
        self.agent = AgentFactory.create_agent("smart-ingest-agent", estimated_tokens=estimated_tokens)
        self.token_buffer = TokenBuffer()

    def analyze_segment(self, text: str) -> SegmentAnalysisResult:
        """
        Analyzes a text segment to find the semantic cut point and generate a summary.
        Returns indices RELATIVE to the start of the provided string.
        """
        #NOTE: add debugging print statements

        # Prompt Optimization: Be explicit about the JSON format
        prompt = (
            f"Analyze this{len(text)} character text to create a document chunk.\n"
            f"\n{text}\n"
            f"\n<instructions>\n"
            f"1. Identify the best stopping point near the end of the text.\n"
            f"2. Identify where the NEXT chunk should start (create overlap for context).\n"
            f"3. Summarize the content from the start up to the cut point.\n"
            f"\nReturn STRICT JSON:\n"
            f"{{\n"
            f"  \"cut_index\": <int, relative index>,\n"
            f"  \"next_chunk_start_index\": <int, relative index < cut_index>,\n"
            f"  \"summary\": <string>,\n"
            f"  \"reasoning\": <string>\n"
            f"}}"
            f"</instructions>\n"
        )
        
        try:
            response = self.agent.run(prompt)
            content = response.content
            print(f"SmartIngest Response Content: {content}")
            # Cleanup code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
            content = think_pattern.sub("", content)
            data = json.loads(content.strip())

            cut = int(data.get('cut_index', len(text)))
            next_start = int(data.get('next_chunk_start_index', len(text) - 100))
            
            cut = min(max(0, cut), len(text))
            next_start = min(max(0, next_start), len(text))
            
            if next_start >= cut:
                next_start = max(0, cut - 50) # Fallback overlap

            return {
                "cut_index": cut,
                "next_chunk_start_index": next_start,
                "summary": data.get("summary", "No summary provided."),
                "reasoning": data.get("reasoning", "")
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # # Fail safe: take the whole segment, overlap 10%
            # l = len(text)
            # return {
            #     "cut_index": l,
            #     "next_chunk_start_index": int(l * 0.9),
            #     "summary": "Automatic fallback summary.",
            #     "reasoning": "Error in LLM processing."
            # }
            raise e
