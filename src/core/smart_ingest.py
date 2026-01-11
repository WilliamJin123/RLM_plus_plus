import json
from typing import Dict, Any
from agno.agent import Agent
from src.core.factory import AgentFactory

class SmartIngestor:
    def __init__(self):
        self.agent = AgentFactory.create_agent("smart-ingest-agent")

    def find_cut_point(self, text: str) -> Dict[str, Any]:
        """
        Asks the LLM to find the best cut point in the text AND summarize the resulting chunk.
        Returns indices relative to the start of 'text' and the summary.
        """
        # We now send the FULL text segment to allow the LLM to summarize the chunk it creates.
        # Previously we only sent the window, but that prevented simultaneous summarization.
        text_to_analyze = text
        
        # Calculate offset is 0 since we are using the full text
        offset = 0

        prompt = (
            f"Analyze this text segment (length: {len(text_to_analyze)} chars), which is a candidate for a document chunk:\n\n"
            f"---\n{text_to_analyze}\n---\n\n"
            f"Task 1: Find the best semantic cut point (natural stopping point) for this chunk.\n"
            f"Task 2: Determine where the NEXT chunk should start (create an overlap).\n"
            f"Task 3: Write a concise summary of the text FROM THE START up to your chosen cut point.\n\n"
            "Return a JSON object with:\n"
            "- 'cut_index' (int): Relative index (0 to length) where this chunk ends.\n"
            "- 'next_chunk_start_index' (int): Relative index where the next chunk begins (MUST be < cut_index).\n"
            "- 'reasoning' (str): Why you chose this point.\n"
            "- 'summary' (str): The summary of the chunk content.\n\n"
            "CRITICAL: Indices must be strictly within the range [0, length_of_segment]."
        )
        
        try:
            response = self.agent.run(prompt)
            content = response.content
            print(f"\n[SmartIngest] RAW LLM OUTPUT:\n{content}\n-----------------------------------")

            # Cleanup code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"[SmartIngest] JSON Decode Error. Content: {content}")
                raise

            print(f"[SmartIngest] Parsed JSON: {data}")

            data.setdefault('reasoning', 'No reasoning provided by LLM.')
            data.setdefault('summary', 'No summary provided by LLM.')
            
            # --- VALIDATION & SANITIZATION ---
            
            raw_cut = int(data.get('cut_index', len(text_to_analyze)))
            raw_next = int(data.get('next_chunk_start_index', len(text_to_analyze) - 200))
            
            print(f"[SmartIngest] Pre-calculation Indices (Relative to analysis window): Cut={raw_cut}, NextStart={raw_next}")

            # 1. Clamp to valid window range
            if raw_cut < 0 or raw_cut > len(text_to_analyze):
                 print(f"[SmartIngest] WARNING: Cut index {raw_cut} out of bounds [0, {len(text_to_analyze)}]. Clamping.")
            raw_cut = max(0, min(raw_cut, len(text_to_analyze)))
            
            if raw_next < 0 or raw_next > len(text_to_analyze):
                print(f"[SmartIngest] WARNING: Next start index {raw_next} out of bounds [0, {len(text_to_analyze)}]. Clamping.")
            raw_next = max(0, min(raw_next, len(text_to_analyze)))
            
            # 2. Enforce overlap (Next < Cut)
            # If next is after cut, we have a gap. Force overlap.
            min_overlap = 50 # Reduced to be less aggressive if LLM wants small overlap
            if raw_next >= raw_cut:
                print(f"[SmartIngest] GAP DETECTED or Negative Overlap: Next ({raw_next}) >= Cut ({raw_cut}). Enforcing overlap.")
                raw_next = max(0, raw_cut - min_overlap)
                data['reasoning'] += " [Auto-corrected: Forced overlap]"
            
            # 3. Apply offset to map back to full 'text'
            data['cut_index'] = raw_cut + offset
            data['next_chunk_start_index'] = raw_next + offset
            
            print(f"[SmartIngest] Final Relative Indices (relative to segment start): Cut={data['cut_index']}, NextStart={data['next_chunk_start_index']}")
            return data
            
        except Exception as e:
            print(f"SmartIngest Error: {e}. Fallback to static overlap.")
            import traceback
            traceback.print_exc()
            # Fallback
            cut_idx = len(text)
            overlap = 200
            return {
                "cut_index": cut_idx,
                "next_chunk_start_index": max(0, cut_idx - overlap),
                "reasoning": "Fallback due to error.",
                "summary": "Fallback summary due to error."
            }
