import json
import logging
import re
from typing import Any, Dict, Generator

from src.chunking.base import BaseChunker, ChunkResult
from src.core.factory import AgentFactory, ModelRotator
from src.utils.token_buffer import TokenBuffer

logger = logging.getLogger(__name__)

# Characters per token estimate for window sizing
CHARS_PER_TOKEN_ESTIMATE = 6
# Default overlap when LLM doesn't provide valid next_start
DEFAULT_OVERLAP_CHARS = 50
# Maximum characters to show in prompt for cut-point detection
MAX_PROMPT_CHARS = 2000


class SemanticBoundaryChunker(BaseChunker):
    def __init__(self, max_tokens: int, token_buffer: TokenBuffer):
        super().__init__(max_tokens, token_buffer)
        self.agent, self.rotator = AgentFactory.create_rotating_agent("smart-ingest-agent")

    def chunk_text(self, text: str) -> Generator[ChunkResult, None, None]:
        if not text:
            return

        current_idx = 0

        while current_idx < len(text):
            # Get a window slightly larger than max_tokens to give the LLM choices
            char_window = self.max_tokens * CHARS_PER_TOKEN_ESTIMATE
            raw_end = min(current_idx + char_window, len(text))
            window_text = text[current_idx:raw_end]

            # Trim to max tokens strict limit to ensure we don't overflow context
            valid_window = self.token_buffer.get_chunk_at(self.max_tokens, text=window_text)

            # Ask LLM to find the break point
            cut_data = self._find_cut_point(valid_window)

            cut_rel = min(cut_data["cut_index"], len(valid_window))
            next_start_rel = cut_data["next_chunk_start_index"]

            # Ensure next_start is before cut point
            if next_start_rel >= cut_rel:
                next_start_rel = max(0, cut_rel - DEFAULT_OVERLAP_CHARS)

            abs_end = current_idx + cut_rel
            chunk_text = text[current_idx:abs_end]

            yield ChunkResult(
                text=chunk_text,
                start_index=current_idx,
                end_index=abs_end,
            )

            if abs_end >= len(text):
                break

            current_idx = current_idx + next_start_rel

    def _find_cut_point(self, text: str) -> Dict[str, Any]:
        # Show a representative portion of the text for context
        display_text = text[-MAX_PROMPT_CHARS:] if len(text) > MAX_PROMPT_CHARS else text

        prompt = (
            f"Analyze this text segment.\n"
            f"1. Identify the best semantic stopping point (end of a topic/paragraph) near the end.\n"
            f"2. Identify where the next chunk should start to maintain context (overlap).\n"
            f"Text length: {len(text)} chars.\n\n"
            f"Text:\n{display_text}\n\n"
            f'Return JSON: {{ "cut_index": <int>, "next_chunk_start_index": <int> }}'
        )

        try:
            # Rotate model before each call
            model_config = self.rotator.get_next_config()
            self.agent.model = AgentFactory.create_model(model_config)
            logger.debug("Using model: %s/%s", model_config.provider, model_config.model_id)

            response = self.agent.run(prompt)
            content = response.content
            logger.debug("SmartIngest response: %s", content)

            parsed = self._extract_json(content)
            cut = min(max(0, int(parsed.get("cut_index", len(text)))), len(text))
            next_start = min(max(0, int(parsed.get("next_chunk_start_index", len(text) - 100))), len(text))

            if next_start >= cut:
                next_start = max(0, cut - DEFAULT_OVERLAP_CHARS)

            return {
                "cut_index": cut,
                "next_chunk_start_index": next_start,
            }

        except Exception as e:
            logger.error("Error finding cut point: %s", e)
            raise

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling code blocks and think tags."""
        # Remove think tags if present
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Try to extract from code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()

        # Try direct JSON parsing
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object pattern
        match = re.search(r"\{[^{}]*\}", content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {content[:200]}")
