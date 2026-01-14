import logging

import tiktoken

logger = logging.getLogger(__name__)

# Default encoding for models without specific tokenizers
DEFAULT_ENCODING = "cl100k_base"


class TokenBuffer:
    def __init__(self, model_name: str = "gpt-4o"):
        """
        A utility to handle token counting and truncation.
        Stateless: Does not store text internally.
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.debug(
                "No specific tokenizer for '%s', using default encoding '%s'",
                model_name,
                DEFAULT_ENCODING,
            )
            self.encoding = tiktoken.get_encoding(DEFAULT_ENCODING)

    def count_tokens(self, text: str) -> int:
        """Counts tokens in a string using the current encoding."""
        if not text:
            return 0
        return len(self.encoding.encode(text, allowed_special='all'))

    def get_chunk_at(self, max_tokens: int, text: str) -> str:
        """
        Accepts a string and truncates it to fit within max_tokens.
        Returns the truncated string.
        """
        if not text:
            return ""

        tokens = self.encoding.encode(text, allowed_special='all')

        # If it fits, return original text to save decoding time
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
