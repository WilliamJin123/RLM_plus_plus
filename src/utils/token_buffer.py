import tiktoken

class TokenBuffer:
    def __init__(self, model_name: str = "gpt-4o"):
        """
        A buffer that tracks the approximate token count of the text it holds.
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.buffer = ""
        self._token_count = 0

    def add_text(self, text: str):
        """Appends text to the buffer and updates token count."""
        self.buffer += text
        # Recalculating full buffer tokens is safer than incremental for multi-byte char issues,
        # but slower. For now, we append. 
        # Optimization: cache the count and only encode the new text?
        # Let's just re-encode for safety in V1, or append tokens.
        self._token_count = len(self.encoding.encode(self.buffer))

    def clear(self):
        self.buffer = ""
        self._token_count = 0

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def text(self) -> str:
        return self.buffer

    def get_chunk_at(self, max_tokens: int) -> str:
        """Returns the text that fits within max_tokens."""
        tokens = self.encoding.encode(self.buffer)
        if len(tokens) <= max_tokens:
            return self.buffer
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
