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
        # Optimization: We sum the token counts of parts. 
        # This effectively is an upper bound (conservative) because merging tokens usually reduces count.
        # e.g. "a" + "pp" -> 2 tokens, "app" -> 1 token.
        # This is safe for context window limits.
        self._token_count += len(self.encoding.encode(text))

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
