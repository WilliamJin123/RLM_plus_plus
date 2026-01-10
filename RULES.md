# Codebase Rules: Context Management

## Core Principle: Sub-Agent Context Delegation

In this codebase, **Context Management** is a critical architectural concern. To maintain the efficiency and accuracy of high-level decision-maker agents (like the `rlm-agent`), raw data and large contexts must **never** be loaded directly into their primary context window if it can be avoided.

Instead, all significant context retrieval and analysis must be delegated to specialized **Sub-Agents**.

### The Rule

**"Context should always be verified and processed by a sub-agent before being passed to an important decision-maker agent."**

### Rationale

1.  **Context Overflow Prevention**: Loading raw files or large data chunks directly can quickly exhaust the token limit of the model.
2.  **Attention Management**: Large contexts dilute the model's attention. A dedicated sub-agent focusing on a single chunk or specific task yields higher accuracy.
3.  **Cost Efficiency**: Processing raw data with smaller, faster calls (or even different models) is often more efficient than loading everything into a large reasoning model's context.

### Implementation Pattern

When a feature requires reading data (files, database records, logs):

1.  **Do NOT** return the raw string content to the main agent.
2.  **Spawn a temporary Sub-Agent** (e.g., using `agno.agent.Agent`).
3.  **Feed the Context** to this sub-agent along with a specific **Query** or **Instruction**.
4.  **Return the Sub-Agent's Response** to the main agent.

### Existing Examples

#### 1. `analyze_chunk` in `src/tools/file_tools.py`

This function is the primary example of this pattern. It allows the `rlm-agent` to "read" specific parts of a large document without loading the text into its own memory.

```python
# src/tools/file_tools.py

def analyze_chunk(chunk_id: int, query: str) -> str:
    """
    Spawns a sub-agent to read the full text of a specific chunk and answer a query.
    This prevents the main agent's context from being flooded with raw text.
    """
    # ... (retrieval logic) ...
    text = chunk.text
    
    # PATTERN: Spawn sub-agent
    sub_agent = Agent(
        model=AgentFactory.create_model(),
        description="You are a precise reading assistant.",
        instructions="Read the provided context and answer the user's question accurately...",
        markdown=True
    )
    
    # PATTERN: Delegate task
    prompt = f"Context:\n{text}\n\nQuestion: {query}"
    response = sub_agent.run(prompt)
    
    # PATTERN: Return distilled insight, not raw data
    return str(response.content)
```

### Usage in Configuration

The agent configuration in `src/config/agents.yaml` explicitly instructs the agent to use this pattern:

```yaml
rlm-agent:
  instructions:
    # ...
    - 3. Use 'analyze_chunk(chunk_id, query)' to ask a sub-agent to find specific information
      within a raw text chunk (prevents context overflow).
```
