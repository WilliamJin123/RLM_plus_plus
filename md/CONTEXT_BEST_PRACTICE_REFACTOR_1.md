# Context Management Best Practice Refactor Report 1

## Overview
This refactor addresses violations of the "Sub-Agent Context Delegation" rule defined in `RULES.md`. The core principle is that raw data and large contexts must never be loaded directly into a main decision-maker agent's context. Instead, they should be processed by a specialized sub-agent.

## Changes

### 1. Refactored `src/tools/context_tools.py`
- **Violation Identification**: The tool `get_agent_history` was returning raw rows of agent interaction history (dicts) directly to the caller. This posed a risk of context overflow and required the caller (e.g., the Architect) to parse raw JSON/DB structures.
- **Resolution**:
    - Renamed `get_agent_history` to `_get_raw_agent_history` (internal helper).
    - Created a new tool `analyze_agent_history(agent_id, query, last_n)`.
    - **Pattern Implementation**: `analyze_agent_history` retrieves the raw history but **does not** return it. Instead, it spawns a temporary sub-agent (`AgentFactory.create_model()`), feeds it the history text, and asks it to answer the specific `query` provided by the caller. The main agent receives only the distilled answer.

### 2. Updated Tool Registry (`src/tools/registry.py`)
- Replaced the registration of `get_agent_history` with `analyze_agent_history`.
- This ensures that agents can only access the safe, sub-agent-backed tool.

### 3. Updated Agent Configuration (`src/config/agents.yaml`)
- Updated the `architect` agent's tool list to use `analyze_agent_history`.

## Verification
- Checked `src/tools/file_tools.py`: `analyze_chunk` already adheres to the pattern. `get_summary_children` returns truncated text (100 chars), which is acceptable for navigation breadcrumbs.
- Checked `src/core/monitor_bus.py`: `monitored_tool` correctly uses `functools.wraps` to preserve tool metadata for LLMs.

## Conclusion
The codebase now better adheres to the Context Management rules. The `Architect` agent is protected from raw history dumps and must now ask semantic questions about agent performance (e.g., "Why did the agent loop?") rather than parsing raw logs itself.
