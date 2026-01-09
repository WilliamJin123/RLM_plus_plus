# Implementation Plan V3: The "Liquid" Agent Architecture

**Goal:** Achieve total behavioral plasticity. Every agent (RLM, Overseer, Optimizer, Ingestor) is instantiated from a dynamic configuration layer. Agents can modify the configuration of other agents (and themselves), effectively "rewriting" the system's runtime behavior without changing the underlying Python code structure.

**Core Philosophy:** "Identity is immutable; Behavior is fluid."

---

## Phase 1: The Universal Configuration Layer
*Objective: Centralize all agent behaviors (prompts, tools, model params) into a queryable, writable state store.*

- [ ] **Config Database (`src/core/config_store.py`)**
  - Implement a dedicated SQLite backend (separate from the document index) for system state.
  - **Table `agent_configs`:**
    - `agent_id` (Primary Key, TEXT): e.g., "rlm_agent", "overseer", "smart_ingestor".
    - `instructions` (JSON): List of system prompt strings.
    - `tools` (JSON): List of string tool names (mapped to functions via Registry).
    - `model_settings` (JSON): Provider, model name, temperature, max_tokens.
    - `storage_settings` (JSON): `add_history_to_messages`, `num_history_responses`, and any other agno configs (research agno docs, search the web if necessary). I want you to amke sure tehse are robust and well-thought out.

- [ ] **Migration Script (`src/utils/migrate_v3.py`)**
  - A one-time script to populate `agent_configs` with the current hardcoded defaults from V2 (e.g., reading `agent_prompt.yaml`).

---

## Phase 2: The Agent Factory & Dynamic Registry
*Objective: Replace direct `Agent()` calls with a factory that hydrates agents from the Config DB.*

- [ ] **Enhanced Tool Registry (`src/tools/registry.py`)**
  - Update to return a *dictionary* `{"tool_name": callable}`.
  - Support strict lookup by name string.

- [ ] **The Agent Factory (`src/core/factory.py`)**
  - **Function:** `create_agent(agent_id: str, session_id: str = None) -> Agent`
  - **Logic:**
    1. Load row from `agent_configs` where `id=agent_id`.
    2. Resolve tool strings to Python callables using `ToolRegistry`.
    3. Initialize `agno.agent.Agent` with:
       - `instructions` (from DB)
       - `tools` (from DB)
       - `model` (from DB)
       - `storage` (SqliteAgentStorage linked to `data/history.db`)
       - `add_history_to_messages` (from DB)

---

## Phase 3: Deep State & Contextual Memory
*Objective: Leverage `agno`'s persistent storage to give agents long-term memory and context awareness.*

- [ ] **Session Management**
  - Ensure every agent run is tagged with a `session_id`.
  - **RLM Agent:** Needs high history retention (`num_history_responses=10`) to remember previous search steps in a complex query.
  - **Overseer:** Needs access to the *RLM's* storage to read the full conversation history, not just the event bus stream.

- [ ] **Context Retrieval Tools (`src/tools/context_tools.py`)**
  - `get_agent_history(agent_id: str, last_n: int)`: Allows the Optimizer/Architect to read what an agent actually *did* in the past.

---

## Phase 4: The Architect (Meta-Optimizer)
*Objective: An agent specifically designed to edit the `agent_configs` table.*

- [ ] **The Architect Agent (`src/core/architect.py`)**
  - **Identity:** `agent_id="architect"`.
  - **Tools:**
    - `update_instructions(target_agent_id, new_instructions_list)`
    - `add_tool(target_agent_id, tool_name)`
    - `remove_tool(target_agent_id, tool_name)`
    - `update_model_params(target_agent_id, params_dict)`
  - **Constraint:** CANNOT delete rows or change `agent_id` (violates Immutable Identity rule).

- [ ] **Evolution Workflow (`src/workflows/evolution.py`)**
  - A directed graph or script that:
    1. Runs a benchmark.
    2. Feeds the logs + RLM history to the **Architect**.
    3. Architect decides to swap the RLM's model to "gpt-4-turbo" or add a specific tool.
    4. Architect executes `update_model_params`.
    5. Next run uses the new config.

---

## Phase 5: Integration
*Objective: Update the CLI to use the Factory.*

- [ ] **Refactor `src/main.py`**
  - Remove direct imports of `RLMAgent`, `Indexer`.
  - Use `AgentFactory.create_agent("indexer")` and `AgentFactory.create_agent("rlm")`.

- [ ] **Bootstrap Check**
  - On startup, check if `agent_configs` is empty. If so, run migration automatically.

---

## Next Steps

1.  Create `src/core/config_store.py` and define the schema.
2.  Refactor `src/tools/registry.py` to map strings to functions.
3.  Implement `src/core/factory.py`.

Prompt:
implement @IMPLEMENTATION_PLAN_V3.md (read @PAST_HISTORY.md  for additional context). respect the current implementations related to agno's SqliteDB, setup_tracing, and MultiProviderWrapper stuff. change anyhting else freely.   