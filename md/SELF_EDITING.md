# Self-Editing & Evolution in RLM++

Welcome to the **Self-Editing** guide for the RLM++ (Retrieval Augmented Language Model Plus Plus) codebase. This document explains the core mechanisms that allow agents in this system to modify their own behavior, tools, and configuration at runtime.

If you are a new developer here, you might be used to static agent configurations (hardcoded prompts, fixed toolsets). **RLM++ is different.** Agents here are "soft-coded" via a database and possess tools to rewrite that database.

---

## 1. High-Level Architecture

The self-editing capability relies on three pillars:

1.  **The Configuration Store (`config.db`):** The single source of truth for how an agent behaves.
2.  **The Agent Factory:** A runtime builder that assembles agents from the config DB.
3.  **The Architect Tools:** A set of Python functions that allow an agent (usually the "Architect") to SQL-update the config DB.

### The Loop
1.  **Run:** An agent is instantiated by the `AgentFactory`.
2.  **Observe:** The `Overseer` (or the agent itself) notices a failure, inefficiency, or user request for change.
3.  **Evolve:** The `Architect` agent is invoked. It uses `Architect Tools` to modify `config.db`.
4.  **Re-instantiate:** The next time the agent runs, the `AgentFactory` reads the *new* config, and the agent has new prompts, new tools, or a new model.

---

## 2. The Brain: Configuration Store

Location: `src/core/config_store.py`
Database: `data/config.db` (SQLite)

Instead of `agent = Agent(instructions="...")` in a Python file, we store configuration in the `agent_configs` table.

| Column | Type | Description |
| :--- | :--- | :--- |
| `agent_id` | String | Unique ID (e.g., "rlm-agent"). |
| `instructions` | JSON List | The system prompt / persona. |
| `tools` | JSON List | List of string keys mapping to functions in `registry.py`. |
| `model_settings` | JSON Dict | Defines the LLM provider, model ID, and temperature. |

**Implication:** If you change a row in this table, you change the agent.

---

## 3. The Hands: Architect Tools

Location: `src/tools/architect_tools.py`

These are standard Python functions exposed as tools to the `Architect` agent (and potentially others).

### `update_instructions(agent_id, new_instructions)`
*   **Purpose:** Re-writes the system prompt.
*   **Use Case:** If an agent is too verbose, the Architect can call this to add "Be concise" to its instructions.

### `add_tool(agent_id, tool_name)` / `remove_tool(agent_id, tool_name)`
*   **Purpose:** Modifies the toolset available to an agent.
*   **Use Case:** If an agent needs to search the web but lacks the tool, the Architect can grant it `search_google`.

### `update_model_params(agent_id, params)`
*   **Purpose:** Changes the underlying LLM.
*   **Use Case:** If a task requires reasoning, switch `model_id` from a fast model to a reasoning model (e.g., o1 or similar).

---

## 4. Dynamic Tool Creation (Advanced)

Location: `src/tools/dynamic/` & `src/tools/registry.py`

Beyond selecting from existing tools, the system can **write new code**.

1.  **Code Generation:** An agent uses standard file writing tools to create a new `.py` file in `src/tools/dynamic/`.
2.  **Auto-Discovery:** The `ToolRegistry` (`src/tools/registry.py`) scans this directory on initialization.
3.  **Integration:** The agent then calls `add_tool(agent_id, "new_function_name")` to equip itself with the code it just wrote.

**Example:**
> "I need to calculate the Fibonacci sequence, but I don't have a tool for it."
> *   *Agent writes `src/tools/dynamic/math_utils.py` with `def fibonacci(n): ...`*
> *   *Agent calls `add_tool('rlm-agent', 'fibonacci')`*
> *   *Next run: Agent calls `fibonacci(10)`*

---

## 5. The Overseer & Monitor Bus

Location: `src/core/overseer.py`, `src/core/monitor_bus.py`

Self-editing isn't always manual. The **Overseer** is a passive monitoring agent that watches the event stream.

*   It subscribes to `monitor_bus`.
*   It detects loops (repeating same tool) or errors.
*   It can trigger an "Intervention" or suggest an "Evolution" to the Architect.

---

## 6. CLI Workflow for Evolution

The system includes a CLI command specifically for this:

```bash
python src/main.py evolve --reason "Agent failed to answer coding questions"
```

**What happens:**
1.  The `Architect` agent starts up.
2.  It reads the prompt: "The system administrator has requested an evolution... Reason: Agent failed..."
3.  The Architect analyzes the `rlm-agent` config.
4.  It might decide: "The agent needs access to the file system to read code."
5.  It calls `add_tool("rlm-agent", "read_file")`.
6.  It might also call `update_instructions("rlm-agent", ["...add step to read files first..."])`.
7.  The process finishes. The `rlm-agent` is now permanently upgraded.

---

## 7. Safety & Security

**Warning:** This system allows an LLM to execute code and modify its own constraints.

*   **Sandboxing:** In a production environment, this **must** run in a container (Docker/Firecracker).
*   **Human-in-the-loop:** Currently, the CLI triggers evolution, providing a checkpoint. Fully autonomous evolution (Overseer triggering Architect directly) is possible but risky.
*   **Rollback:** The `config.db` is just a SQLite file. Backup this file to snapshot agent states.

---

## Summary for New Devs

*   **Don't hardcode prompts.** Put them in `config.db`.
*   **Don't hardcode tool lists.** Put them in `config.db`.
*   **If you want the agent to improve,** invoke the Architect.
*   **If you want new powers,** write a dynamic tool.

Happy Hacking!
