# Implementation Plan V2: Autonomous RLM (Smart Ingestion & Self-Evolution)

**Goal:** Evolve the RLM++ into a self-regulating, self-improving system. This includes LLM-driven dynamic chunking and a meta-layer where agents can monitor performance and safely evolve their own configuration.

**Core Philosophy:** "The model defines the boundaries, but the system ensures safety."

---

## Phase 1: Dynamic Context-Aware Chunking ("Smart Ingest")
*Objective: Replace rigid sliding windows with semantic, model-driven segmentation using high-speed inference.*

- [ ] **Token-Aware Buffer (`src/utils/token_buffer.py`)**
  - Implement a helper that maintains a rolling buffer of text, tracking estimated token counts (using `tiktoken` or char approximations) to ensure we respect the "Planning Window" (e.g., 60-80% of context).

- [ ] **The Segmentation Agent (`src/core/smart_ingest.py`)**
  - **Role:** Analyzes the buffer to find semantic break points (chapter ends, topic shifts).
  - **Model:** **MUST use the Fast Model (Groq/Cerebras)**. Using GPT-4 here would be too slow/expensive.
  - **Prompt:** "Identify the best semantic stopping point. Determine how many lines of overlap are strictly necessary for context continuity."
  - **Output:** JSON `{ "cut_index": int, "next_chunk_start_index": int, "reasoning": str }`.

- [ ] **Update Indexer (`src/core/indexer.py`)**
  - Integrate `SmartIngestor` into the ingestion pipeline.
  - Store the "reasoning" for the cut in the DB (creating a "map of intent" for the document).

---

## Phase 2: The Overseer (Monitor Agent)
*Objective: A "God-mode" agent that observes the RLM's execution flow and intervenes.*

- [ ] **Event Bus & Logging (`src/core/monitor_bus.py`)**
  - A singleton that streams: Tool calls, outputs, agent "thoughts", and errors.

- [ ] **The Overseer Agent (`src/core/overseer.py`)**
  - **Role:** Watches the stream.
  - **Triggers:**
    - "Loop detected" (Agent repeats actions).
    - "Stagnation" (No meaningful tool output for N steps).
  - **Intervention:** The Overseer can inject "System Messages" into the running RLM Agent's history (e.g., *"You are stuck. Stop searching for 'X' and try 'Y'."*) without restarting the session.

---

## Phase 3: Recursive Self-Improvement (Safe Evolution)
*Objective: Allow the system to patch itself. Prompts are treated as data, tools as dynamic plugins.*

- [ ] **Externalize Prompts (`src/prompts/*.yaml`)**
  - Move hardcoded system prompts from `agent.py` to YAML files.
  - **Benefit:** The Optimizer can rewrite a text file safely without parsing/breaking Python code.

- [ ] **Tool Registry & Dynamic Loading (`src/tools/registry.py`)**
  - A system to load tools from `src/tools/dynamic/`.
  - **The Optimizer (`src/core/optimizer.py`)**:
    1. **Prompt Tuning:** Rewrites `src/prompts/agent_prompt.yaml` based on benchmark failures.
    2. **Tool Generation:** Writes new Python files to `src/tools/dynamic/`.
    3. **Validation:** **CRITICAL.** The Optimizer must write a corresponding unit test for any new tool. The tool is only registered if the test passes.

- [ ] **Evolution Loop (`src/main.py --evolve`)**
  - A mode that runs benchmarks -> analyzes logs -> generates patches (prompts/tools) -> verifies patches -> commits changes.

---

## Phase 4: Integration & Safety
*Objective: Prevent the agent from deleting the project or creating malware.*

- [ ] **Safety Sandbox**
  - **File Restrictions:** The Tool Manager can ONLY write to `src/tools/dynamic/` and `src/prompts/`.
  - **Import Restrictions:** Static analysis (AST) to ban `os.system`, `subprocess`, `shutil.rmtree` in generated tools.

- [ ] **Master Runner (`src/main.py`)**
  - Unified CLI:
    - `ingest <file> --strategy smart`
    - `query <text> --monitor`
    - `evolve --iterations 3`

---

## Next Steps

1.  Create `src/utils/token_buffer.py`.
2.  Implement `src/core/smart_ingest.py` using Groq/Cerebras.
3.  Refactor `src/core/agent.py` to load prompts from a file (preparing for Phase 3).