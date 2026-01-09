Create src/config.py
# Implementation Plan: Recursive Language Model (RLM++)

**Goal:** Build a Recursive Language Model that treats context as an external environment (REPL + Database), utilizing hierarchical summarization (Tree Index) to outperform standard context windows.

**Stack:** Python 3.12 (Windows), `uv`, `agno` (Agents), `groq`/`cerebras` (Fast Inference), `openai`/`anthropic` (Reasoning).

---

## Phase 0: Project Initialization & Config
*Objective: Robust configuration for multiple API providers.*

- [ ] **Setup `.env`**
  - Add keys for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, and `CEREBRAS_API_KEY`.

- [ ] **Create `src/config.py`**
  - Define an `LLMConfig` class to easily swap between:
    - **"Fast Model"** (Groq/Cerebras) for reading/summarizing.

- [ ] **Create Directory Structure**
```text
RLM_plus_plus/
├── .env                # API Keys
├── data/               # Storage for SQLite & Text & MD files
├── src/
│   ├── core/           # Main logic (Agent, DB setup)
│   ├── tools/          # Python REPL & DB Tools
│   └── utils/          # Ingestion & Chunking
├── benchmarks/         # S-NIAH, OOLONG, CODEQA scripts
└── implementation_plan.md

---

## Phase 1: The "Smart" Data Layer (Ingestion & Tree Index)
*Objective: Instead of a flat file, we build a "Tree-Inspired" index using SQLAlchemy.*

- [ ] **Chunking Utility (`src/utils/ingest.py`)**
  - Implement a sliding window chunker (overlap is crucial for S-NIAH).

- [ ] **Database Schema (`src/core/db.py`)**
  - Use `sqlalchemy` to create a SQLite DB with two tables:
    - **`Chunks`**: `id`, `text`, `start_index`, `end_index`.
    - **`Summaries`**: `id`, `summary_text`, `level` (0=leaf, 1=branch), `parent_id`.

- [ ] **The Indexer (`src/core/indexer.py`)**
  - **Optimization:** Use `Groq` (Llama-3-70b) or `Cerebras` (Llama-3.1) here for extreme speed.
  - **Logic:** Read text -> Chunk -> Summarize Chunks -> Recursively Summarize Summaries -> Store in DB.
  - **Outcome:** A "Map" of the document is created before the agent even starts.

---

## Phase 2: The Agentic Core (The RLM)
*Objective: Build the Agent that can "browse" the data layer.*

- [ ] **Define Tools (`src/tools/file_tools.py`)**
  - `read_chunk(chunk_id)`: Returns raw text.
  - `query_index(search_term)`: Searches the `Summaries` table (SQL).
  - `get_document_structure()`: Returns the top-level tree nodes (the "Table of Contents").

- [ ] **Define the RLM Agent (`src/core/agent.py`)**
  - Use `agno` to define the Agent.
  - **System Prompt:** Critical. Instruct the model *not* to hallucinate context but to use Python/SQL tools to find it.
  - **The REPL:** Allow the agent to execute Python code to loop through chunks (needed for OOLONG).
  - **Constraints:** Implement `MAX_STEPS` to prevent infinite loops on your API bill.

---

## Phase 3: Benchmark Implementation
*Objective: Verify the system works on the three chosen tasks.*

- [ ] **S-NIAH Test (`benchmarks/test_s_niah.py`)**
  - Generate a synthetic 100k token file with a random "needle" (UUID).
  - **Success Condition:** Retrieving the exact UUID.

- [ ] **OOLONG-PAIRS Test (`benchmarks/test_oolong.py`)**
  - Create a dataset of 50 "People" with "Locations" and "Timestamps".
  - **Task:** "Find all pairs of people who were in Paris at the same time."
  - **Success Condition:** The agent writes a Python loop to check the DB, rather than guessing.

- [ ] **CODEQA Test (`benchmarks/test_codeqa.py`)**
  - Ingest a small open-source repo (e.g., the `requests` library).
  - **Task:** "Which function handles the SSL verification logic?"

---

## Phase 4: Refinement & "Budget-Awareness"
*Objective: Make it cheaper and safer.*

- [ ] **Implement Budgeting**
  - Add a `token_spend` counter in the Agent state.
  - If spend > threshold, force the agent to stop "Reading" and start "Guessing" or return "I don't know."

- [ ] **Windows Path Safety**
  - Ensure all file operations in `src/tools` use Python's `pathlib` to handle `\` vs `/`.

---

## Next Immediate Steps

1.  Create the folder structure outlined in Phase 0.
2.  Create the `.env` file with your API keys.
3.  **Run the first ingest test:** Write a script to ingest a sample `.txt` file into your SQLite DB using `sqlalchemy` and `groq` for summarization.