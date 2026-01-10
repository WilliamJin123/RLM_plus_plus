# RLM++ Usage Guide

Welcome to the **RLM++** (Retrieval-Loop-Monitor Plus Plus) system. This repository implements a "Liquid Agent" architecture designed for high behavioral plasticity, allowing agents to evolve their own configurations (prompts, tools, models) at runtime.

---

## 🏗️ Architecture Overview

The core philosophy of RLM++ is: **"Identity is immutable; Behavior is fluid."**

Instead of hardcoding agent behaviors in Python classes, all agent definitions are stored in a SQLite configuration database (`data/config.db`).
- **The Factory:** At runtime, the `AgentFactory` reads this DB to instantiate agents.
- **The Architect:** A specialized agent that can write to this DB, effectively "rewriting" how other agents behave without changing code.
- **The Registry:** Maps string names in the DB to actual Python functions.

### Key Agents
1.  **RLM Agent (`rlm-agent`):** The primary worker for Retrieval Augmented Generation (RAG) tasks. It performs document search, summarization, and answering.
2.  **Overseer (`overseer`):** Monitors the RLM Agent to prevent loops and ensure logical progress (currently in basic logging mode).
3.  **Architect (`architect`):** The meta-optimizer. It analyzes performance and modifies the `rlm-agent` or `overseer` configurations (e.g., changing models, updating system prompts).

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- `uv` (recommended) or `pip`

### 1. Clone and Install Dependencies
```bash
git clone <repository-url>
cd RLM_plus_plus

# Using uv (recommended)
uv sync

# OR using pip
pip install .
```

### 2. Environment Configuration
Create a `.env` file in the root directory. You will need API keys for the models you intend to use (e.g., Cerebras, OpenAI, Anthropic).

```env
# Example .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
CEREBRAS_API_KEY=...
# ... add other provider keys as needed
```

### 3. Initialize the System
The first time you run any command, the system will automatically:
1.  Initialize the configuration database (`data/config.db`).
2.  Run the migration script to populate it with default agents (`rlm-agent`, `overseer`, `architect`).

You can also force this manually:
```bash
python src/utils/migrate_v3.py
```

---

## 💻 CLI Usage

The primary entry point is `src/main.py`.

### 1. Ingest Documents
Ingests a file into the RAG system. This builds the document index used by the RLM Agent.

```bash
python src/main.py ingest <path_to_file> --strategy smart
```
*   `--strategy`: Options are `smart` (hierarchical summarization) or `basic`. Default is `smart`.

### 2. Query the System
Runs the **RLM Agent** to answer a question based on ingested data.

```bash
python src/main.py query "What are the key themes in the document?" --monitor
```
*   `--monitor`: (Optional) Enables the Overseer agent to log/monitor the session.

### 3. Evolve the System (The Architect)
Triggers the **Architect Agent** to review and optimize the configuration of other agents. This simulates a "nightly build" or optimization cycle.

```bash
python src/main.py evolve --reason "RLM agent failed to answer complex queries"
```
*   `--reason`: Context for why the evolution is happening. The Architect uses this to decide what changes to make (e.g., "switch to a stronger model" or "clarify instructions").

---

## 🧠 Configuration & "Liquid" State

All agent state is stored in `data/config.db` in the `agent_configs` table.

### Schema (`src/core/config_store.py`)
| Column | Type | Description |
| :--- | :--- | :--- |
| `agent_id` | TEXT | Unique ID (e.g., "rlm-agent"). Immutable. |
| `instructions` | JSON | List of system prompt strings. |
| `tools` | JSON | List of tool names (strings). |
| `model_settings` | JSON | Provider, model ID, temperature. |
| `storage_settings` | JSON | Database paths and memory settings. |

### Inspecting Configs
Since it's a standard SQLite database, you can inspect it using any SQLite viewer or the CLI (if a command is added in the future). Currently, the `migrate_v3.py` script resets these to defaults.

---

## 🛠️ Tool Registry

Tools are decoupled from agents. The **Registry** (`src/tools/registry.py`) maps string names to Python callables.

### Available Tools
*   **Standard Tools:** `get_document_structure`, `get_summary_children`, `analyze_chunk`, `search_summaries`.
*   **Architect Tools:** `update_instructions`, `add_tool`, `remove_tool`, `update_model_params`.
*   **Context Tools:** `get_agent_history`.

### Adding a New Tool
1.  Create a function in `src/tools/` (or `src/tools/dynamic/`).
2.  Register it in `src/tools/registry.py` inside the `get_tool_map()` method.
3.  (Optional) Use the Architect to add this tool to an agent at runtime, or update `src/utils/migrate_v3.py` to add it by default.

---

## 📊 Benchmarks

The `benchmarks/` directory contains tests for specific datasets.

```bash
# Example: Run the BrowseComp+ benchmark
python benchmarks/test_browsecomp_plus.py
```

*   `test_browsecomp_plus.py`: Tests browsing/search capabilities.
*   `test_longbenchv2.py`: Long-context understanding.
*   `test_oolong.py`: Synthetic logic/reasoning tests.
*   `test_s_niah.py`: "Needle In A Haystack" retrieval tests.

---

## 📂 Project Structure

```
RLM_plus_plus/
├── benchmarks/         # Performance tests
├── data/               # SQLite databases (config.db, history.db)
├── src/
│   ├── core/           # Core logic (Factory, DB, Monitor)
│   ├── prompts/        # Static prompts (YAML)
│   ├── tools/          # Tool definitions & Registry
│   ├── utils/          # Migration & helpers
│   └── main.py         # CLI Entry point
└── ...
```
