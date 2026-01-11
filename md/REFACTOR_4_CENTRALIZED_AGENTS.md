# Refactor 4: Centralized Agent Configuration & Sub-Agent Isolation

## Overview
This refactor completes the transition to a fully configuration-driven architecture ("Liquid Agent" pattern). Previously, several helper agents (Indexers, Segmenters, Analyzers) were hardcoded in Python files, making them invisible to the configuration system. This update centralizes all agent definitions into `src/config/agents.yaml` and enforces strict "Read-Only" protection for specialized sub-agents to prevent the automated Architect from destabilizing them.

## Changes Implemented

### 1. Centralized Configuration (`src/config/agents.yaml`)
- **Consolidation**: Moved hardcoded agent definitions from Python code into the YAML config.
- **New Agents Added**:
    - `summarization-agent`: For high-speed text compression during ingestion.
    - `smart-ingest-agent`: For semantic segmentation of text buffers.
    - `chunk-analyzer-agent`: For precise reading of specific text chunks during RAG queries.
    - `history-analyzer-agent`: For summarizing verbose logs for the Architect/Overseer.
    - `optimizer-agent`: For the meta-reasoning logic of the Optimizer.
- **New Flag**: Introduced `readonly_model: true/false` in the configuration schema.

### 2. Enforced Restrictions (`src/tools/architect_tools.py`)
- Updated `update_model_params` to check the `readonly_model` flag.
- **Impact**: The Architect agent can no longer swap the models of specialized sub-agents (e.g., forcing a 70B model onto a lightweight summarizer), ensuring system stability and cost control.

### 3. Factory Refactor (`src/core/factory.py`)
- Updated `AgentFactory.create_agent` to handle ephemeral agents (agents with `storage: null`).
- Removed the requirement for a database path if storage is not configured, allowing for lightweight, in-memory agent instantiation.

### 4. Codebase Migration
- Refactored `src/core/indexer.py`, `src/core/smart_ingest.py`, `src/tools/file_tools.py`, `src/tools/context_tools.py`, and `src/core/optimizer.py` to use `AgentFactory.create_agent()` instead of direct `Agent()` class instantiation.

## Sub-Agent Breakdown

The system now explicitly defines and separates these specialized roles:

1.  **Summarization Agent (`summarization-agent`)**
    *   **Role:** Compresses raw text chunks into hierarchical summaries.
    *   **Why Separate:** Requires a density-focused prompt distinct from the conversational main agent.

2.  **Smart Ingest Agent (`smart-ingest-agent`)**
    *   **Role:** Analyzes text buffers to find semantic cut points (e.g., paragraph ends) for chunking.
    *   **Why Separate:** Requires strict JSON output and logic-heavy processing, independent of the main query context.

3.  **Chunk Analyzer Agent (`chunk-analyzer-agent`)**
    *   **Role:** Reads a specific ~1000 token chunk to answer a targeted query from the RLM Agent.
    *   **Why Separate:** Prevents the RLM Agent's context from flooding with raw text. The RLM delegates the "reading" and receives only the "answer".

4.  **History Analyzer Agent (`history-analyzer-agent`)**
    *   **Role:** Summarizes long execution logs for the Overseer or Architect.
    *   **Why Separate:** Keeps the meta-agents' context windows clean by distilling thousands of log lines into insights.

5.  **Optimizer Agent (`optimizer-agent`)**
    *   **Role:** Analyzes failure reports to generate prompt updates or new tools.
    *   **Why Separate:** Performs meta-reasoning on the system configuration itself.
