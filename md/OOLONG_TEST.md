# Oolong Benchmark Test

This document summarizes the refactoring of `benchmarks/test_oolong.py` to support the Oolong dataset.

## Changes Implemented

1.  **Dataset Loading**:
    -   Utilized the `datasets` library to load Oolong parquet files (`metaphors_1024000_plus.parquet` and `negation_1024000_plus.parquet`).
    -   Added a `load_oolong_dataset` function that selects the specified subset.

2.  **Context Ingestion**:
    -   Implemented `ingest_context` which writes the `context_window_text` to a temporary file and ingests it using the `Indexer`.
    -   Uses a target chunk size of 25,000 tokens (similar to LongBenchV2 settings) to handle large contexts.

3.  **Agent Interaction**:
    -   Refactored the `run_benchmark` loop to:
        -   Reset the temporary database (`data/oolong_temp.db`) for each item.
        -   Initialize a fresh `Indexer`.
        -   Create a new `rlm-agent` with a unique session ID.
        -   Prompt the agent with the question and instructions to follow the "Label: answer" format.

4.  **Evaluation**:
    -   Implemented `evaluate_answer` to compare the agent's response with the ground truth.
    -   The ground truth is stored as a stringified list (e.g., `"['correct']"`). The evaluator parses this and compares it against the agent's output, looking for the pattern `Label: <answer>` or checking the final word.

## Usage

Run the benchmark using the following command:

```bash
python benchmarks/test_oolong.py <subset> [--limit <N>]
```

-   `<subset>`: Choose between `metaphors` or `negation`.
-   `--limit <N>`: (Optional) Process only the first N items.

**Example:**

```bash
python benchmarks/test_oolong.py metaphors --limit 1
```
