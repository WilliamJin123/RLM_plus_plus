from typing import Optional

from agno.tools import Toolkit
from agno.tools.python import PythonTools

from src.config.config import CONFIG
from src.core.storage import StorageEngine


class RLMTools(Toolkit):
    def __init__(self, db_path: str, **kwargs):
        self.storage = StorageEngine(db_path)
        tools = [
            self.inspect_document_hierarchy,
            self.examine_summary_node,
            self.search_summaries,
        ]
        super().__init__(name="rlm_tools", tools=tools, **kwargs)

        # Setup model rotator for chunk-analyzer-agent
        from src.core.factory import ModelRotator

        chunk_config = CONFIG.get_agent("chunk-analyzer-agent")
        if chunk_config and chunk_config.model_pool:
            self._chunk_rotator: Optional[ModelRotator] = ModelRotator(
                configs=chunk_config.model_pool.models,
                calls_per_model=chunk_config.model_pool.calls_per_model,
            )
        else:
            self._chunk_rotator = None

    def inspect_document_hierarchy(self) -> str:
        """
        Returns the top-level (root) summaries to give an overview of the document structure.
        Use this to find a starting point for investigation.
        """
        nodes = self.storage.get_root_summaries()
        if not nodes:
            return "No document structure found. The index might be empty."

        output = "Document Root Nodes:\n"
        for node in nodes:
            output += f"<id>{node[0]}</id>\n<text>\n{node[1]}\n</text>\n\n"
        return output

    def examine_summary_node(self, summary_id: int, query: str = "") -> str:
        """
        The primary navigation and analysis tool.

        1. If the node is a FOLDER (high-level), it returns a list of child nodes to explore.
        2. If the node is a LEAF (detailed), it reads the raw text associated with it.
           CRITICAL: If it is a leaf, a Sub-Agent is spawned to read the text and answer 'query'.

        Args:
            summary_id: The ID of the node to inspect.
            query: Required only if examining a specific leaf node to extract details.
                   If just navigating folders, this can be empty.
        """
        node = self.storage.get_node_metadata(summary_id)
        if not node:
            return f"Error: Node ID {summary_id} not found."

        level = node["level"]
        summary_text = node["text"]

        # BRANCH A: High Level (Folder) -> Return Navigation Data
        if level > 0:
            children = self.storage.get_child_summaries(summary_id)
            if not children:
                return f"Node {summary_id} (Level {level}) is empty (no children)."

            output = f"Node {summary_id}\n<level>{level}</level>\n<summary>{summary_text[:75]}...</summary>\n"
            output += f"Contains {len(children)} children.\n<children>\n"
            for child in children:
                output += f"<child_id>{child[0]}</child_id>\n<child_summary>\n{child[1]}</child_summary>\n"
            output += "</children>\n"
            return output

        # BRANCH B: Low Level (Leaf) -> Trigger Sub-Agent
        if not query:
            return (
                f"Node {summary_id} is a Leaf Node containing raw text. "
                f"To prevent context overflow, you must provide a specific 'query' argument "
                f"to analyze this text. (e.g., examine_summary_node({summary_id}, query='What is the specific date mentioned?'))"
            )

        chunk_id = self.storage.get_linked_chunk_id(summary_id)
        if not chunk_id:
            return f"Error: Leaf Node {summary_id} has no linked raw text chunk."

        raw_text = self.storage.get_chunk_text(chunk_id)
        if not raw_text:
            return f"Error: Could not retrieve text for chunk {chunk_id}."

        return self._spawn_sub_agent(raw_text, query)

    def read_neighbor_node(self, current_node_id: int, direction: str) -> str:
        """
        Navigates to the adjacent node in the narrative flow. Useful for reading "next page"
        or "previous page" without going back up to the parent.

        Args:
            current_node_id: The ID of the node you are currently looking at.
            direction: One of "next", "prev" (previous), or "parent".
        """
        valid_directions = {"next", "prev", "parent"}
        if direction not in valid_directions:
            return f"Error: direction must be one of {valid_directions}."

        neighbors = self.storage.get_adjacent_nodes(current_node_id)
        target_id = neighbors.get(direction)

        if target_id is None:
            return f"No {direction} node exists for Node {current_node_id} (It might be the start/end of the section)."

        text = self.storage.get_summary(target_id)
        node_meta = self.storage.get_node_metadata(target_id)

        if not node_meta:
            return f"Error: Could not retrieve metadata for Node {target_id}."

        return (
            f"Navigated {direction} to Node {target_id} (Level {node_meta['level']}).\n"
            f"<content>\n{text}\n</content>"
        )

    def _spawn_sub_agent(
        self, context_text: str, user_query: str, max_retries: int = 3
    ) -> str:
        """Creates a temporary agent with retry and force rotation on failure."""
        from src.core.factory import AgentFactory

        sub_agent = AgentFactory.create_agent("chunk-analyzer-agent")
        prompt = f"<context>\n{context_text}\n</context>\n\n<question>{user_query}</question>"

        for attempt in range(max_retries):
            try:
                # Apply model rotation if configured
                if self._chunk_rotator:
                    model_config = self._chunk_rotator.get_next_config()
                    sub_agent.model = AgentFactory.create_model(model_config)

                response = sub_agent.run(prompt)
                content = response.content

                # Check for provider error in response
                if "Provider returned error" in content or "No endpoints found" in content:
                    if self._chunk_rotator:
                        self._chunk_rotator.force_rotate()
                    continue

                return f"<subagent>{content}</subagent>"

            except Exception as e:
                if self._chunk_rotator:
                    self._chunk_rotator.force_rotate()
                if attempt == max_retries - 1:
                    return f"Error in sub-agent execution: {e}"

        return "Error: All retry attempts failed for sub-agent."

    def search_summaries(self, query: str) -> str:
        """Keyword search through the summary tree to find relevant starting nodes."""
        if not query or not query.strip():
            return "Error: Query cannot be empty."

        matches = self.storage.search_summaries(query.strip())
        if not matches:
            return f"No matches found for '{query}'."

        output = f"Search Results for '{query}':\n"
        for node_id, level, text in matches:
            snippet = text[:150] if text else ""
            output += f"- <id>{node_id}</id>\n<level>{level}</level>\n<summary_snippet>{snippet}...</summary_snippet>\n"
        return output


TOOL_REGISTRY = {
    "RLMTools": RLMTools,
    "PythonTools": PythonTools,
}