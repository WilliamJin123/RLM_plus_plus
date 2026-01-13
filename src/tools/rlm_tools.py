from typing import List, Optional
from agno.agent import Agent
from src.core.storage import StorageEngine
from agno.tools import Toolkit
from agno.tools.python import PythonTools
class RLMTools(Toolkit):
    def __init__(self, db_path: str, **kwargs):
        self.storage = StorageEngine(db_path)
        tools = [
            self.inspect_document_hierarchy,
            self.examine_summary_node,
            self.search_summaries,
        ]
        super().__init__(name="rlm_tools", tools=tools, **kwargs)
        
    def inspect_document_hierarchy(self) -> str:
        """
        Returns the top-level (root) summaries to give an overview of the document structure.
        Use this to find a starting point for investigation.
        """
        ids, texts = self.storage.get_root_summaries()
        if not ids:
            return "No document structure found. The index might be empty."
        
        output = "Document Root Nodes:\n"
        for i, t in zip(ids, texts):
            output += f"<id>{i}</id>\n<text>\n{t}\n</text>\n\n"
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
        # Fetch Metadata
        node = self.storage.get_node_metadata(summary_id)
        if not node:
            return f"Error: Node ID {summary_id} not found."

        level = node['level']
        summary_text = node['text']

        # ROUTING LOGIC
        
        # --- BRANCH A: High Level (Folder) -> Return Navigation Data ---
        if level > 0:
            children = self.storage.get_child_summaries(summary_id)
            if not children:
                return f"Node {summary_id} (Level {level}) is empty (no children)."
            # summary of the node should always be in context already
            output = f"Node {summary_id}\n <level>{level}</level>\n <summary>{summary_text[:75]}...</summary>\n"
            output += f"Contains {len(children)} children.\n<children>\n"
            for child in children:
                # Provide ID and a snippet so the Agent knows which one to pick next
                output += f"<child_id>{child[0]}</child_id>\n<child_summary>\n{child[1]}</child_summary>\n"
            output += "</children>\n"
            return output

        # --- BRANCH B: Low Level (Leaf) -> Trigger Sub-Agent ---
        else:
            # Level 0 means this links to a raw text Chunk.
            if not query:
                return (
                    f"Node {summary_id} is a Leaf Node containing raw text. "
                    f"To prevent context overflow, you must provide a specific 'query' argument "
                    f"to analyze this text. (e.g., examine_summary_node({summary_id}, query='What is the specific date mentioned?'))"
                )
            # Retrieve the linked raw chunk ID
            chunk_id = self.storage.get_linked_chunk_id(summary_id)
            if not chunk_id:
                return f"Error: Leaf Node {summary_id} has no linked raw text chunk."
            
            # Retrieve the HEAVY text (hidden from main context)
            raw_text = self.storage.get_chunk_text(chunk_id)
            
            # Spawn the Sub-Agent
            return self._spawn_sub_agent(raw_text, query)

    def _spawn_sub_agent(self, context_text: str, user_query: str) -> str:
        """
        Private method. Creates a temporary agent to read large context 
        and return a concise answer.
        """
        from src.core.factory import AgentFactory

        try:
            # Create a lightweight agent specifically for reading comprehension
            sub_agent = AgentFactory.create_agent("chunk-analyzer-agent")
            
            prompt = (
                f"<context>\n{context_text}\n</context>\n\n"
                f"<question>{user_query}</question>"
            )
            
            response = sub_agent.run(prompt)
            return f"<subagent>{response.content}</subagent>"
            
        except Exception as e:
            return f"Error in sub-agent execution: {str(e)}"

    def search_summaries(self, query: str) -> str:
        """
        Keyword search through the summary tree to find relevant starting nodes.
        """
        matches = self.storage.search_summaries(query)
        if not matches:
            return "No matches found."
            
        output = f"Search Results for '{query}':\n"
        for m in matches:
            # m = (id, level, text)
            output += f"- <id>{m[0]}</id>\n<level>{m[1]}</level>\n<summary_snippet>{m[2][:150]}...</summary_snippet>\n"
        return output



TOOL_REGISTRY = {
    "RLMTools": RLMTools,
    "PythonTools": PythonTools,
}