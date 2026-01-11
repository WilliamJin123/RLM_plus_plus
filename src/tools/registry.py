import importlib.util
import os
import inspect
from pathlib import Path
from typing import List, Callable, Dict, Any, Union
from agno.tools.python import PythonTools

class ToolRegistry:
    def __init__(self):
        pass
            
    def get_tool_map(self) -> Dict[str, Union[Callable, Any]]:
        """
        Returns a dictionary mapping tool names to functions or tool instances.
        """
        from src.tools.file_tools import (
            get_document_structure, 
            get_summary_children, 
            analyze_chunk, 
            search_summaries
        )
        
        # Standard tools
        tool_map = {
            "get_document_structure": get_document_structure,
            "get_summary_children": get_summary_children,
            "analyze_chunk": analyze_chunk,
            "search_summaries": search_summaries,
            "PythonTools": PythonTools(), # Instantiated by default, or we could handle instantiation in factory
        }
        
        return tool_map

    def get_all_tools(self) -> List[Any]:
        """
        Legacy method: returns list of tools.
        """
        return list(self.get_tool_map().values())

registry = ToolRegistry()
