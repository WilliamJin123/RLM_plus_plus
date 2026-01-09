import importlib.util
import os
import inspect
from pathlib import Path
from typing import List, Callable, Dict, Any, Union
from agno.tools.python import PythonTools

class ToolRegistry:
    def __init__(self, dynamic_tools_dir: str = "src/tools/dynamic"):
        self.dynamic_tools_dir = Path(dynamic_tools_dir)
        self.dynamic_tools_dir.mkdir(parents=True, exist_ok=True)
        # Create init if not exists
        if not (self.dynamic_tools_dir / "__init__.py").exists():
            (self.dynamic_tools_dir / "__init__.py").touch()
            
    def get_tool_map(self) -> Dict[str, Union[Callable, Any]]:
        """
        Returns a dictionary mapping tool names to functions or tool instances.
        """
        from src.tools.file_tools import (
            get_document_structure, 
            get_summary_children, 
            read_chunk, 
            search_summaries
        )
        from src.tools.architect_tools import (
            update_instructions,
            add_tool,
            remove_tool,
            update_model_params
        )
        from src.tools.context_tools import get_agent_history
        
        # Standard tools
        tool_map = {
            "get_document_structure": get_document_structure,
            "get_summary_children": get_summary_children,
            "read_chunk": read_chunk,
            "search_summaries": search_summaries,
            "PythonTools": PythonTools(), # Instantiated by default, or we could handle instantiation in factory
            # Architect Tools
            "update_instructions": update_instructions,
            "add_tool": add_tool,
            "remove_tool": remove_tool,
            "update_model_params": update_model_params,
            "get_agent_history": get_agent_history
        }
        
        # Load dynamic tools
        for file in self.dynamic_tools_dir.glob("*.py"):
            if file.name == "__init__.py":
                continue
            
            module_name = f"src.tools.dynamic.{file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find functions in module that are valid tools
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and obj.__module__ == module_name:
                            # Simple heuristic: if it has a docstring and isn't private
                            if obj.__doc__ and not name.startswith("_"):
                                tool_map[name] = obj
            except Exception as e:
                print(f"Failed to load tool from {file}: {e}")
                
        return tool_map

    def get_all_tools(self) -> List[Any]:
        """
        Legacy method: returns list of tools.
        """
        return list(self.get_tool_map().values())

registry = ToolRegistry()
