import yaml
import os
import re
from pathlib import Path
from agno.agent import Agent
from src.config.config import config
from src.core.factory import AgentFactory

class Optimizer:
    def __init__(self):
        self.agent = AgentFactory.create_agent("optimizer-agent")

    def optimize_prompts(self, failure_reason: str):
        # Use absolute path for robustness
        prompt_path = Path(__file__).resolve().parents[2] / "src" / "prompts" / "agent_prompt.yaml"
        current_prompts = ""
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                current_prompts = f.read()

        prompt = f"""
        Current Instructions:
        {current_prompts}

        Failure Analysis:
        {failure_reason}

        Task:
        Rewrite the instructions to prevent this failure. Keep the format (YAML list).
        Return ONLY the YAML content.
        """
        
        response = self.agent.run(prompt)
        new_yaml = response.content
        if "```yaml" in new_yaml:
            new_yaml = new_yaml.split("```yaml")[1].split("```")[0].strip()
        elif "```" in new_yaml:
            new_yaml = new_yaml.split("```")[1].strip()
            
        with open(prompt_path, 'w') as f:
            f.write(new_yaml)
        print("Optimized prompts updated.")

    def create_tool(self, tool_request: str):
        print(f"Tool generation requested: {tool_request}")
        
        prompt = f"""
        You are an expert Python developer.
        Task: Create a standalone Python function that performs the following task:
        "{tool_request}"
        
        Requirements:
        1. Return ONLY valid Python code.
        2. The code must contain a single function definition.
        3. Include a docstring describing the function's purpose, arguments, and return value.
        4. Do not include any usage examples or extra text outside the function definition.
        5. Use standard libraries where possible.
        6. The function name should be snake_case and descriptive.
        
        Example Output:
        def my_tool(arg1: str) -> str:
            \"\"\"
            Description of tool.
            \"\"\"
            return "result"
        """
        
        response = self.agent.run(prompt)
        content = response.content
        
        # Extract code block
        code = content
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            code = content.split("```")[1].strip()
            
        # Extract function name
        match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
        if match:
            func_name = match.group(1)
            
            # Save to dynamic tools directory
            # Use absolute path for robustness
            dynamic_tools_dir = Path(__file__).resolve().parents[2] / "src" / "tools" / "dynamic"
            dynamic_tools_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = dynamic_tools_dir / f"{func_name}.py"
            
            with open(file_path, "w") as f:
                f.write(code)
                
            print(f"Dynamic tool '{func_name}' created at {file_path}")
        else:
            print("Failed to extract function name from generated code.")
            print("Generated content:", content)
