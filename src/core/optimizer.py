import yaml
import os
from pathlib import Path
from agno.agent import Agent
from src.config.config import config
from src.core.factory import AgentFactory

class Optimizer:
    def __init__(self):
        self.agent = AgentFactory.create_agent("optimizer-agent")

    def optimize_prompts(self, failure_reason: str):
        prompt_path = Path("src/prompts/agent_prompt.yaml")
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
        # This would involve generating python code and a test. 
        # For this prototype, we'll just stub the logic.
        print(f"Tool generation requested: {tool_request}")
        # Logic: 
        # 1. Generate code
        # 2. Write to src/tools/dynamic/new_tool.py
        # 3. Write test to tests/test_new_tool.py
        # 4. Run pytest
        # 5. If pass, keep it.
