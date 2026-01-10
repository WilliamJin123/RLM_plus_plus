import yaml
import os
from pathlib import Path
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from src.config.config import config

class Optimizer:
    def __init__(self):
        self.model = OpenAIChat(id=config.REASONING_MODEL_NAME)
        self.agent = Agent(
            model=self.model,
            description="You are a System Optimizer.",
            instructions=[
                "Analyze the failure report from the RLM agent.",
                "Decide if the failure is due to bad PROMPTS or missing TOOLS.",
                "If bad prompts: rewrite the instruction in 'src/prompts/agent_prompt.yaml'.",
                "If missing tools: write a NEW python tool in 'src/tools/dynamic/'.",
                "Output your action as JSON."
            ]
        )

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
