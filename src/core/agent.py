import yaml
from pathlib import Path
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing
from agno.tools.python import PythonTools
from src.config import config
from src.core import get_model
from src.core.monitor_bus import monitor_bus, Event
from src.tools.file_tools import (
    get_document_structure, 
    get_summary_children, 
    read_chunk, 
    search_summaries
)

class RLMAgent:
    def __init__(self, max_steps: int = 20):
        self.model = get_model()
        self.instructions = self._load_instructions()
        db_path = Path(__file__).resolve().parent.parent / "data" / "rlm_agent.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent = Agent(
            id="rlm-agent",
            name="RLM++ Agent",
            model=self.model,
            tools=[
                get_document_structure,
                get_summary_children,
                read_chunk,
                search_summaries,
                PythonTools()
            ],
            description="Recursive Language Model (RLM++) Agent",
            instructions=self.instructions + [f"You have a maximum of {max_steps} steps to find the answer."],
            markdown=True,
            db=SqliteDb(
                db_file=db_path.as_posix(),
                session_table="rlm_agent_sessions",
            )
        )
        setup_tracing(db=self.agent.db, batch_processing=True)


    def _load_instructions(self):
        try:
            path = Path("src/prompts/agent_prompt.yaml")
            if path.exists():
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get("instructions", [])
        except Exception as e:
            print(f"Error loading prompts: {e}")
        
        # Fallback
        return ["You are a helpful agent."]

    def run(self, query: str):
        monitor_bus.publish(Event(type="agent_thought", content=f"Starting query: {query}"))
        response = self.agent.run(query)

        print(f"--- Query Completed ---")
        print(f"Metrics: {response.metrics}")
            
        monitor_bus.publish(Event(type="agent_thought", content=f"Finished with response: {response.content[:100]}..."))
        return response

if __name__ == "__main__":
    # Test agent (requires DB to be populated)
    rlm = RLMAgent()
    rlm.run("What is the main topic of the document?")
