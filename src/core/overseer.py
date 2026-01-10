from agno.agent import Agent
from agno.models.openai import OpenAIChat
from src.config.config import config
from src.core.monitor_bus import monitor_bus, Event

class OverseerAgent:
    def __init__(self):
        # The Overseer needs a smart model to judge the RLM's performance
        self.model = OpenAIChat(id=config.REASONING_MODEL_NAME)
        self.agent = Agent(
            model=self.model,
            description="You are the Overseer. You monitor an autonomous RLM agent.",
            instructions=[
                "Analyze the stream of events from the RLM agent.",
                "Look for LOOPS (repeating the same query/tool) or STAGNATION (getting nowhere).",
                "If the agent is stuck, generate a helpful HINT or DIRECTION.",
                "If the agent is doing well, return 'PASS'.",
                "Be concise."
            ]
        )
        # Subscribe to the bus
        monitor_bus.subscribe(self.on_event)
        self.recent_tool_calls = []

    def on_event(self, event: Event):
        # Truncate content to avoid context explosion
        content = event.content
        if len(content) > 500:
            content = content[:500] + "...(truncated)"
            
        if event.type == 'tool_call':
            self.recent_tool_calls.append(content)
            self._check_health()
        
        # Optional: Print that overseer is alive
        # print(f"[Overseer] Processed {event.type}")

    def _check_health(self):
        # Simple heuristic: if last 3 tool calls are identical, alarm.
        if len(self.recent_tool_calls) >= 3:
            last_3 = self.recent_tool_calls[-3:]
            if all(x == last_3[0] for x in last_3):
                print("OVERSEER ALERT: Loop detected.")
                self.intervene("You seem to be repeating the same action. Try a different approach.")
                self.recent_tool_calls = [] # Reset
        
        # Only check occasionally or if list gets too long
        if len(self.recent_tool_calls) > 10:
            self.recent_tool_calls = self.recent_tool_calls[-5:]


    def intervene(self, message: str):
        # In a real system, this would push a message to the RLM's context.
        # For this prototype, we log it as a System Message event which the RLM *should* read if it were event-driven.
        # However, since RLM is currently a linear script, we can just print it or throw an exception to be caught.
        print(f"\n[OVERSEER INTERVENTION]: {message}\n")
        monitor_bus.publish(Event(type="system_message", content=message))

overseer = OverseerAgent()
