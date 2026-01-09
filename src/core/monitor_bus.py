from typing import List, Dict, Any
from dataclasses import dataclass, field
import datetime

@dataclass
class Event:
    type: str # 'tool_call', 'agent_thought', 'error', 'system_message'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class MonitorBus:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MonitorBus, cls).__new__(cls)
            cls._instance.events = []
            cls._instance.subscribers = []
        return cls._instance

    def publish(self, event: Event):
        self.events.append(event)
        # Notify subscribers (like the Overseer)
        for sub in self.subscribers:
            sub(event)

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def get_recent_events(self, limit=10) -> List[Event]:
        return self.events[-limit:]

monitor_bus = MonitorBus()
