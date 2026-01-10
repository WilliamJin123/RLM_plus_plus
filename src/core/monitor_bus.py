from typing import List, Dict, Any
from dataclasses import dataclass, field
import datetime
import functools

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

def monitored_tool(func):
    """
    Decorator that wraps a tool function to publish events to the monitor_bus.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Truncate args for logging
        args_str = str(args)
        kwargs_str = str(kwargs)
        if len(args_str) > 500:
            args_str = args_str[:500] + "...(truncated)"
        if len(kwargs_str) > 500:
            kwargs_str = kwargs_str[:500] + "...(truncated)"
            
        monitor_bus.publish(Event(
            type="tool_call", 
            content=f"Calling {func.__name__}({args_str}, {kwargs_str})"
        ))
        
        try:
            result = func(*args, **kwargs)
            
            # Truncate result for logging
            result_str = str(result)
            if len(result_str) > 500:
                result_log = result_str[:500] + "...(truncated)"
            else:
                result_log = result_str
                
            monitor_bus.publish(Event(
                type="tool_result", 
                content=f"Result from {func.__name__}: {result_log}"
            ))
            return result
            
        except Exception as e:
            monitor_bus.publish(Event(
                type="error", 
                content=f"Error in {func.__name__}: {str(e)}"
            ))
            raise e
            
    return wrapper
