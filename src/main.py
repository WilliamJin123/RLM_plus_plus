import argparse
import sys
from src.core.indexer import Indexer
from src.core.monitor_bus import monitor_bus, Event
from src.core.factory import AgentFactory
from src.core.config_store import init_config_db, SessionLocal, AgentConfig
from src.utils.migrate_v3 import migrate

def ensure_config_exists():
    # Ensure tables exist first
    try:
        init_config_db()
    except Exception as e:
        print(f"Error initializing DB schema: {e}")

    db = SessionLocal()
    try:
        # Check if empty
        if db.query(AgentConfig).count() == 0:
            print("Config DB empty. Running migration...")
            migrate()
    except Exception as e:
        print(f"Error checking config DB: {e}. Running migration might fix it.")
        migrate()
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="RLM++ CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("file", help="File to ingest")
    ingest_parser.add_argument("--strategy", choices=["smart", "basic"], default="smart")

    # Query
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--monitor", action="store_true", help="Enable Overseer")

    # Evolve
    evolve_parser = subparsers.add_parser("evolve")
    evolve_parser.add_argument("--reason", help="Reason for evolution (simulated failure)")

    args = parser.parse_args()

    ensure_config_exists()

    if args.command == "ingest":
        # We keep Indexer as class for now, as it's not fully migrated to Agent config yet
        indexer = Indexer()
        indexer.ingest_file(args.file)
    
    elif args.command == "query":
        # Use Factory to create RLM Agent
        try:
            agent = AgentFactory.create_agent("rlm-agent", session_id="cli_query_session")
            
            if args.monitor:
                # If monitor is requested, we might want to start the overseer thread
                # For now, just logging to bus
                monitor_bus.publish(Event(type="system", content="Monitoring enabled (basic logging)."))

            monitor_bus.publish(Event(type="agent_thought", content=f"Starting query: {args.text}"))
            
            response = agent.run(args.text)
            
            print("\nFinal Answer:")
            print(response.content)
            
            # Print metrics if available
            if hasattr(response, 'metrics'):
                print(f"\nMetrics: {response.metrics}")
                
            monitor_bus.publish(Event(type="agent_thought", content="Query Completed."))
            
        except Exception as e:
            print(f"Error running agent: {e}")
            import traceback
            traceback.print_exc()

    elif args.command == "evolve":
        try:
            architect = AgentFactory.create_agent("architect", session_id="cli_evolve_session")
            reason = args.reason or "Routine maintenance"
            print(f"Architect starting. Reason: {reason}")
            
            # The Architect acts on the system configuration
            prompt = (
                f"The system administrator has requested an evolution/optimization cycle.\n"
                f"Reason: {reason}\n"
                f"Please review the available agents and their configurations. "
                f"If you see any opportunity for improvement based on your knowledge or history, make changes.\n"
                f"Current task: Check 'rlm-agent' and 'overseer' configs."
            )
            
            response = architect.run(prompt)
            print("\nArchitect Report:")
            print(response.content)
            
        except Exception as e:
            print(f"Error running architect: {e}")
            import traceback
            traceback.print_exc()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()