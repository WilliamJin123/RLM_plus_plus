import argparse
import sys
import traceback
from pathlib import Path

# Ensure root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(BASE_DIR.as_posix())

# Ensure your python path is set, or run as module
from src.core.indexer import Indexer
from src.core.factory import AgentFactory

def main():
    parser = argparse.ArgumentParser(description="RLM++ CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Ingest Command ---
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text file into the index")
    ingest_parser.add_argument("file", help="Path to the file to ingest")
    # Added DB argument for persistence
    ingest_parser.add_argument("--db", default="rlm_storage.db", help="Path to SQLite database")
    ingest_parser.add_argument("--strategy", choices=["smart", "basic"], default="smart")

    # --- Query Command ---
    query_parser = subparsers.add_parser("query", help="Ask a question to the RLM Agent")
    query_parser.add_argument("text", help="Query text")
    # Added DB argument so Agent connects to the same DB we ingested into
    query_parser.add_argument("--db", default="rlm_storage.db", help="Path to SQLite database")

    args = parser.parse_args()

    if args.command == "ingest":
        print(f"Initializing Indexer (DB: {args.db})...")
        try:
            # Pass the DB path so data persists
            indexer = Indexer(db_path=args.db)
            indexer.ingest_file(args.file)
        except Exception as e:
            print(f"Ingestion failed: {e}")
            traceback.print_exc()
    
    elif args.command == "query":
        try:
            # We assume your AgentFactory/AgentConfig can accept a 'db_path' 
            # override or that it's configured in agents.yaml. 
            # If strictly config-based, ensure agents.yaml points to the same DB as above.
            
            print(f"Initializing Agent for query: '{args.text}'")
            
            # Create the agent
            agent = AgentFactory.create_agent("rlm-agent", session_id="cli_query_session")
            
            # NOTE: If your agent relies on the 'db' from args, you might need 
            # to manually inject it here if it's not in agents.yaml.
            # e.g., agent.storage_engine = StorageEngine(args.db)
            
            print("--- Agent Running ---")
            response = agent.run(args.text)
            
            print("\n=== Final Answer ===")
            print(response.content)
            print("====================")
            
            if hasattr(response, 'metrics') and response.metrics:
                print(f"\nMetrics: {response.metrics}")
            
        except Exception as e:
            print(f"Error running agent: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()