import argparse
import sys
from src.core.indexer import Indexer

from src.core.factory import AgentFactory

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


    args = parser.parse_args()

    if args.command == "ingest":
        # We keep Indexer as class for now, as it's not fully migrated to Agent config yet
        indexer = Indexer()
        indexer.ingest_file(args.file)
    
    elif args.command == "query":
        # Use Factory to create RLM Agent
        try:
            agent = AgentFactory.create_agent("rlm-agent", session_id="cli_query_session")
            
            print(f"Starting query: {args.text}")
            
            response = agent.run(args.text)
            
            print("\nFinal Answer:")
            print(response.content)
            
            # Print metrics if available
            if hasattr(response, 'metrics'):
                print(f"\nMetrics: {response.metrics}")
                
            print("Query Completed.")
            
        except Exception as e:
            print(f"Error running agent: {e}")
            import traceback
            traceback.print_exc()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
