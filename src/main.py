import argparse
import sys
from src.core.indexer import Indexer
from src.core.agent import RLMAgent
from src.core.monitor_bus import monitor_bus
from src.core.overseer import overseer # Start overseer
from src.core.optimizer import Optimizer

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

    if args.command == "ingest":
        indexer = Indexer()
        # Triggering the smart ingest path we built
        indexer.ingest_file(args.file)
    
    elif args.command == "query":
        agent = RLMAgent()
        response = agent.run(args.text)
        print("\nFinal Answer:")
        print(response.content)

    elif args.command == "evolve":
        optimizer = Optimizer()
        if args.reason:
            print(f"Optimizing for reason: {args.reason}")
            optimizer.optimize_prompts(args.reason)
        else:
            print("Please provide a --reason for evolution.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
