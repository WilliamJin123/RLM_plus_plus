import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.factory import AgentFactory
from src.core.indexer import Indexer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handle the ingest command."""
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error("File not found: %s", args.file)
        return 1

    logger.info("Initializing Indexer (DB: %s)...", args.db)
    try:
        indexer = Indexer(db_path=args.db, strategy=args.strategy)
        indexer.ingest_file(str(file_path))
        logger.info("Ingestion complete.")
        return 0
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        return 1


def cmd_query(args: argparse.Namespace) -> int:
    """Handle the query command."""
    try:
        logger.info("Initializing Agent for query: '%s'", args.text)
        agent = AgentFactory.create_agent(
            "rlm-agent",
            content_db_path=args.db,
            session_id="cli_query_session",
        )

        logger.info("Running agent...")
        response = agent.run(args.text)

        print("\n=== Final Answer ===")
        print(response.content)
        print("====================")

        if hasattr(response, "metrics") and response.metrics:
            print(f"\nMetrics: {response.metrics}")

        return 0
    except Exception as e:
        logger.exception("Error running agent: %s", e)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RLM++ CLI")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text file into the index")
    ingest_parser.add_argument("file", help="Path to the file to ingest")
    ingest_parser.add_argument(
        "--db",
        default="rlm_storage.db",
        help="Path to SQLite database (default: rlm_storage.db)",
    )
    ingest_parser.add_argument(
        "--strategy",
        choices=["fixed", "llm"],
        default="fixed",
        help="Chunking strategy (default: fixed)",
    )

    # Query Command
    query_parser = subparsers.add_parser("query", help="Ask a question to the RLM Agent")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument(
        "--db",
        default="rlm_storage.db",
        help="Path to SQLite database (default: rlm_storage.db)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "query":
        return cmd_query(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
