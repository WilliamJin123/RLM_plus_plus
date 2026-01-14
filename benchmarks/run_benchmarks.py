import argparse
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from base import BenchmarkEngine

AVAILABLE_BENCHMARKS = {
    "longbench": "longbenchv2",
    "oolong": "oolong",
}


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RLM Benchmark Runner")
    parser.add_argument(
        "benchmark",
        choices=list(AVAILABLE_BENCHMARKS.keys()),
        help="Benchmark name",
    )
    parser.add_argument(
        "subset",
        help="Subset (e.g., 'code_qa', 'history_qa', 'metaphors', 'negation')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N items",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=25000,
        help="Max chunk size in tokens for ingestion (default: 25000)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Optional directory to store DB files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Import the appropriate strategy
    strategy = None
    if args.benchmark == "longbench":
        from longbenchv2 import LongBenchLogic

        strategy = LongBenchLogic()
    elif args.benchmark == "oolong":
        from oolong import OolongLogic

        strategy = OolongLogic()

    if not strategy:
        logger.error("Unknown benchmark: %s", args.benchmark)
        sys.exit(1)

    runner = BenchmarkEngine(
        args.benchmark,
        args.subset,
        strategy,
        max_chunk_tokens=args.chunk,
        db_output_dir=args.db_dir,
    )

    try:
        runner.run(limit=args.limit)
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Benchmark failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
