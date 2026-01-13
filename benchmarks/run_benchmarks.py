import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(BASE_DIR.as_posix())

from base import BenchmarkEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified RLM Benchmark Runner")
    parser.add_argument("benchmark", choices=["longbench", "oolong"], help="Top level benchmark name")
    parser.add_argument("subset", help="Subset (e.g., 'code_qa', 'history_qa', 'metaphors', 'negation')")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N items")
    parser.add_argument("--chunk", type=int, default=25000, help="Max chunk size in tokens for ingestion")
    
    args = parser.parse_args()

    strategy = None
    if args.benchmark == "longbench":
        from longbenchv2 import LongBenchLogic
        strategy = LongBenchLogic()
    elif args.benchmark == "oolong":
        from oolong import OolongLogic
        strategy = OolongLogic()
    
    runner = BenchmarkEngine(args.benchmark, args.subset, strategy, max_chunk_tokens=args.chunk)
    runner.run(limit=args.limit)