from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_law_v2.evaluation import run_retrieval_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 retrieval evaluation.")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()
    print(json.dumps(run_retrieval_eval(Path(args.index_dir)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
