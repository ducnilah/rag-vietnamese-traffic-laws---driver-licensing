#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_rag.state.service import ConversationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed test user for local chat testing.")
    parser.add_argument("--db-url", default="sqlite:///data/app.db")
    parser.add_argument("--user-id", default="u1")
    parser.add_argument("--email", default="u1@example.com")
    parser.add_argument("--thread-title", default=None, help="Optional: create a starter thread")
    args = parser.parse_args()

    svc = ConversationService(args.db_url)
    svc.create_schema()
    svc.ensure_user(args.user_id, args.email)

    output = {
        "ok": True,
        "user_id": args.user_id,
        "email": args.email,
        "db_url": args.db_url,
    }

    if args.thread_title:
        thread = svc.create_thread(args.user_id, args.thread_title)
        output["thread"] = thread.__dict__

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
