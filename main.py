#!/usr/bin/env python3
"""
RLM CLI - Query large contexts using Recursive Language Models

Usage:
    python main.py "your question" --file context.txt
    python main.py "your question" --context "inline context here"
    echo "context" | python main.py "your question" --stdin
"""

import argparse
import asyncio
import sys

from rlm import RLM
from rlm.rlm import ROOT_MODEL, WORKER_MODEL


async def main():
    parser = argparse.ArgumentParser(
        description="RLM - Query large contexts with recursive language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Find all error messages" --file server.log
  python main.py "Summarize this code" --file large_codebase.py
  python main.py "What is the main topic?" --context "Your text here..."

  # Use smarter root model with cheaper workers:
  python main.py "Find X" --file doc.txt --root-model openai/gpt-4.1 --worker-model openai/gpt-4.1-mini
        """,
    )

    parser.add_argument(
        "question",
        help="Question or instruction to process against the context",
    )

    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument(
        "--file", "-f",
        help="Path to file containing the context",
    )
    context_group.add_argument(
        "--context", "-c",
        help="Inline context string",
    )
    context_group.add_argument(
        "--stdin", "-s",
        action="store_true",
        help="Read context from stdin",
    )

    parser.add_argument(
        "--root-model", "-r",
        default=ROOT_MODEL,
        help=f"Model for root planning/coordination (default: {ROOT_MODEL})",
    )

    parser.add_argument(
        "--worker-model", "-w",
        default=WORKER_MODEL,
        help=f"Model for sub_llm workers (default: {WORKER_MODEL})",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print tool calls, trace, and stats",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum tool call iterations (default: 30)",
    )

    args = parser.parse_args()

    # Load context
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                context = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.context:
        context = args.context
    else:  # stdin
        context = sys.stdin.read()

    if not context.strip():
        print("Error: Empty context provided", file=sys.stderr)
        sys.exit(1)

    # Show context info
    if args.verbose:
        print(f"Context: {len(context):,} chars, {context.count(chr(10))+1:,} lines")
        print(f"Root model: {args.root_model}")
        print(f"Worker model: {args.worker_model}")
        print(f"Question: {args.question}")
        print("-" * 50)

    # Run RLM
    rlm = RLM(
        root_model=args.root_model,
        worker_model=args.worker_model,
        max_iterations=args.max_iterations,
    )

    try:
        answer = await rlm.query(args.question, context, verbose=args.verbose)
        print(answer)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
