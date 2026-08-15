"""
cli.py
------
Simple command-line interface.

  # 1. Put PDFs/.txt papers into data/papers/, then build the index:
  python cli.py index

  # 2. Ask questions against the indexed papers:
  python cli.py query "What dataset was used for evaluation?"

  # 3. Or just chat interactively:
  python cli.py chat
"""

import argparse
import sys

from pipeline import RAGPipeline
import config


def cmd_index(args):
    rag = RAGPipeline()
    rag.build_index()
    print("\nIndex built successfully. You can now run: python cli.py query \"...\"")


def cmd_query(args):
    rag = RAGPipeline()
    rag.load_index()
    answer = rag.query(args.question, top_n=args.top_n)
    print("\n" + "=" * 70)
    print(f"Q: {args.question}")
    print("=" * 70)
    print(answer.render())


def cmd_chat(args):
    rag = RAGPipeline()
    rag.load_index()
    print("RAG chat over your indexed papers. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = rag.query(question)
        print("\n" + answer.render() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Local hybrid-search RAG for research papers.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Ingest papers from data/papers/ and build the indexes.")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="Ask a single question.")
    p_query.add_argument("question", type=str)
    p_query.add_argument("--top-n", type=int, default=config.RERANK_TOP_N)
    p_query.set_defaults(func=cmd_query)

    p_chat = sub.add_parser("chat", help="Interactive Q&A loop.")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
