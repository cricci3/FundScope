"""
ask.py — Interactive entrypoint for FundScope (thin wrapper over the agent)

The keyword router that used to live here (query type, ISINs and doc_type
guessed from hard-coded keywords) has been removed: questions now go to the
tool-calling agent in agent.py, which decides itself which funds, documents
and retrieval mode to search. This file keeps the familiar name and commands.

Usage:
    # Interactive mode (commands: funds, chunks, calls, reset, help, exit)
    python src/ask.py

    # Single question mode
    python src/ask.py --query "What is the TER of the iShares MSCI World ETF?"

    # Show the retrieved chunks / the tool calls / the cost of each answer
    python src/ask.py --show_chunks --show_calls --show_cost

    # Use a different provider / model
    python src/ask.py --provider groq --model <model-id>
"""

from agent import main

if __name__ == "__main__":
    main()
