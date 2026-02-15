"""
analytics_app.py

One-shot Analytics mode:
- Fetch chapter stats from MCP tools
- Print a simple HTML bar chart (renders nicely in your Gradio transcript bubble)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.mcp_client.mcp_tools import PracticeResultsClient

import contextlib
import io

# Match your practice identity (or change as needed)
STUDENT_ID = "student_001"


async def main() -> None:
    load_dotenv()

    # NOTE: ensure this path matches how you run MCP in practice_app.py
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        async with PracticeResultsClient("mcp_server/server.py") as client:
            stats = await client.get_chapter_stats(STUDENT_ID)


    # If your MCP tool returns a plain dict, this works:
    chapters = stats.get("chapters", []) if isinstance(stats, dict) else []

    if not chapters:
        print("No progress history yet.")
        return

    print("Progress Report:\n")
    for row in chapters:
        print(f"Chapter {row['chapter_id']}: {row['avg_percent']:.1f}% average over {row['quizzes']} quiz(es)\n")

    weakest = chapters[0]["chapter_id"]
    print(f"\nNext suggestion: practice Chapter {weakest} again.\n")


if __name__ == "__main__":
    asyncio.run(main())
