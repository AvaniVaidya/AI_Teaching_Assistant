"""
mcp_server.py

FastMCP server exposing tools to store and analyze quiz results in CSV files.

Tools:
- save_quiz_result: stores final quiz score for a chapter (one row per quiz attempt)
- get_chapter_stats: returns chapter-wise averages for a student

Storage:
- practice_data/quiz_results.csv

Run:
  python mcp_server.py
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List
from pathlib import Path

from mcp.server.fastmcp import FastMCP  # FastMCP server class


# Create the MCP server instance
mcp = FastMCP(name="Class5 Practice Results Server")


# Storage locations
QUIZ_RESULTS_CSV = (Path(__file__).resolve().parents[1] / "StudentDB" / "quiz_results.csv").resolve()
print(QUIZ_RESULTS_CSV)

@mcp.tool()
def save_quiz_result(
    student_id: str,
    quiz_id: str,
    chapter_id: int,
    score: int,
    total: int,
    created_at: str,
) -> Dict[str, Any]:
    """
    Save final quiz result into quiz_results.csv.

    Returns:
      {"ok": true, "percent": <float>}
    """

    percent = (float(score) / float(total)) * 100.0 if total else 0.0

    with open(QUIZ_RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "student_id",
                "quiz_id",
                "chapter_id",
                "score",
                "total",
                "percent",
                "created_at",
            ],
        )
        writer.writerow(
            {
                "student_id": student_id,
                "quiz_id": quiz_id,
                "chapter_id": int(chapter_id),
                "score": int(score),
                "total": int(total),
                "percent": f"{percent:.2f}",
                "created_at": created_at,
            }
        )

    return {"ok": True, "percent": percent}


@mcp.tool()
def get_chapter_stats(student_id: str) -> Dict[str, Any]:
    """
    Read quiz_results.csv and compute chapter-wise stats for a student.

    Returns:
      {
        "student_id": "...",
        "chapters": [
          {"chapter_id": 1, "quizzes": 2, "avg_percent": 65.0},
          ...
        ]
      }
    """

    # chapter_id -> list of percent scores
    bucket: Dict[int, List[float]] = {}

    with open(QUIZ_RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("student_id") != student_id:
                continue

            chap = int(row["chapter_id"])
            pct = float(row["percent"])

            bucket.setdefault(chap, []).append(pct)

    chapters: List[Dict[str, Any]] = []
    for chap, percents in bucket.items():
        avg = sum(percents) / len(percents) if percents else 0.0
        chapters.append(
            {
                "chapter_id": chap,
                "quizzes": len(percents),
                "avg_percent": avg,
            }
        )

    # Sort weakest first (lowest average)
    chapters.sort(key=lambda x: x["avg_percent"])

    return {"student_id": student_id, "chapters": chapters}


if __name__ == "__main__":
    # Start the MCP server over stdio (default FastMCP transport)
    mcp.run(transport="stdio")
