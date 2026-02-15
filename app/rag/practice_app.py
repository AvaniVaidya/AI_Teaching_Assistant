"""
practice_app.py

Practice mode using:
- LangChain: ChatAnthropic + Chroma retrieval + structured output (Pydantic)
- LangGraph: quiz session as a state machine (router pattern so it doesn't re-init every turn)
- MCP client: save quiz result + fetch stats (CSV-backed)

Fixes included:
1) No more "Q1/10 forever" (init runs only once using `initialized` + router entrypoint)
2) No recursion loop (graph does NOT auto-run ask->evaluate; router chooses next node)
3) Faster runtime (init happens once; you can also reduce retrieval_k or use a faster model)
4) Structured output (no json.loads; LangChain parses/validates MCQs via Pydantic)
"""

from __future__ import annotations

import os
import datetime
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import Literal as TypingLiteral

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END

from app.rag.prompts import QUIZ_PROMPT

import contextlib
import io

# Your MCP client wrapper (must support: async with PracticeResultsClient(...) as client)
from app.mcp_client.mcp_tools import PracticeResultsClient


# ----------------------------
# Hardcoded identity (requested)
# ----------------------------
STUDENT_ID = "student_001"
QUIZ_ID = "quiz1"


# ----------------------------
# Config
# ----------------------------
@dataclass
class PracticeConfig:
    persist_dir: str = "indexes/chroma"
    collection_name: str = "class5_science"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Consider switching to a faster Claude model if Opus feels slow
    anthropic_model: str = "claude-opus-4-5-20251101"
    temperature: float = 0.2

    questions_per_quiz: int = 10

    # Lower this to speed up generation (less context).
    # 8–12 is usually enough for quiz generation.
    retrieval_k: int = 10


# ----------------------------
# Structured output schemas
# ----------------------------
class QuizQuestion(BaseModel):
    question: str = Field(..., description="The question text")
    options: Dict[str, str] = Field(..., description="Options A/B/C/D mapped to text")
    correct: TypingLiteral["A", "B", "C", "D"] = Field(..., description="Correct option letter")
    rationale: str = Field(..., description="Short explanation why the correct option is correct")


class QuizPayload(BaseModel):
    questions: List[QuizQuestion] = Field(..., description="List of quiz questions")


# ----------------------------
# Vector DB + LLM helpers
# ----------------------------
def load_vectordb(cfg: PracticeConfig) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)
    return Chroma(
        persist_directory=cfg.persist_dir,
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
    )


def build_llm(cfg: PracticeConfig) -> ChatAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (env or .env).")

    return ChatAnthropic(
        model=cfg.anthropic_model,
        temperature=cfg.temperature,
        anthropic_api_key=api_key,
    )


def similarity_search_chapter(vectordb: Chroma, chapter_id: int, k: int) -> List[Any]:
    """
    Chapter-filtered retrieval without triggering:
      TypeError: ... got multiple values for keyword argument 'where'

    We inspect the wrapper signature and pass ONLY ONE of (filter/where).
    """
    query = "Key concepts and important facts for this chapter"

    sig = inspect.signature(vectordb.similarity_search)
    params = sig.parameters

    if "filter" in params:
        return vectordb.similarity_search(query, k=k, filter={"chapter_id": chapter_id})
    if "where" in params:
        return vectordb.similarity_search(query, k=k, where={"chapter_id": chapter_id})

    # Fallback: no filtering supported
    return vectordb.similarity_search(query, k=k)


def format_context(docs: List[Any], max_chars_per_chunk: int = 1500) -> str:
    """
    Formats retrieved Documents into a context string.
    Also trims each chunk to keep prompts smaller (faster + more reliable).
    """
    blocks: List[str] = []
    for d in docs:
        src = getattr(d, "metadata", {}).get("source", "unknown")
        text = (getattr(d, "page_content", "") or "").strip()
        if not text:
            continue
        # Trim chunk text to control prompt size (speed)
        text = text[:max_chars_per_chunk]
        blocks.append(f"Source: {src}\n{text}")
    return "\n\n".join(blocks)


def generate_quiz_structured(
    llm: ChatAnthropic,
    chapter_id: int,
    context: str,
    n: int,
) -> List[Dict[str, Any]]:
    """
    Uses LangChain structured output instead of raw JSON parsing.
    Returns a list of dicts so the rest of your code stays the same.
    """
    structured_llm = llm.with_structured_output(QuizPayload)

    prompt = QUIZ_PROMPT.format(number=n, chapter_id=chapter_id, context=context)

    payload: QuizPayload = structured_llm.invoke(prompt)

    questions: List[Dict[str, Any]] = []
    for q in payload.questions:
        # Ensure keys A/B/C/D exist (sometimes models can omit, we normalize defensively)
        options = dict(q.options)
        for key in ("A", "B", "C", "D"):
            options.setdefault(key, "")

        questions.append(
            {
                "question": q.question,
                "options": options,
                "correct": q.correct,
                "rationale": q.rationale,
            }
        )

    # If the model returns fewer than n (rare), you can decide to accept or error.
    if len(questions) != n:
        # Best practice: fail loudly so you can see it and fix prompts/model settings.
        raise RuntimeError(f"Expected {n} questions, got {len(questions)}")

    return questions


# ----------------------------
# LangGraph state definition
# ----------------------------
class QuizState(TypedDict):
    # Inputs
    chapter_id: int

    # Router control
    initialized: bool
    done: bool

    # Generated quiz data
    context: str
    questions: List[Dict[str, Any]]

    # Progress counters
    current_index: int
    score: int
    total: int

    # Student's current answer (set by UI between invocations)
    user_answer: Optional[str]

    # UI outputs
    ui_question: Optional[Dict[str, Any]]
    ui_feedback: Optional[str]


def build_quiz_graph(cfg: PracticeConfig, llm: ChatAnthropic, vectordb: Chroma):
    """
    Builds a graph with a router entrypoint.

    The router ensures:
    - init runs only once
    - after that, we alternate between ask/evaluate based on whether user_answer is set
    """
    g = StateGraph(QuizState)

    # ----------------------------
    # Nodes
    # ----------------------------
    def node_router(state: QuizState) -> QuizState:
        # Router node does not modify state; it just routes via conditional edges.
        return state

    def node_init(state: QuizState) -> QuizState:
        # Retrieve chapter-specific context
        docs = similarity_search_chapter(vectordb, chapter_id=state["chapter_id"], k=cfg.retrieval_k)
        context = format_context(docs)

        if not context.strip():
            raise RuntimeError(
                f"No context found for chapter_id={state['chapter_id']}. "
                "Ensure you re-indexed with metadata chapter_id from filenames like lesson1_*.txt"
            )

        # Generate quiz once (slow step)
        questions = generate_quiz_structured(
            llm=llm,
            chapter_id=state["chapter_id"],
            context=context,
            n=cfg.questions_per_quiz,
        )

        # Store quiz in state
        state["context"] = context
        state["questions"] = questions
        state["current_index"] = 0
        state["score"] = 0
        state["total"] = len(questions)

        # Clear UI fields
        state["ui_question"] = None
        state["ui_feedback"] = None
        state["user_answer"] = None

        # Mark initialized so init won't run again
        state["initialized"] = True
        return state

    def node_ask(state: QuizState) -> QuizState:
        i = state["current_index"]

        # If we're out of questions, mark done
        if i >= state["total"]:
            state["done"] = True
            state["ui_question"] = None
            return state

        q = state["questions"][i]
        state["ui_question"] = {
            "index": i + 1,
            "total": state["total"],
            "question": q["question"],
            "options": q["options"],
        }

        return state

    def node_evaluate(state: QuizState) -> QuizState:
        i = state["current_index"]
        q = state["questions"][i]

        chosen = (state["user_answer"] or "").strip().upper()

        # If invalid input, don't advance, keep asking for valid choice
        if chosen not in {"A", "B", "C", "D"}:
            state["ui_feedback"] = "Please answer with A, B, C, or D."
            state["user_answer"] = None
            return state

        correct = q["correct"].strip().upper()
        is_correct = (chosen == correct)

        if is_correct:
            state["score"] += 1
            state["ui_feedback"] = "Correct!\n"
        else:
            correct_text = q["options"].get(correct, "")
            rationale = (q.get("rationale") or "").strip()
            state["ui_feedback"] = (
                "Not quite. "
                f"Correct answer is [ {correct} : {correct_text} ]"
                f" as {rationale}\n"
            )

        # Advance
        state["current_index"] += 1

        # Clear answer so router will go to ask next
        state["user_answer"] = None
        return state

    def node_finish(state: QuizState) -> QuizState:
        state["done"] = True
        state["ui_question"] = None
        state["ui_feedback"] = f"\nQuiz finished. Your score: {state['score']}/{state['total']}"
        return state

    # ----------------------------
    # Routing logic
    # ----------------------------
    def route_from_router(state: QuizState) -> Literal["init", "evaluate", "ask", "finish"]:
        if state.get("done"):
            return "finish"

        if not state.get("initialized"):
            return "init"

        # If a user answer is present, evaluate it
        if state.get("user_answer"):
            return "evaluate"

        # Otherwise ask the next question
        return "ask"

    def route_after_evaluate(state: QuizState) -> Literal["ask", "finish"]:
        if state["current_index"] >= state["total"]:
            return "finish"
        return "ask"

    # ----------------------------
    # Graph wiring
    # ----------------------------
    g.add_node("router", node_router)
    g.add_node("init", node_init)
    g.add_node("ask", node_ask)
    g.add_node("evaluate", node_evaluate)
    g.add_node("finish", node_finish)

    # Entry is router (not init), so we don't re-init every invoke
    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        route_from_router,
        {"init": "init", "evaluate": "evaluate", "ask": "ask", "finish": "finish"},
    )

    # After init, show the first question
    g.add_edge("init", "ask")

    # After evaluate, either ask next or finish
    g.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"ask": "ask", "finish": "finish"},
    )

    g.add_edge("finish", END)

    return g.compile()


# ----------------------------
# CLI runner that uses MCP tools for saving result + stats
# ----------------------------
async def run_practice(chapter_id: int) -> None:
    load_dotenv()

    cfg = PracticeConfig()
    vectordb = load_vectordb(cfg)
    llm = build_llm(cfg)
    graph = build_quiz_graph(cfg, llm, vectordb)

    # Initial state: init has NOT happened yet
    state: QuizState = {
        "chapter_id": chapter_id,
        "initialized": False,
        "done": False,

        "context": "",
        "questions": [],

        "current_index": 0,
        "score": 0,
        "total": 0,

        "user_answer": None,
        "ui_question": None,
        "ui_feedback": None,
    }

    print("Practice Mode\n")
    print(f"Student: {STUDENT_ID} | Quiz: {QUIZ_ID} | Chapter: {chapter_id}\n")

    # First invoke: router -> init -> ask
    state = graph.invoke(state)

    while not state["done"]:
        q = state["ui_question"]
        if not q:
            # If no question payload, let router decide next step
            state = graph.invoke(state)
            continue

        print(f"Q{q['index']}/{q['total']}: {q['question']}\n")
        for opt_key, opt_val in q["options"].items():
            print(f"{opt_key}) {opt_val}\n")

        #ans = input("Your answer (A/B/C/D): ").strip().upper()
        ans = input().strip().upper()

        # Put answer into state, then invoke router -> evaluate -> ask/finish
        state["user_answer"] = ans
        state = graph.invoke(state)

        if state["ui_feedback"]:
            #print("\nTeacher:", state["ui_feedback"])
            print(state["ui_feedback"], "\n")
            print()
            state["ui_feedback"] = None

    # Ensure finish message is shown
    if state["ui_feedback"]:
        #print("Teacher:", state["ui_feedback"])
        print(state["ui_feedback"])
        print()

    # Persist result and fetch stats via MCP tools
    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    # NOTE: Ensure your PracticeResultsClient points to the correct MCP server path
    # and that it returns dict outputs. If it returns a raw MCP result object, you’ll
    # need to extract .data or parse .content depending on your MCP SDK.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        async with PracticeResultsClient("mcp_server/server.py") as client:
            await client.save_quiz_result(
                {
                    "student_id": STUDENT_ID,
                    "quiz_id": QUIZ_ID,
                    "chapter_id": chapter_id,
                    "score": int(state["score"]),
                    "total": int(state["total"]),
                    "created_at": created_at,
                }
            )



if __name__ == "__main__":
    import asyncio
    asyncio.run(run_practice(chapter_id=1))
