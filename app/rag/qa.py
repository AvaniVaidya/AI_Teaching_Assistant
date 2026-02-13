from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Vector DB + embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.rag.prompts import EXPLAIN_PROMPT

load_dotenv()


def load_vectordb() -> Chroma:
    """
    Loads the persisted Chroma vector database used for RAG retrieval.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="indexes/chroma",
        collection_name="class5_science",
        embedding_function=embeddings,
    )


def format_context(docs: Any) -> str:
    """
    Formats retrieved Documents into a context block that includes sources.
    The LLM will be instructed to use ONLY this context for factual content.
    """
    if not isinstance(docs, list):
        raise TypeError(f"Expected retrieved docs to be a list, got: {type(docs)}")

    blocks: List[str] = []
    for d in docs:
        if not hasattr(d, "page_content"):
            raise TypeError(f"Expected Document-like object, got: {type(d)}")

        src = getattr(d, "metadata", {}).get("source", "unknown")
        snippet = (d.page_content or "").strip()
        if snippet:
            blocks.append(f"Source: {src}\n{snippet}")

    return "\n\n".join(blocks)


def answer_question(
    question: str,
    messages: List[Dict[str, str]],
    k: int = 4,
    llm: Optional[ChatAnthropic] = None,
    vectordb: Optional[Chroma] = None,
) -> Dict[str, Any]:
    """
    Answers a student question using ONLY the context retrieved from the vector DB.
    Returns the answer + unique citations (filenames).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (env or .env).")

    # Load vector DB once (or reuse if passed in)
    vectordb = vectordb or load_vectordb()

    # Retrieve top-k relevant chunks (semantic search)
    retrieved = vectordb.similarity_search(question, k=k)

    # Convert retrieved chunks into a prompt-friendly context string
    context = format_context(retrieved)

    # If retrieval is empty, we should refuse (syllabus constraint)
    if not context.strip():
        return {
            "question": question,
            "answer": "This is not covered in our course notes.",
            "citations": [],
        }

    # Create/reuse Claude model client
    llm = llm or ChatAnthropic(
        model="claude-opus-4-5-20251101",
        temperature=0,
        anthropic_api_key=api_key,
    )

    # Build per-turn augmented prompt (context + question)
    augmented_user_prompt = EXPLAIN_PROMPT.format(question=question, context=context)

    # ✅ DO NOT append the augmented prompt into history
    # Instead, send it as the last message only for this call
    temp_messages = messages + [{"role": "user", "content": augmented_user_prompt}]

    # Ask the model to answer grounded in the context
    response = llm.invoke(temp_messages).content

    # Create unique citations from retrieved chunk metadata
    citations: List[str] = []
    for d in retrieved:
        src = getattr(d, "metadata", {}).get("source", "unknown")
        if src not in citations:
            citations.append(src)

    return {"question": question, "answer": response, "citations": citations}


def run_chatbot() -> None:
    """
    Simple terminal chatbot:
    - Student types questions
    - Assistant answers using ONLY retrieved course context
    - Prints citations so the student can trace the source
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (env or .env).")

    # Load heavy objects once for speed
    vectordb = load_vectordb()
    llm = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        temperature=0,
        anthropic_api_key=api_key,
    )

    print("\nClass 5 Science TA (RAG Chatbot)")
    print("Type your question. Type 'exit' to quit.\n")

    messages = [
        {
            "role": "system",
            "content": "You are a Class 5 Science Teaching Assistant. Follow the Rules mentioned in the Prompt."
        }
    ]

    while True:
        user_input = input("You: ").strip()

        # Exit conditions
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("TA: Bye! Keep being curious 🧠📚")
            break

        messages.append({"role": "user", "content": user_input})

        # Answer using RAG
        out = answer_question(user_input, messages, k=4, llm=llm, vectordb=vectordb)
        messages.append({"role": "assistant", "content": out["answer"]})

        # Print answer
        print("\nTA:", out["answer"])

        # Print citations (if any)
        if out["citations"]:
            print("Sources:", ", ".join(out["citations"]))
        else:
            print("Sources: (none)")

        print()  # blank line for readability


if __name__ == "__main__":
    run_chatbot()
