# AI Teaching Assistant

An AI-powered Teaching Assistant designed for students that
supports:

-   **Explain Mode (RAG-based Q&A)**
-   **Practice Mode (Chapter-wise Quiz Engine)**
-   **Progress Tracking (via Student Database tools)**

This system uses:

-   **LangChain** for LLM orchestration\
-   **Chroma** for vector database\
-   **LangGraph** for quiz state management\
-   **Anthropic Claude** as the LLM\
-   **FastMCP** for tool-based student progress storage

------------------------------------------------------------------------

# Architecture Overview

## 1. Explain Mode (RAG)

**Flow:**

1.  Student asks a question.
2.  Vector database retrieves top-k relevant syllabus chunks.
3.  Context is formatted and grounded.
4.  Claude answers strictly from syllabus.
5.  Citations are returned.

------------------------------------------------------------------------

## 2. Practice Mode (Quiz Engine)

**Flow:**

1.  Student selects chapter.
2.  Chapter-specific chunks retrieved via metadata filter.
3.  Claude generates 10 MCQs using structured output.
4.  LangGraph controls quiz state:
    -   initialize\
    -   ask\
    -   evaluate\
    -   finish\
5.  Final score stored via Student DB tool.
6.  Progress analytics retrieved.

------------------------------------------------------------------------

## 3. Student Database Layer

The system abstracts persistence behind MCP tools:

-   `save_quiz_result`
-   `get_chapter_stats`

The Practice module interacts only with tools --- not with the storage
mechanism directly.

------------------------------------------------------------------------

# Project Structure

    AI_Teaching_Assistant/
    │
    ├── course_data/              # Chapter-wise lesson text files
    │   ├── lesson1_plants.txt
    │   ├── lesson2_...
    │
    ├── indexes/chroma/           # Persisted Chroma vector DB
    │
    ├── mcp_server/
    │   └── server.py             # FastMCP server exposing tools
    │
    ├── app/
    │   ├── rag/
    │   │   └── build_index.py
    │   │
    │   ├── qa.py
    │   │
    │   ├── practice_app.py
    │   │
    │   └── mcp_client/
    │       └── mcp_tools.py
    │
    └── README.md

------------------------------------------------------------------------

# Setup Instructions

## 1. Create Virtual Environment

``` bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
```

------------------------------------------------------------------------

## 2. Install Dependencies

``` bash
pip install langchain
pip install langchain-anthropic
pip install langchain-chroma
pip install langchain-huggingface
pip install langgraph
pip install chromadb
pip install sentence-transformers
pip install python-dotenv
pip install fastmcp
pip install pydantic
```

------------------------------------------------------------------------

## 3. Add Environment Variables

Create a `.env` file:

    ANTHROPIC_API_KEY=your_api_key_here

------------------------------------------------------------------------

# Step 1: Build Vector Index

Ensure lesson files exist inside:

    course_data/

Run:

``` bash
python app/rag/build_index.py
```

------------------------------------------------------------------------

# Step 2: Run Explain Mode

``` bash
python app/qa.py
```

------------------------------------------------------------------------

# Step 3: Run Practice Mode

``` bash
python app/practice_app.py
```

------------------------------------------------------------------------

# Key Design Decisions

## Structured Output for Quiz Generation

Quiz questions are generated using Pydantic schema via:

``` python
llm.with_structured_output()
```

This prevents invalid JSON parsing issues.

------------------------------------------------------------------------

## Router-Based LangGraph Flow

The graph entrypoint is a router node to prevent:

-   Infinite recursion
-   Re-running initialization
-   Regenerating quiz repeatedly

------------------------------------------------------------------------

## Project Goal

To build a syllabus-aware AI teaching assistant that:

-   Explains concepts\
-   Tests understanding\
-   Tracks progress\
-   Identifies weak areas

All grounded strictly in provided course material.
