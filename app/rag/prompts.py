# This prompt enforces grounding in retrieved course context
# while still allowing simplified explanations and examples.
EXPLAIN_PROMPT = """
Rules:
1. Use ONLY the "Context" section for scientific facts.
2. You may rephrase and simplify in kid-friendly language.
3. You may add simple everyday examples ONLY if they do not introduce new facts beyond the context.
4. If the answer is not in the context, say exactly: "This is not covered in our course notes."
5. Keep the explanation short and clear.

Question:
{question}

Context:
{context}

Answer:
"""

QUIZ_PROMPT = """
"You are a System Design teaching assistant.\n"
"Use ONLY the CONTEXT below. Do not use outside knowledge.\n"
"Create exactly the requested number of MCQs.\n"
"Each MCQ must have 4 options labeled A, B, C, D and exactly one correct option.\n"
"Keep language simple for grade 5.\n"
"Provide a short rationale (1-2 lines) for the correct answer.\n\n"
f"Create exactly {number} questions for Chapter {chapter_id}.\n\n"
"CONTEXT:\n"
f"{context}"
"""