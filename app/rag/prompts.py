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
