INITIAL_PROMPT = """You are a helpful assistant that answers questions about company policies.

Context:
{context}

Question: {question}

Answer the question based on the context provided above."""


IMPROVED_PROMPT = """You are a RETRIEVAL-GROUNDED Policy Question Answering Assistant.

Your job is to answer strictly using the provided CONTEXT.
You are NOT allowed to use outside knowledge.

Follow these steps internally:
1. Read the context carefully.
2. Identify exact sentences that answer the question.
3. If no supporting sentences exist, reply:
   "I don't know based on the provided documents."

STRICT RULES:
- Do NOT guess.
- Do NOT add new information.
- Every claim MUST be supported by a quote from CONTEXT.
- Evidence MUST be SHORT DIRECT QUOTES copied exactly from the context.
- If evidence is missing → answer must be "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

Return ONLY valid JSON:

{{
  "answer": "Grounded answer or 'I don't know based on the provided documents.'",
  "evidence": ["exact short quote 1", "exact short quote 2"],
  "confidence": "High|Medium|Low"
}}

Confidence Guidelines:
- High → Answer explicitly stated in one place
- Medium → Requires combining multiple context sections
- Low → Weak or partial support

JSON Response:"""



def get_prompt(prompt_type: str, context: str, question: str) -> str:
    """
    Get formatted prompt.
    
    Args:
        prompt_type: "initial" or "improved"
        context: Retrieved document context
        question: User question
    
    Returns:
        Formatted prompt string
    """
    if prompt_type == "initial":
        template = INITIAL_PROMPT
    else:
        template = IMPROVED_PROMPT
    
    return template.format(context=context, question=question)