"""Prompt Management & Grounding Templates Module.

Defines strict system instructions, context formatting templates,
and grounding directives for RAG generation.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# System Instruction to enforce strict grounding and prevent hallucinations
RAG_SYSTEM_INSTRUCTION = """You are an expert AI & Machine Learning Assistant specializing in technical document analysis.
Your task is to answer user questions strictly based on the provided retrieved context documents (PDF papers, Scikit-Learn documentation, KTU lecture notes, and ML textbooks).

CRITICAL GROUNDING RULES:
1. Base your answer ONLY on the provided context passages. Do NOT extrapolate or introduce external facts.
2. If the answer cannot be determined from the context, state clearly: "I cannot find sufficient evidence in the retrieved documents to answer your question."
3. Include clear inline references to source documents and page numbers where applicable.
4. Maintain a professional, technical, and precise tone.
"""

RAG_PROMPT_TEMPLATE = """Retrieved Context Passages:
--------------------------------------------------------------------------------
{context_passages}
--------------------------------------------------------------------------------

User Question:
{user_query}

Instructions:
Synthesize a comprehensive, accurate answer grounded strictly in the context above.
Include citations to source filenames and page numbers in brackets [Source: filename, Page: N].
"""


def format_context_passages(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved document chunks into a clean context string block.

    Args:
        retrieved_chunks: List of dictionaries containing 'text' and 'metadata'.

    Returns:
        Formatted context text string for prompt injection.
    """
    logger.debug("Formatting %d context chunks into prompt context block.", len(retrieved_chunks))
    formatted_passages = []

    for idx, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("filename", "unknown_document")
        page = meta.get("page_number", "N/A")
        text = chunk.get("text", "").strip()

        passage = f"[Passage {idx}] Source: {source} (Page {page})\nContent: {text}\n"
        formatted_passages.append(passage)

    return "\n".join(formatted_passages)


def build_rag_prompt(user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Constructs the complete RAG prompt by interpolating context and query.

    Args:
        user_query: User question string.
        retrieved_chunks: List of context chunks retrieved from VectorStore.

    Returns:
        Complete formatted prompt string ready for LLM consumption.
    """
    context_str = format_context_passages(retrieved_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(
        context_passages=context_str if context_str else "No context passages retrieved.",
        user_query=user_query,
    )
    logger.debug("Successfully constructed RAG prompt.")
    return prompt
