"""End-to-End RAG Pipeline Orchestrator Module.

Coordinates vector retrieval, passage reranking, prompt assembly, Google Gemini LLM text generation,
confidence evaluation, and source citation formatting.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm.gemini import GeminiLLMClient
from llm.prompts import RAG_SYSTEM_INSTRUCTION, build_rag_prompt
from rag.citations import CitationTracker, SourceCitation
from rag.confidence import ConfidenceEvaluator
from rag.reranker import CrossEncoderReranker
from rag.retriever import VectorRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Dataclass holding final RAG query execution results.

    Attributes:
        query: Original user question string.
        answer: Generated grounded text answer.
        confidence_score: Float confidence score in range [0.0, 1.0].
        sources: List of SourceCitation objects.
        metadata: Pipeline execution metadata dict.
    """

    query: str
    answer: str
    confidence_score: float
    sources: List[SourceCitation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result object to dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence_score": self.confidence_score,
            "sources": [s.to_dict() for s in self.sources],
            "metadata": self.metadata,
        }


class RAGPipeline:
    """Core RAG pipeline orchestrator executing clean search and generation."""

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm_client: Optional[GeminiLLMClient] = None,
        confidence_evaluator: Optional[ConfidenceEvaluator] = None,
        citation_tracker: Optional[CitationTracker] = None,
    ) -> None:
        """Initializes RAG Pipeline modules using dependency injection.

        Args:
            retriever: VectorRetriever instance.
            reranker: CrossEncoderReranker instance.
            llm_client: GeminiLLMClient instance.
            confidence_evaluator: ConfidenceEvaluator instance.
            citation_tracker: CitationTracker instance.
        """
        self.retriever = retriever or VectorRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client or GeminiLLMClient()
        self.confidence_evaluator = confidence_evaluator or ConfidenceEvaluator()
        self.citation_tracker = citation_tracker or CitationTracker()

        logger.info("RAGPipeline fully initialized with all dependent sub-modules.")

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> RAGResult:
        """Executes full RAG workflow for a user query.

        Args:
            query: Question text string.
            top_k: Optional top K vector documents to retrieve.
            category_filter: Optional domain category filter.

        Returns:
            RAGResult instance containing grounded answer, confidence score, and sources.
        """
        logger.info("Starting RAG Pipeline execution for query: '%s'", query)

        # 1. Vector Retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
        )

        # 2. Rerank Chunks
        ranked_chunks = self.reranker.rerank(query=query, candidate_chunks=retrieved_chunks)

        # 3. Construct Prompt
        prompt = build_rag_prompt(user_query=query, retrieved_chunks=ranked_chunks)

        # 4. Generate LLM Answer
        llm_output = self.llm_client.generate_answer(
            prompt=prompt,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
        )
        answer_text = llm_output.get("answer", "")

        # 5. Compute Confidence Score
        confidence = self.confidence_evaluator.compute_confidence(
            retrieved_chunks=ranked_chunks,
            llm_answer=answer_text,
        )

        # 6. Extract Sources and Citations
        citations = self.citation_tracker.build_citations(retrieved_chunks=ranked_chunks)

        result = RAGResult(
            query=query,
            answer=answer_text,
            confidence_score=confidence,
            sources=citations,
            metadata={
                "model": llm_output.get("model", "gemini"),
                "retrieved_chunks_count": len(ranked_chunks),
            },
        )

        logger.info(
            "RAG Pipeline execution finished. Confidence: %.2f, Sources: %d",
            confidence,
            len(citations),
        )
        return result
