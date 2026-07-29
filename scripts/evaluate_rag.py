"""Standalone Script: RAG Pipeline Benchmark & Evaluator.

Runs test benchmark queries through the RAG pipeline to measure confidence scores,
latency, and retrieval accuracy.
"""

import logging
import time

from app.config import setup_logging
from rag.pipeline import RAGPipeline

setup_logging("INFO")
logger = logging.getLogger(__name__)

SAMPLE_BENCHMARK_QUERIES = [
    "What is the difference between supervised and unsupervised learning?",
    "How does gradient boosting reduce variance and bias?",
    "Explain the kernel trick in Support Vector Machines.",
    "What are the key hyperparameters for tuning Random Forest in Scikit-Learn?",
]


def evaluate_pipeline():
    """Runs benchmark evaluation across sample queries."""
    logger.info("Initializing RAG Pipeline Evaluation...")
    pipeline = RAGPipeline()

    total_queries = len(SAMPLE_BENCHMARK_QUERIES)
    total_latency = 0.0

    print(f"\n=======================================================")
    print(f"       ML RAG AGENT PIPELINE BENCHMARK EVALUATOR       ")
    print(f"=======================================================\n")

    for idx, query in enumerate(SAMPLE_BENCHMARK_QUERIES, 1):
        start_time = time.time()
        result = pipeline.run(query=query)
        latency = time.time() - start_time
        total_latency += latency

        print(f"Query [{idx}/{total_queries}]: '{query}'")
        print(f"  -> Answer: {result.answer[:120]}...")
        print(f"  -> Confidence Score: {result.confidence_score * 100:.1f}%")
        print(f"  -> Sources Count: {len(result.sources)}")
        print(f"  -> Latency: {latency:.3f} seconds\n")

    avg_latency = total_latency / total_queries if total_queries > 0 else 0.0
    print(f"=======================================================")
    print(f"Benchmark Complete | Avg Latency: {avg_latency:.3f}s")
    print(f"=======================================================\n")


if __name__ == "__main__":
    evaluate_pipeline()
