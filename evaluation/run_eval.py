"""
LLMOps Evaluation Runner (Step 8)

Runs the full RAG pipeline end-to-end against a set of test question-answer pairs
and computes evaluation metrics (BLEU, ROUGE, faithfulness) for each.

Flow:
  1. Load test cases from evaluation/test_cases.json
  2. Build the full pipeline (PDF → chunks → embeddings → FAISS + Neo4j)
  3. For each test case: run hybrid search → generate answer → compute metrics
  4. Save detailed results to evaluation/eval_results.json
  5. Print summary with average scores

Usage:
    python -m evaluation.run_eval

Requires INPUT_FILE environment variable to point to the source PDF.
Which metrics are evaluated is controlled by config.yaml → llmops.evaluation.metrics.
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# [Step 1] Import config loader for evaluation settings
from config.settings import get as cfg

# [Step 2] Import structured logger (replaces print() for status/error messages)
from utils.logger import get_logger

# Pipeline module imports — used to build the full pipeline before evaluation
from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.hybrid_search import hybrid_search
from graph_db.entity_extractor import extract_graph_from_chunks
from graph_db.neo4j_store import get_driver, build_knowledge_graph
from llm.generator import generate_answer

# [Step 8] Import the evaluate function that runs all enabled metrics
from evaluation.metrics import evaluate

logger = get_logger(__name__)

# Path to the JSON file containing test question-answer pairs
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")


def run():
    """
    Run the full evaluation pipeline.

    Steps:
      1. Load test cases from test_cases.json
      2. Build the RAG pipeline (PDF → chunks → embeddings → FAISS + Neo4j)
      3. For each test case: hybrid search → generate answer → compute metrics
      4. Save results to eval_results.json and print summary
    """
    # --- 1. Load test cases from JSON file ---
    if not os.path.exists(TEST_CASES_PATH):
        logger.error("Test cases not found at %s", TEST_CASES_PATH)
        logger.info("Create evaluation/test_cases.json with format:")
        logger.info('[{"query": "...", "reference_answer": "..."}]')
        sys.exit(1)

    with open(TEST_CASES_PATH, "r") as f:
        test_cases = json.load(f)

    logger.info("Loaded %d test cases.", len(test_cases))

    # --- 2. Build the full RAG pipeline from INPUT_FILE env var ---
    input_file = os.getenv("INPUT_FILE")
    if not input_file or not os.path.exists(input_file):
        logger.error("INPUT_FILE not set or not found.")
        sys.exit(1)

    # Pipeline: load PDF → chunk → embed → build FAISS index
    text = load_pdf_text(input_file)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    index = build_faiss_index(vectors)

    # Build knowledge graph (non-fatal: evaluation still works with vector-only search)
    driver = None
    try:
        entities, relationships = extract_graph_from_chunks(chunks)
        driver = get_driver()
        build_knowledge_graph(driver, entities, relationships)
    except Exception as e:
        logger.warning("Graph build failed: %s", e)

    # --- 3. Evaluate each test case against the pipeline ---
    all_results = []
    for i, tc in enumerate(test_cases):
        query = tc["query"]
        reference = tc["reference_answer"]

        # Run hybrid search and generate an answer
        results = hybrid_search(query, index, chunks, driver=driver)
        answer = generate_answer(query, results["combined_context"])

        # [Step 8] Compute all enabled metrics (BLEU, ROUGE, faithfulness)
        scores = evaluate(reference, answer, context=results["combined_context"])

        case_result = {
            "query": query,
            "reference": reference,
            "generated": answer,
            "scores": scores,
        }
        all_results.append(case_result)

        # Log per-case metrics for debugging
        logger.info("Case %d/%d: '%s'", i + 1, len(test_cases), query[:60])
        for metric, score in scores.items():
            logger.info("  %s: %s", metric, json.dumps(score))

    # --- 4. Save detailed results to JSON ---
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # --- 5. Print summary with average scores across all test cases ---
    print("\n============= EVALUATION SUMMARY =============\n")
    if all_results and "bleu" in all_results[0].get("scores", {}):
        avg_bleu4 = sum(r["scores"]["bleu"]["bleu_4"] for r in all_results) / len(all_results)
        print(f"  Avg BLEU-4:        {avg_bleu4:.4f}")
    if all_results and "rouge" in all_results[0].get("scores", {}):
        avg_rougeL = sum(r["scores"]["rouge"]["rougeL"]["fmeasure"] for r in all_results) / len(all_results)
        print(f"  Avg ROUGE-L F1:    {avg_rougeL:.4f}")
    if all_results and "faithfulness" in all_results[0].get("scores", {}):
        avg_faith = sum(r["scores"]["faithfulness"]["score"] for r in all_results) / len(all_results)
        print(f"  Avg Faithfulness:  {avg_faith:.4f}")
    print(f"\n  Total cases: {len(all_results)}")
    print("\n==============================================\n")

    # Cleanup: close Neo4j connection
    if driver:
        driver.close()


if __name__ == "__main__":
    run()
