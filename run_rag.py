"""
CLI Runner for the Hybrid RAG Pipeline

Runs the full pipeline from the command line:
  1. Load PDF from INPUT_FILE env var
  2. Chunk text using config-driven parameters
  3. Generate embeddings via Azure OpenAI
  4. Build FAISS vector index
  5. Build knowledge graph in Neo4j (GraphRAG)
  6. Run hybrid search (vector + graph)
  7. Generate answer using combined context

Uses structured logging (Step 2) instead of print() for all status/error messages.

Usage: python run_rag.py
"""

from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.hybrid_search import hybrid_search
from graph_db.entity_extractor import extract_graph_from_chunks
from graph_db.neo4j_store import get_driver, build_knowledge_graph
from llm.generator import generate_answer

# [Step 2] Import structured logger (replaces all print() statements)
from utils.logger import get_logger

import os
import sys
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


def main():
    # 1. Load document from INPUT_FILE environment variable
    input_file = os.getenv("INPUT_FILE")
    if not input_file:
        logger.error("INPUT_FILE environment variable is not set.")
        sys.exit(1)
    if not os.path.exists(input_file):
        logger.error("File not found: %s", input_file)
        sys.exit(1)

    try:
        text = load_pdf_text(input_file)
    except Exception as e:
        logger.error("Error loading PDF: %s", e)
        sys.exit(1)

    if not text or not text.strip():
        logger.error("No text extracted from PDF.")
        sys.exit(1)

    # 2. Chunk text (chunk_size and overlap from config.yaml)
    try:
        chunks = chunk_text(text)
    except Exception as e:
        logger.error("Error chunking text: %s", e)
        sys.exit(1)

    if not chunks:
        logger.error("No chunks produced from text.")
        sys.exit(1)

    logger.info("Document loaded: %d chunks created.", len(chunks))

    # 3. Create embeddings via Azure OpenAI
    try:
        vectors = embed_texts(chunks)
    except Exception as e:
        logger.error("Error generating embeddings: %s", e)
        sys.exit(1)

    # 4. Build FAISS vector index
    try:
        index = build_faiss_index(vectors)
    except Exception as e:
        logger.error("Error building FAISS index: %s", e)
        sys.exit(1)

    logger.info("Vector index built successfully.")

    # 5. Build knowledge graph in Neo4j (GraphRAG) — non-fatal if it fails
    driver = None
    try:
        logger.info("Extracting entities and relationships from chunks...")
        entities, relationships = extract_graph_from_chunks(chunks)

        logger.info("Building knowledge graph in Neo4j...")
        driver = get_driver()
        build_knowledge_graph(driver, entities, relationships)
    except Exception as e:
        logger.warning("Graph construction failed, will use vector-only search: %s", e)

    # 6. Ask a question using hybrid search (vector + graph)
    query = "Tell more about audit committee responsibilities."
    try:
        results = hybrid_search(query, index, chunks, driver=driver)
    except Exception as e:
        logger.error("Error during hybrid search: %s", e)
        sys.exit(1)

    if not results["vector_results"] and not results["graph_results"]:
        logger.info("No relevant results found for the query.")
        if driver:
            driver.close()
        sys.exit(0)

    # 7. Generate answer using combined context from both search methods
    try:
        answer = generate_answer(query, results["combined_context"])
    except Exception as e:
        logger.error("Error generating answer: %s", e)
        if driver:
            driver.close()
        sys.exit(1)

    logger.info("Answer generated successfully.")
    print("\n================ ANSWER ================\n")
    print(answer)
    print("\n=======================================\n")

    if results["graph_results"]:
        logger.info("Knowledge graph context used: %d relationships", len(results["graph_results"]))
        for ctx in results["graph_results"]:
            print(f"  {ctx}")
        print()

    if driver:
        driver.close()


if __name__ == "__main__":
    main()
