from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.hybrid_search import hybrid_search
from graph_db.entity_extractor import extract_graph_from_chunks
from graph_db.neo4j_store import get_driver, build_knowledge_graph
from llm.generator import generate_answer

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Load document
    input_file = os.getenv("INPUT_FILE")
    if not input_file:
        print("Error: INPUT_FILE environment variable is not set.")
        sys.exit(1)
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    try:
        text = load_pdf_text(input_file)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        sys.exit(1)

    if not text or not text.strip():
        print("Error: No text extracted from PDF.")
        sys.exit(1)

    # 2. Chunk text
    try:
        chunks = chunk_text(text)
    except Exception as e:
        print(f"Error chunking text: {e}")
        sys.exit(1)

    if not chunks:
        print("Error: No chunks produced from text.")
        sys.exit(1)

    # 3. Create embeddings
    try:
        vectors = embed_texts(chunks)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)

    # 4. Build vector index
    try:
        index = build_faiss_index(vectors)
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        sys.exit(1)

    # 5. Build knowledge graph (GraphRAG)
    driver = None
    try:
        print("Extracting entities and relationships from chunks...")
        entities, relationships = extract_graph_from_chunks(chunks)

        print("Building knowledge graph in Neo4j...")
        driver = get_driver()
        build_knowledge_graph(driver, entities, relationships)
    except Exception as e:
        print(f"Warning: Graph construction failed, will use vector-only search: {e}")

    # 6. Ask a question (Hybrid: vector + graph search)
    query = "Tell more about audit committee responsibilities."
    try:
        results = hybrid_search(query, index, chunks, driver=driver)
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        sys.exit(1)

    if not results["vector_results"] and not results["graph_results"]:
        print("No relevant results found for the query.")
        if driver:
            driver.close()
        sys.exit(0)

    # 7. Generate answer using combined context
    try:
        answer = generate_answer(query, results["combined_context"])
    except Exception as e:
        print(f"Error generating answer: {e}")
        if driver:
            driver.close()
        sys.exit(1)

    print("\n================ ANSWER ================\n")
    print(answer)
    print("\n=======================================\n")

    if results["graph_results"]:
        print("--- Knowledge Graph Context Used ---")
        for ctx in results["graph_results"]:
            print(f"  {ctx}")
        print()

    if driver:
        driver.close()


if __name__ == "__main__":
    main()
