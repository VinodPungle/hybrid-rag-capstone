from ingestion.pdf_loader import load_pdf_text
from ingestion.chunker import chunk_text
from embeddings.embedder import embed_texts
from vector_db.faiss_store import build_faiss_index
from retrieval.search import search
from llm.generator import generate_answer

import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load document
text = load_pdf_text(os.getenv("INPUT_FILE"))

# 2. Chunk text
chunks = chunk_text(text)

# 3. Create embeddings
vectors = embed_texts(chunks)

# 4. Build vector index
index = build_faiss_index(vectors)

# 5. Ask a question
query = "Tell more about audit committee responsibilities."
results = search(query, index, chunks)

# 6. Generate answer
context = "\n".join(results)
answer = generate_answer(query, context)

print("\n================ ANSWER ================\n")
print(answer)
print("\n=======================================\n")
