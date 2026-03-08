import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

EXTRACTION_PROMPT = """Extract entities and relationships from the following text.
Return a JSON object with two keys:
- "entities": a list of objects with "name" (string) and "type" (string, e.g. "Person", "Organization", "Concept", "Regulation", "Committee", "Role", "Process", "Document")
- "relationships": a list of objects with "source" (string), "target" (string), and "relation" (string)

Only extract clearly stated facts. Keep entity names concise and normalized (e.g. "Audit Committee" not "the audit committee").

Text:
{text}

Return ONLY valid JSON, no markdown formatting."""


def extract_entities_and_relationships(chunk):
    """Extract entities and relationships from a single text chunk using LLM."""
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are an entity and relationship extraction assistant. Always return valid JSON."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(text=chunk)}
        ],
        temperature=0.0,
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    result = json.loads(raw)
    entities = result.get("entities", [])
    relationships = result.get("relationships", [])
    return entities, relationships


def extract_graph_from_chunks(chunks):
    """Extract entities and relationships from all chunks. Returns (all_entities, all_relationships)."""
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        try:
            entities, relationships = extract_entities_and_relationships(chunk)
            # Tag each entity/relationship with its source chunk index
            for e in entities:
                e["source_chunk"] = i
            for r in relationships:
                r["source_chunk"] = i
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        except Exception as e:
            print(f"Warning: Failed to extract from chunk {i}: {e}")

    print(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships from {len(chunks)} chunks.")
    return all_entities, all_relationships
