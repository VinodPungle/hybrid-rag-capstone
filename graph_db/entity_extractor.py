"""
Entity & Relationship Extraction Module (GraphRAG)

Uses Azure OpenAI to extract structured entities and relationships
from text chunks. These are then stored in Neo4j to form a knowledge graph.

All extraction parameters (temperature, max_tokens, prompt template,
entity types) are driven by config.yaml → entity_extraction section.
Token usage and latency are tracked for LLMOps monitoring.
"""

import os
import json
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

# [Step 1] Import config loader — extraction params from config.yaml
from config.settings import get as cfg

# [Step 2] Import structured logger and LLM call tracking
from utils.logger import get_logger, log_llm_call, LOG_PROMPTS

# [Step 9] Import Prometheus metrics recorder for token/latency tracking
from monitoring.prometheus_metrics import record_llm_call

load_dotenv()

logger = get_logger(__name__)

# [Step 1] Load entity extraction config (temperature, max_tokens, entity_types, prompt)
_ext = cfg("entity_extraction")

# [Step 1] api_version shared with LLM config; API keys from .env
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=cfg("llm", "api_version"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# [Step 1] Extraction prompt template from config.yaml (was hardcoded EXTRACTION_PROMPT)
EXTRACTION_PROMPT = _ext["extraction_prompt"]


def extract_entities_and_relationships(chunk):
    """
    Extract entities and relationships from a single text chunk using LLM.

    Sends the chunk to Azure OpenAI with an extraction prompt that specifies
    the entity types to look for. The LLM returns JSON with entities and
    relationships which are parsed and returned.

    Args:
        chunk: A text chunk string

    Returns:
        Tuple of (entities_list, relationships_list)
    """
    # [Step 1] Entity types list from config.yaml (was hardcoded in the prompt)
    entity_types = ", ".join(_ext["entity_types"])
    prompt_content = EXTRACTION_PROMPT.format(text=chunk, entity_types=entity_types)

    # [Step 2] Optionally log the extraction prompt for debugging
    if LOG_PROMPTS:
        logger.debug("Extraction prompt: %s", prompt_content[:500])

    # [Step 2] Measure LLM call latency
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are an entity and relationship extraction assistant. Always return valid JSON."},
            {"role": "user", "content": prompt_content}
        ],
        # [Step 1] temperature and max_tokens from config.yaml
        temperature=_ext["temperature"],
        max_tokens=_ext["max_tokens"]
    )
    latency = time.perf_counter() - start

    # [Step 2] Log token usage and latency via structured logger
    log_llm_call(logger, response, call_type="entity_extraction", latency_s=latency)

    # [Step 9] Record Prometheus metrics for monitoring dashboard
    record_llm_call("entity_extraction", response, latency)

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present (LLMs sometimes wrap JSON in ```json ... ```)
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
    """
    Extract entities and relationships from all chunks.

    Iterates over every chunk, calling the LLM for each one.
    Tags each entity/relationship with its source chunk index
    for traceability back to the original document.

    Args:
        chunks: List of text chunk strings

    Returns:
        Tuple of (all_entities, all_relationships)
    """
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
            # [Step 2] Log warning instead of print() — continues processing remaining chunks
            logger.warning("Failed to extract from chunk %d: %s", i, e)

    # [Step 2] Structured log of extraction summary
    logger.info("Extracted %d entities and %d relationships from %d chunks.",
                len(all_entities), len(all_relationships), len(chunks))
    return all_entities, all_relationships
