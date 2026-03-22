"""
Embedding Module

Generates vector embeddings for text chunks using Azure OpenAI's embedding API.
The API version is read from config.yaml → embedding.api_version.
API keys and endpoints remain in .env (secrets).
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# [Step 1] Import config loader to read api_version from config.yaml
from config.settings import get as cfg

load_dotenv()

# [Step 1] api_version now comes from config.yaml instead of being hardcoded
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_KEY"),
    api_version=cfg("embedding", "api_version"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
)


def embed_texts(texts):
    """
    Generate embeddings for a list of text strings.

    Calls Azure OpenAI embedding API once per text chunk.
    Returns a list of embedding vectors (each is a list of floats).

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (list of float lists)
    """
    vectors = []
    for text in texts:
        resp = client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            input=text
        )
        vectors.append(resp.data[0].embedding)
    return vectors
