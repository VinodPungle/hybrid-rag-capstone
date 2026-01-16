import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_answer(query, context):
    prompt = f"""
You are a compliance assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        input=prompt
    )
    return response.output_text
