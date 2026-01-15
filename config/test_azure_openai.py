import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

print("Using endpoint:", endpoint)
print("Using deployment:", deployment)

client = AzureOpenAI(
    api_key=key,
    api_version="2024-02-01",
    azure_endpoint=endpoint
)

response = client.responses.create(
    model=deployment,
    input="Say hello from my Hybrid RAG project"
)

print("\n✅ Azure OpenAI Response:\n")
print(response.output_text)
