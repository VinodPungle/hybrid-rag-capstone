import os
from openai import AzureOpenAI

# Azure OpenAI configuration
endpoint = "https://myopenai-demo-vinod.cognitiveservices.azure.com/"
api_key = "7mPYy9EHNMwukUL9sWLMlo5SvcMUpxo0pDqK5G3zMOOmhDCrhR9WJQQJ99BLAC77bzfXJ3w3AAABACOG1A89"
deployment = "embedding-model"
api_version = "2024-02-01"

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version
)

# Call the embeddings API
response = client.embeddings.create(
    input=["first phrase", "second phrase", "third phrase"],
    model=deployment  # Use 'model' parameter with your deployment name
)

# Process and print the response
for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )

print(f"\nUsage: {response.usage}")