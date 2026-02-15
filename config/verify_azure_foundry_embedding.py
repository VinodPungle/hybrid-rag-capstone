import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
api_version = os.getenv("AZURE_EMBEDDING_API_VERSION")

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