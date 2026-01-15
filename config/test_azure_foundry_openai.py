import os
from openai import AzureOpenAI

endpoint = "https://vinod-mkf3fo19-eastus2.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "3mrH37EXbcf3yvbt05hOS3MHsqKTKoYZgWb9bCET39dtKch2lssYJQQJ99CAACHYHv6XJ3w3AAAAACOGF3Ux"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)

