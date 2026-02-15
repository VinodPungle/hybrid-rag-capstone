import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # Changed from AZURE_OPENAI_KEY
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_answer(query, context):
    """
    Generate an answer using Azure OpenAI based on the provided context.
    
    Args:
        query: The user's question
        context: Relevant context from retrieved documents
        
    Returns:
        Generated answer as a string
    """
    prompt = f"""You are a compliance assistant.
Answer ONLY using the context below. If the answer cannot be found in the context, say "I cannot answer this based on the provided context."

Context:
{context}

Question:
{query}

Answer:"""

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a helpful compliance assistant. Answer questions based only on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    #print("Prompt:"+ prompt +"\n")

    
    return response.choices[0].message.content