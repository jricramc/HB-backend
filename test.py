from fastapi import FastAPI
import os
from getpass import getpass
# from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text):
    try:
        # Use the Embeddings endpoint

        response = client.embeddings.create(
            input= text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

text = "hello"

res = generate_embedding(text)

print(res)