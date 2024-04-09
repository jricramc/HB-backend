from fastapi import FastAPI
import os
from getpass import getpass
# from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = "langchain-multi-query-demo2"
index = pc.Index(index_name)

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text):
    try:
        response = client.embeddings.create(
            input= text,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def complete(prompt):
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {
          "role": "user",
          "content": prompt
        }
      ],
      temperature=1,
      max_tokens=5000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    return response



def get_response(query):
    embed = generate_embedding(query)
    # print('embed', embed)
    res = index.query(vector=[embed], top_k=3, include_metadata=True)

    contexts = []
    titles = []
    for i in res['matches']:
        contexts.append(i['metadata']['text'])
        titles.append(i['metadata']['title'])

    prompt = "Answer the question based on the contexts below., and list the titles of the contexts that you used\n\n" + "contexts:" + str(contexts) + "\n\ntitles:" + str(titles) + "\n\nQuestion: " + query + "\nAnswer:"
    result = complete(prompt)
    return result.choices[0].message.content



class PostRequest ( BaseModel):
    Query : str


@app.post ('/')

async def scoring_endpoint(item: PostRequest):
    result = get_response(item.Query)
    return result