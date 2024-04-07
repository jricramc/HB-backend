from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from getpass import getpass
# from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# configure client
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# model_name = "text-embedding-ada-002"
index_name = "langchain-multi-query-demo2"
index = pc.Index(index_name)

# embed = OpenAIEmbeddings(
#     model=model_name, openai_api_key=OPENAI_API_KEY, disallowed_special=()
# )

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text):
    try:
        # Use the Embeddings endpoint
        response = OpenAI.Embedding.create(
            input=text,
            model="text-embedding-ada-002"  # Ensure this is the correct model for embeddings
        )
        # Assuming response format includes an embeddings array
        embedding = response['data'][0]['embedding']
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
    res = index.query(vector=[embed], top_k=3, include_metadata=True)

    contexts = []
    titles = []
    for i in res['matches']:
        contexts.append(i['metadata']['text'])
        titles.append(i['metadata']['title'])

    prompt = "Answer the question based on the contexts below., and list the titles of the contexts that you used\n\n" + "contexts:" + str(contexts) + "\n\ntitles:" + str(titles) + "\n\nQuestion: " + query + "\nAnswer:"
    result = complete(prompt)
    return result.choices[0].message.content

# query = "what happens when men are infertile"

# print (get_response(query))
# def api_get_response():
#     data = request.get_json()
#     query = data.get('query')
#     response = get_response(query)
#     return jsonify({'response': response})

# if __name__ == '__main__':
    
#     app.run(debug=True)


app = Flask(__name__)

@app.route('/get_response', methods=['POST'])
def api_get_response():
    print('here')
    data = request.get_json()
    query = data.get('query')
    response = get_response(query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)