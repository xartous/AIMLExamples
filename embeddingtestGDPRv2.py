from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import textwrap
from typing import Any, List, Tuple, Dict
import numpy as np
import openai
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from numpy import ndarray

# Load OpenAI API key
with open("apikey_openai.txt") as file:
    openai.api_key = file.readline().strip()

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Create a collection
collection_name = "legislation"
dimension = 1536  # Dimension of your embeddings
client.recreate_collection(collection_name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))

# Define constants
URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679&from=EN"
ENCODING_MODEL = "text-davinci-003"

# Fetch legislation text
response = requests.get(URL)
soup = BeautifulSoup(response.content, "html.parser")
legislation = soup.text

# Split legislation into sections
sections = legislation.split("HAVE ADOPTED THIS REGULATION:")[1].split('\nArticle')[1:]
sections = ["Article" + section for section in sections]

# Fetch section titles
section_titles = [title.text for title in soup.find_all(class_='sti-art')]

# Initialize tiktoken encoding
enc = tiktoken.encoding_for_model(ENCODING_MODEL)

# Compute tokens per section
tokens_per_section = [len(enc.encode(section)) for section in sections]

# Prepare dataframe
headings = ["Article " + str(i + 1) for i in range(len(sections))]
df = pd.DataFrame(
    data={'title': section_titles, 'heading': headings, 'content': sections, 'tokens': tokens_per_section})

# Save dataframe to CSV
df.to_csv('legislation.csv', index=False)

# Load dataframe from CSV
df = pd.read_csv('legislation.csv')
df = df.set_index(["title", "heading"])


# Define helper functions
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    return {idx: get_embedding(r.content) for idx, r in df.iterrows()}


document_embeddings = compute_doc_embeddings(df)


def chunks(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield list(data.items())[i:i + batch_size]


# Define batch size
batch_size = 10  # Adjust this value based on your needs

# Split embeddings into batches
embedding_batches = list(chunks(document_embeddings, batch_size))

# Save each batch to Qdrant
for batch_idx, batch in enumerate(embedding_batches):
    points = []
    for idx, ((title, heading), embedding) in enumerate(batch):
        payload = {"title": title, "heading": heading}
        point_id = batch_idx * batch_size + idx
        point = PointStruct(id=point_id, vector=embedding, payload=payload)
        points.append(point)
    client.upsert(collection_name=collection_name, points=points)


# Define helper functions for GPT
def order_by_similarity(query: str, collection_name: str, client: QdrantClient, top_n: int = 10) -> list[
    tuple[float, dict]]:
    query_embedding = get_embedding(query)
    hits = client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_n)
    document_similarities = [(hit.score, hit.payload) for hit in hits]
    return document_similarities


def construct_prompt(query: str, collection_name: str, df: pd.DataFrame, top_n: int = 3) -> Tuple[List[Any], int]:
    document_similarities = order_by_similarity(query, collection_name, client)
    top_documents = document_similarities[:top_n]
    prompt = []
    total_tokens = 0
    for _, payload in top_documents:
        title = payload["title"]
        heading = payload["heading"]
        section = df.loc[(title, heading)]
        section_content = section["content"]
        section_tokens = section["tokens"]
        if total_tokens + section_tokens > 4096:
            break
        prompt.append(section_content)
        total_tokens += section_tokens
    return prompt, total_tokens


def answer_with_gpt(query: str, df: pd.DataFrame, collection_name: Dict[Tuple[str, str], np.ndarray],
                    show_prompt: bool = False) -> Tuple[str, int]:
    messages = [
        {"role": "system",
         "content": "You are a GDPR chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]

    prompt, section_length = construct_prompt(query, collection_name, df)

    if show_prompt:
        print(prompt)

    context = '\n'.join(prompt) + '\n\n --- \n\n + ' + query
    messages.append({"role": "user", "content": context})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=messages)

    return '\n' + response['choices'][0]['message']['content'], section_length


# Usage
prompt = "Do I have permission to review my information?"
response, sections_tokens = answer_with_gpt(prompt, df, collection_name)
print(textwrap.fill(response, initial_indent='', subsequent_indent='    '))
