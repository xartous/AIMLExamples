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


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Fetch embedding for given text using specified model.
    """
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """
    Compute embeddings for all documents in the dataframe.
    """
    return {idx: get_embedding(r.content) for idx, r in df.iterrows()}


document_embeddings = compute_doc_embeddings(df)


def vector_similarity(x: List[float], y: List[float]) -> np.ndarray:
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(np.array(x), np.array(y))


def order_by_similarity(query: str, contexts: Dict[Tuple[str, str], np.ndarray]) -> list[
    tuple[ndarray, tuple[str, str]]]:
    """
    Order documents by their similarity to a given query.
    """
    query_embedding = get_embedding(query)
    document_similarities = sorted(
        [(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in
         contexts.items()], reverse=True)
    return document_similarities


def construct_prompt(query: str, document_embeddings: Dict[Tuple[str, str], np.ndarray], df: pd.DataFrame,
                     top_n: int = 3) -> \
        Tuple[List[Any], int]:
    """
    Construct a prompt from top_n documents similar to the query.
    """
    document_similarities = order_by_similarity(query, document_embeddings)
    top_documents = document_similarities[:top_n]
    prompt = []
    total_tokens = 0
    for _, (title, heading) in top_documents:
        section = df.loc[title, heading]
        section_content = section["content"]
        section_tokens = section["tokens"]
        if total_tokens + section_tokens > 4096:
            break
        prompt.append(section_content)
        total_tokens += section_tokens
    return prompt, total_tokens


def answer_with_gpt(query: str, df: pd.DataFrame, document_embeddings: Dict[Tuple[str, str], np.ndarray],
                    show_prompt: bool = False) -> Tuple[str, int]:
    """
    Generate an answer to a query using GPT, given the documents dataframe and their embeddings.
    """
    messages = [
        {"role": "system",
         "content": "You are a GDPR chatbot, only answer the question by using the provided context. If your are unable to answer the question using the provided context, say 'I don't know'"}
    ]

    prompt, section_length = construct_prompt(query, document_embeddings, df)

    if show_prompt:
        print(prompt)

    context = '\n'.join(prompt) + '\n\n --- \n\n + ' + query
    messages.append({"role": "user", "content": context})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=messages)

    return '\n' + response['choices'][0]['message']['content'], section_length


# Usage
prompt = "Do I have permission to review my information?"
response, sections_tokens = answer_with_gpt(prompt, df, document_embeddings)
print(textwrap.fill(response, initial_indent='', subsequent_indent='    '))
