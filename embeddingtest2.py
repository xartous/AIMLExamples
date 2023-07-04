import openai
import pandas as pd
import tiktoken
import numpy as np


# Load OpenAI API key
with open("apikey_openai.txt") as file:
    openai.api_key = file.readline().strip()

# Define short texts
texts = ["This is the first short text.", "This is the second short text about devils and hell.", "This is the third short text."]

# Get the encoding for the model
enc = tiktoken.encoding_for_model("text-davinci-003")

# Count tokens in each text
tokens_per_text = [len(enc.encode(text)) for text in texts]

# Create a DataFrame
df = pd.DataFrame()
df['title'] = ["Title 1", "Title 2", "Title 3"]
df['content'] = texts
df['tokens'] = tokens_per_text
df = df.set_index("title")

# Define functions for embeddings and similarity
def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[str, list[float]]:
    return {idx: get_embedding(r.content) for idx, r in df.iterrows()}

def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[str, np.array]) -> list[(float, str)]:
    query_embedding = get_embedding(query)
    document_similarities = sorted([(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()], reverse=True)
    return document_similarities

def construct_prompt(query: str, document_embeddings: dict[str, np.array], df: pd.DataFrame, top_n: int = 3) -> list[str]:
    document_similarities = order_by_similarity(query, document_embeddings)
    top_documents = document_similarities[:top_n]
    prompt = []
    total_tokens = 0
    for _, title in top_documents:
        section = df.loc[title]
        section_content = section["content"]
        section_tokens = section["tokens"]
        if total_tokens + section_tokens > 4096:
            break
        prompt.append(section_content)
        total_tokens += section_tokens
    return prompt, total_tokens

def answer_with_gpt_4(query: str, df: pd.DataFrame, document_embeddings: dict[str, np.array], show_prompt: bool = False) -> str:
    messages = [
        {"role": "system", "content": "You are a chatbot. Answer the question using the provided context. If you are unable to answer the question using the provided context, say 'I don't know'"}
    ]

    prompt, _ = construct_prompt(query, document_embeddings, df)

    if show_prompt:
        print(prompt)

    context = "\n".join(prompt) + '\n\n --- \n\n + ' + query

    messages.append({"role": "user", "content": context})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=messages)

    return '\n' + response['choices'][0]['message']['content']

# Compute embeddings
document_embeddings = compute_doc_embeddings(df)

# Use the chat model
prompt = "Tell me about the second text."
response = answer_with_gpt_4(prompt, df, document_embeddings)
print(response)
