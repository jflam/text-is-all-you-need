import faiss
import numpy as np
import openai
import os
import sqlite_utils
import struct
import streamlit as st
import time
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

"""## DeepMemory Search"""

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
EMBEDDING_TABLE = "embeddings"
EMBEDDING_DB = "embeddings.db"
TOP_K = 10

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_key = os.environ[f"{ENVIRONMENT}_API_KEY"]
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = 'azure'
openai.api_version = '2022-12-01' # this may change in the future
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

def evaluate_prompt(prompt: str) -> str:
    response = openai.Completion.create(
        engine=DEPLOYMENT_ID,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1000,
    )
    return response.choices[0].text

def decode(blob):
    return struct.unpack("f" * (len(blob) // 4), blob)

template = PromptTemplate(
    input_variables=["user_question", "context"],
    template="""
The following is a question from the user: {user_question} 

Please answer the question using the context below:
{context}
"""
)

db = sqlite_utils.Database(EMBEDDING_DB)
table = db[EMBEDDING_TABLE]
if not table.exists():
    raise Exception(f"Table {EMBEDDING_TABLE} does not exist in {EMBEDDING_DB}")

if "model" not in st.session_state:
    model = SentenceTransformer(EMBEDDING_MODEL)
    st.session_state.model = model

    rows = db.execute(f"SELECT * FROM {EMBEDDING_TABLE}").fetchall()
    embeddings = [decode(row[0]) for row in rows]
    texts = [row[1] for row in rows]
    st.session_state.texts = texts
    embedding_size = len(embeddings[0])

    # Now we have the embeddings and the texts, we can create the index
    start = time.perf_counter()
    index = faiss.IndexFlatL2(embedding_size)
    index.add(np.array(embeddings))
    elapsed = time.perf_counter() - start
    st.write(f"Created index with {index.ntotal} entries in {elapsed:.2f} seconds")
    st.session_state.index = index

query = st.text_input("Enter a search query")
if query:
    query_embedding = st.session_state.model.encode([query])[0]
    _, b = st.session_state.index.search(np.array([query_embedding]), TOP_K)
    context = ""
    print(f"result vector from faiss: {b}")
    for i in b[0]:
        context += st.session_state.texts[i]
        context += "\n\n- - - - - - -\n\n"
    
    prompt = template.format(context=context, user_question=query)
    with st.expander("Show query"):
        st.write(prompt)

    with st.spinner("Waiting for OpenAI to respond..."):
        response = evaluate_prompt(prompt)
    
    st.write(response)