import modal
import os

stub = modal.Stub("remote embeddings")
image = (
    modal.Image.debian_slim()
    .pip_install("sentence-transformers")
    .pip_install("numpy")
    .pip_install("langchain")
    .pip_install("pinecone-client")
)

LOCAL_FILE="bible.txt"
MODEL_CACHE_DIR="/cache"
EMBEDDING_MODEL="sentence-transformers/gtr-t5-large"
PINECONE_INDEX=os.environ["PINECONE_INDEX"]
PINECONE_API_KEY=os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT=os.environ["PINECONE_ENVIRONMENT"]

# This is the volume that is used to cache the model files. The 
# model used, gtr-t5-large is ~644MB, so it's a good idea to cache it.
# You will see the model downloaded on first execution only.
volume = modal.SharedVolume().persist("transformer-cache")

import pinecone
import time
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone

@stub.function(
    gpu="any",
    shared_volumes={MODEL_CACHE_DIR: volume},
    image=image,
    secret=modal.Secret(
        {
            "TORCH_HOME": MODEL_CACHE_DIR, 
            "TRANSFORMERS_CACHE": MODEL_CACHE_DIR,
            "PINECONE_API_KEY": PINECONE_API_KEY,
            "PINECONE_ENVIRONMENT": PINECONE_ENVIRONMENT
        }
    )
)
def remote(text: str):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_text(text)
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Pinecone.from_texts(texts, model, index_name=PINECONE_INDEX)

@stub.local_entrypoint
def main():
    with open(LOCAL_FILE, "r") as f:
        text = f.read()

    start = time.perf_counter()
    result = remote.call(text) # run remotely
    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"{len(result)} embeddings generated from {len(text)} characters and {len(text.split())} words")
