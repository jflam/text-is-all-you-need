import modal

stub = modal.Stub("remote embeddings")
image = (
    modal.Image.debian_slim()
    .pip_install("sentence-transformers")
    .pip_install("numpy")
    .pip_install("langchain")
)

LOCAL_FILE="bible.txt"
EMBEDDING_MODEL="sentence-transformers/gtr-t5-large"

# This is the volume that is used to cache the model files. The 
# model used, gtr-t5-large is ~644MB, so it's a good idea to cache it.
# You will see the model downloaded on first execution only.
volume = modal.SharedVolume().persist("transformer-cache")

import time
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

@stub.function(
    gpu="any",
    shared_volumes={"/cache": volume},
    image=image,
    secret=modal.Secret(
        {"TORCH_HOME": "/cache", "TRANSFORMERS_CACHE": "/cache"}
    )
)
def remote(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_text(text)
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = model.embed_documents(texts)
    return np.array(embeddings, dtype=np.float32)

@stub.local_entrypoint
def main():
    with open(LOCAL_FILE, "r") as f:
        text = f.read()

    start = time.perf_counter()
    result = remote.call(text) # run remotely
    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"{len(result)} embeddings generated from {len(text)} characters and {len(text.split())} words")
