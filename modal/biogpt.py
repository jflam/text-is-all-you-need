import modal
import time
from transformers import pipeline

stub = modal.Stub("biogpt")
image = (
    modal.Image.debian_slim()
    .pip_install("transformers[torch]")
    .pip_install("sacremoses")
)

MODEL_NAME="microsoft/biogpt-large"
MODEL_CACHE_DIR="/cache"
PROMPT="""
A 17 year old male patient presents with a clicking left knee
"""

volume = modal.SharedVolume().persist("transformer-cache")

@stub.function(
    gpu="any",
    shared_volumes={MODEL_CACHE_DIR: volume},
    image=image,
    secret=modal.Secret(
        {
            "TORCH_HOME": MODEL_CACHE_DIR,
            "TRANSFORMERS_CACHE": MODEL_CACHE_DIR,
        }
    )
)
def remote(prompt: str) -> str:
    model = pipeline("text-generation", model=MODEL_NAME, device="cuda:0")
    return model(prompt, max_length=500)[0]["generated_text"]

@stub.local_entrypoint
def main():
    print("Starting the engines...")
    start = time.perf_counter()
    result = remote.call(PROMPT)
    elapsed = time.perf_counter() - start
    print(f"BIOGPT ({elapsed:.2f}s): {result}")
