import sqlite_utils
import struct
import time
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split on double newlines or 1000 characters, whichever comes first
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len
)

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
INPUT_FILE = "avg.txt"
OUTPUT_FILE = "embeddings.db"
EMBEDDING_TABLE = "embeddings"

print("Loading file...")
with open(INPUT_FILE) as f:
    lines = f.readlines()

doc = "".join(lines)
texts = text_splitter.split_text(doc)

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

start = time.perf_counter()
embeddings = model.encode(texts)
end = time.perf_counter() - start

print(f"Generated {len(embeddings)} embeddings in {end:.2f} seconds")
print(embeddings.shape)

print("Saving to database...")

# Save the embedding and the original text used to generate the embedding
start = time.perf_counter()
db = sqlite_utils.Database(OUTPUT_FILE)
table = db[EMBEDDING_TABLE]
if not table.exists():
    table.create({
        "embedding": bytes,
        "text": str
    })

assert len(texts) == len(embeddings)

for text, embedding in zip(texts, embeddings):
    table.insert({
        "embedding": struct.pack("f" * 768, *embedding), 
        "text": text}, replace=True)

db.close()
end = time.perf_counter() - start
print(f"Saved {len(embeddings)} embeddings to {OUTPUT_FILE} in {end:.2f} seconds")