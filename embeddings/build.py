import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len
)

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
INPUT_FILE = "avg.txt"
INDEX_FILE = "index.pkl"
MAPPING_FILE = "mappings.pkl"
DOCUMENTS_FILE = "documents.pkl"

print(f"Loading {INPUT_FILE}...")
with open(INPUT_FILE) as f:
    lines = f.readlines()

doc = "".join(lines)
texts = text_splitter.split_text(doc)

print(f"Loading embedding model {EMBEDDING_MODEL}...")
model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("Computing embeddings...")
start = time.perf_counter()
db = FAISS.from_texts(texts, model)
db.save_local(INDEX_FILE)

with open(MAPPING_FILE, "wb") as f:
    pickle.dump(db.index_to_docstore_id, f)

with open(DOCUMENTS_FILE, "wb") as f:
    pickle.dump(db.docstore, f)

elapsed = time.perf_counter() - start
print(f"Saved embeddings to {INDEX_FILE}, {MAPPING_FILE}, {DOCUMENTS_FILE} "
    f"in {elapsed:.2f} seconds")