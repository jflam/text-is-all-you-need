import faiss
import openai
import os
import pickle
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import AzureOpenAI
from langchain import VectorDBQA, FAISS
from streamlit_chat import message

"""
## Deep Questions
(with apologies to Cal Newport)"""

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
INDEX_FILE = "index.pkl"
MAPPING_FILE = "mappings.pkl"
DOCUMENTS_FILE = "documents.pkl"
TOP_K = 10

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

llm = AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
    openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
    model_name="text-davinci-003", temperature=0.0, max_tokens=1000)

def evaluate_prompt(prompt: str) -> str:
    return llm(prompt)

if "model" not in st.session_state:
    model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    st.session_state.model = model

    index = faiss.read_index(INDEX_FILE)

    with open(MAPPING_FILE, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    with open(DOCUMENTS_FILE, "rb") as f:
        docstore = pickle.load(f)
    
    db = FAISS(model.embed_query, 
        index, 
        docstore=docstore, 
        index_to_docstore_id=index_to_docstore_id)

    st.session_state.db = db
    st.session_state.past = []
    st.session_state.generated = []

query = st.text_input("Enter a question")
if query:
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="refine", 
                                    vectorstore=st.session_state.db)

    with st.spinner("Waiting for OpenAI to respond..."):
        response = qa.run(query)
        st.session_state.past.append(query)
        st.session_state.generated.append(response)
    
    if 'generated' in st.session_state:
        for i in range(len(st.session_state.generated)-1, -1, -1):
            message(st.session_state.generated[i], key=str(i))
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user")