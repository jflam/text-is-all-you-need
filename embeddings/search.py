import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import openai
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

def read_index(base_embeddings) -> FAISS:
    index = faiss.read_index(INDEX_FILE)

    with open(MAPPING_FILE, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    with open(DOCUMENTS_FILE, "rb") as f:
        docstore = pickle.load(f)
    
    return FAISS(base_embeddings.embed_query, 
        index, 
        docstore=docstore, 
        index_to_docstore_id=index_to_docstore_id)

if "db" not in st.session_state:
    base_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    st.session_state.db = read_index(base_embeddings)
    st.session_state.past = []
    st.session_state.generated = []
    st.session_state.sources = []

query = st.text_input("Enter a question")
if query:
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", k=10, 
                                    vectorstore=st.session_state.db,
                                    return_source_documents=True)
    
    with st.spinner("Waiting for OpenAI to respond..."):
        result = qa({"query": query})
        answer = result["result"]
        st.session_state.past.append(query)
        st.session_state.generated.append(answer)
        st.session_state.sources.append(result["source_documents"])
    
    if 'generated' in st.session_state:
        for i in range(len(st.session_state.generated)-1, -1, -1):
            message(st.session_state.generated[i], key=str(i))
            with st.expander("Show sources"):
                for result in st.session_state.sources[i]:
                    st.write(result.page_content)
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user")