import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_HANDLER"] = "langchain"

import faiss
import openai
import pickle
import streamlit as st
from langchain.chains import HypotheticalDocumentEmbedder
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
USE_HYDE = False

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

def read_index(embeddings) -> FAISS:
    index = faiss.read_index(INDEX_FILE)

    with open(MAPPING_FILE, "rb") as f:
        index_to_docstore_id = pickle.load(f)

    with open(DOCUMENTS_FILE, "rb") as f:
        docstore = pickle.load(f)
    
    return FAISS(embeddings.embed_query, 
        index, 
        docstore=docstore, 
        index_to_docstore_id=index_to_docstore_id)

if "db" not in st.session_state:
    llm = AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
        openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
        model_name="text-davinci-003", temperature=0.0, max_tokens=1000,
        n=4, best_of=4)
    base_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if USE_HYDE:
        embeddings = HypotheticalDocumentEmbedder.from_llm(llm=llm, 
            base_embeddings=base_embeddings, prompt_key="web_search")
        st.session_state.db = read_index(embeddings)
    else:
        st.session_state.db = read_index(base_embeddings)

    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", k=TOP_K, 
                                    vectorstore=st.session_state.db,
                                    return_source_documents=True)
    st.session_state.qa = qa
    st.session_state.past = []
    st.session_state.generated = []
    st.session_state.sources = []

query = st.text_input("Enter a question")
if query:
    with st.spinner("Waiting for OpenAI to respond..."):
        result = st.session_state.qa({"query": query})
        st.session_state.past.append(query)
        st.session_state.generated.append(result["result"])
        st.session_state.sources.append(result["source_documents"])
    
    if 'generated' in st.session_state:
        for i in range(len(st.session_state.generated)-1, -1, -1):
            message(st.session_state.generated[i], key=str(i))
            with st.expander("Show sources"):
                for result in st.session_state.sources[i]:
                    st.write(result.page_content)
            message(st.session_state.past[i], is_user=True, key=str(i) + "_user")