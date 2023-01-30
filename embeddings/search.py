import faiss
import openai
import os
import pickle
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS

"""## DeepMemory Search"""

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
    model_name="text-davinci-003", temperature=0)

def evaluate_prompt(prompt: str) -> str:
    return llm(prompt)

template = PromptTemplate(
    input_variables=["user_question", "context"],
    template="""
The following is a question from the user: {user_question} 

Please answer the question using the context below:
{context}
"""
)

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

query = st.text_input("Enter a search query")
if query:
    results = st.session_state.db.similarity_search(query, k=TOP_K)
    context = ""
    for result in results:
        context += result.page_content
        context += "\n\n- - - - - - -\n\n"
    
    prompt = template.format(context=context, user_question=query)
    with st.expander("Show query"):
        st.write(prompt)

    with st.spinner("Waiting for OpenAI to respond..."):
        response = evaluate_prompt(prompt)
    
    st.write(response)