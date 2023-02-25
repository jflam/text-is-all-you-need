import json
import importlib
import streamlit as st
from langchain.prompts import PromptTemplate
from streamlit_chat import message
from llm import evaluate_prompt

modality_lookup = {
    "direct": "llm",
    "code": "exec_code",
    "search": "search",
}

template = PromptTemplate(
    input_variables=["user_question"],
    template="""
The following is a question from the user. I want you to consider whether it
is best to answer the question directly (direct), write a Python program
(code), or search the web (search). 

If you choose to write a Python program, only return Python code in your 
response.

If you choose to search the web, only return search terms in your response.

Anything else you can answer directly and return your answer in your response.

I want you to return the modality of your response structured in JSON, like
the following example:

{{
  "modality": "direct, code or search",
  "response": "your response"
}}

User's question:
{user_question}
"""
)

"""
## Text Is All You Need

A contrarian take on the user interface of the future. 
"""

if "generated" not in st.session_state:
    st.session_state.generated = []
    st.session_state.past = []

user_input = st.text_input("You", 
    placeholder="Talk to your AI overlords", key="input")

if user_input:
    expanded_prompt = template.format(user_question=user_input)
    st.write(f"Expanded prompt: {expanded_prompt}")
    response_json = evaluate_prompt(expanded_prompt)
    st.json(response_json)
    response_obj = json.loads(response_json)
    modality = response_obj["modality"]
    response = response_obj["response"]
    if modality not in modality_lookup:
        st.write(f"Unknown modality: {modality}")
    else:
        st.write(f"Modality: {modality}")
        st.write(f"Response: {response}")
        # TODO: some kind of confirmation
        modality_module = modality_lookup[response_obj["modality"]]
        module = importlib.import_module(modality_module)
        if modality != "direct":
            evaluate_prompt_func = getattr(module, "evaluate_prompt")
            final_response = evaluate_prompt_func(response)
            if modality == "search":
                st.markdown(final_response, unsafe_allow_html=True)
        else:
            final_response = response

        st.session_state.past.append(user_input)
        st.session_state.generated.append(final_response)

if len(st.session_state.generated) > 0:
    for i in range(len(st.session_state.generated) - 1, -1, -1):
        message(st.session_state.generated[i], key=str(i))
        message(st.session_state.past[i], is_user=True, 
            avatar_style="jdenticon", key=f"{i}_user")
        