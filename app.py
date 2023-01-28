import streamlit as st
from streamlit_chat import message

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
    response = f"I'm sorry, Dave. I'm afraid I can't do that: {user_input}" 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if len(st.session_state.generated) > 0:
    for i in range(len(st.session_state.generated) - 1, -1, -1):
        message(st.session_state.generated[i], key=str(i))
        message(st.session_state.past[i], is_user=True, 
            avatar_style="jdenticon", key=f"{i}_user")
        