import streamlit as st
import rag_backend

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

st.set_page_config(page_title="Book Q&A with RAG")

new_title = '<p style="font-family:sans-serif;  color:Green; font-size:42px;">Book Q&A with RAG </p>'

st.markdown(new_title, unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner("loading..."):
        st.session_state.vector_index = rag_backend.book_index()

input_text = st.text_area("input text", label_visibility="collapsed")
go_button = st.button("Ask", type="primary")

if go_button:

    with st.spinner("running..."):
        response_content = rag_backend.book_rag_response(index=st.session_state.vector_index,
                                                         question=input_text)
        st.write(response_content)
