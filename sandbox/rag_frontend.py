# Import Streamlit to create a simple web app
import streamlit as st
# Import the RAG backend module
import rag_backend

import os

# Set environment variable to avoid TensorFlow warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Set the page title
st.set_page_config(page_title="Right Ho, Jeeves Q&A")
# update the page title font with html code
new_title = '<p style="font-family:serif;  color:Gray; font-size:42px;">Right Ho, Jeeves Q&A </p>'
# Render the HTML code for the page title
st.markdown(new_title, unsafe_allow_html=True)

# Check if the vector index is already in the session state
if 'vector_index' not in st.session_state:
    with st.spinner("loading..."):  # Display a spinner while loading the index
        st.session_state.vector_index = rag_backend.book_index()  # Load the vector index from the RAG backend

character = st.radio(
    "Who would you like to answer your questions?",
    ["Bertie Wooster", "Jeeves"],
    captions=[
        "Young gentleman and narrator of the story",
        "Bertie's wise and infinitely capable valet"
    ],
)

input_text = st.text_area("input text", label_visibility="collapsed")  # Create a text area for user input
go_button = st.button("Ask", type="primary")  # Create a button to trigger the query

if go_button:

    with st.spinner("running..."):  # Display another spinner while running the query
        # Use the stored vector index to get a response to the question
        response_content = rag_backend.book_rag_response(index=st.session_state.vector_index,
                                                         question=input_text)
        # Display the response
        st.write(response_content)



# run with the command streamlit run /Users/Steven/PycharmProjects/book-RAG/src/rag_frontend.py

