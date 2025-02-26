from socketserver import BaseRequestHandler

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000/api')

st.title("DocConverse")

# File upload section
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    files = {'file': uploaded_file}
    response = requests.post(f"{BACKEND_URL}/upload/", files=files)

    if response.status_code == 201:
        st.success("Document uploaded and processed successfully!")
    else:
        st.error("Error processing document")

# Query section
st.header("Ask Questions")
query = st.text_input("Enter your question about the document:")

if query:
    if st.button("Get Answer"):
        response = requests.post(
            f"{BACKEND_URL}/query/",
            json={'query': query}
        )

        if response.status_code == 200:
            st.write("Answer:")
            st.write(response.json()['context'])
        else:
            st.error("Error processing query")