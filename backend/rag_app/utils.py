import os
from lib2to3.fixes.fix_input import context

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pinecone
from dotenv import load_dotenv
from openai import api_key, embeddings

load_dotenv()

# Import API keys from .env file
class RAGProcessor:
    def __init__(self):
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENV')
        )
        self.index_name = "document-store"
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

    def process_document(self, file_path):
        # Load and split document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # Generate embeddings and store in Pinecone Vector DB
        index = pinecone.Index(self.index_name)
        vector_ids = []

        for i,chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk.page_content)
            # page_content an attribute of class Document can be used for storing content of the document or chunk
            vector_id = f"doc_{os.path.basename(file_path)}_{i}"
            index.upsert([(vector_id, embedding, {"text": chunk.page_content})])
            vector_ids.append(vector_id)

        return vector_ids

    def query_documents(self, query, top_k=3):
        # Generate query embeddings
        query_embedding = self.embeddings.embed_query(query)

        # Search in Pinecone
        index = pinecone.Index(self.index_name)
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract and return relevant context
        contexts = [result.metadata["text"] for result in results.matches]
        return "\n".join(contexts)