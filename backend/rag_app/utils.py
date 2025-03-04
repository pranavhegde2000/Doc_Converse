import os
from lib2to3.fixes.fix_input import context
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.chains import LLMChain
from openai import api_key, embeddings

load_dotenv()

# Import API keys from .env file
class RAGProcessor:
    def __init__(self):
        # Initialize Pinecone with new syntax
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = "document-store"

        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )


        self.index = self.pc.Index(self.index_name)
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))

        #Prompt Template
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that answers questions based on the provided context.

            Context: {context}

            Question: {query}

            Answer the question based only on the provided context. If the context doesn't contain
            enough information to answer the question, simply say that you don't have enough information.
            """
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def process_document(self, file_path):
        file_path = Path(file_path)

        # Load and split document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Generate embeddings and store in Pinecone Vector DB
        vector_ids = []

        for i,chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk.page_content)
            vector_id = f"doc_{file_path.name}_{i}"
            self.index.upsert([(vector_id, embedding, {"text": chunk.page_content})])
            vector_ids.append(vector_id)

        return vector_ids

    def query_documents(self, query, top_k=3):
        # Generate query embeddings
        query_embedding = self.embeddings.embed_query(query)

        # Search in Pinecone

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Extract and return relevant context
        contexts = [match['metadata']['text'] for match in results['matches']]
        clean_context = " ".join(contexts).replace("\n", " ").replace("  ", " ")
        return clean_context

    def generate_answer(self, query, context):
        return self.chain.run(context=context, query=query)