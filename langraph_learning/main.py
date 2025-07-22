import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import pinecone
import uuid
# import os

# Load API keys
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# api_key = os.getenv("OPENAI_API_KEY")
# db_uri = os.getenv("POSTGRES_URI")
# pinecone_key = os.getenv("PINECONE_API_KEY")
import fitz  # PyMuPDF

# pdf_path = "SiddharthWani_Tech_Resume.pdf"  # Update with your PDF file path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384 dimensions

def generate_embedding(text):
    return model.encode(text).tolist()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))

def store_in_pinecone(text, metadata):
    embedding = generate_embedding(text)
    unique_id = str(uuid.uuid4())
    index.upsert([(unique_id, embedding, metadata)])

def query_pinecone(query_text, top_k=3):
    query_embedding = generate_embedding(query_text)
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_onboarding_message(context_docs, query):
    context_str = "\n".join([doc["metadata"].get("content", "") for doc in context_docs])
    prompt = f"""Use the following context to create a welcome message:

Context:
{context_str}

Query:
{query}
"""
    return llm([HumanMessage(content=prompt)]).content

resume_text = extract_text_from_pdf("SiddharthWani_Tech_Resume.pdf")
store_in_pinecone(resume_text, metadata={"name": "Siddharth Wani", "content": resume_text})

results = query_pinecone("Tell me about the new employee joining our team")
message = generate_onboarding_message(results["matches"], "Write a welcome message for the new hire.")

print(message)

