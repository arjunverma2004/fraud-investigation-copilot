from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from pathlib import Path


from dotenv import load_dotenv

load_dotenv()


def create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_db = Chroma(collection_name="my_collection", embedding_function=embeddings, persist_directory='my_chroma_db')
    return vector_db

def add_documents(vector_db, documents):
    vector_db.add_documents(documents)

def load_data(path):
    loader = PyPDFLoader(path, mode="page")
    data = loader.load()
    return data


def split_text_into_chunks(docs):
    spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = spliter.split_documents(docs)
    return chunks


def retrieve_with_mmr(vectorstore, k: int = 3):
    # Enable MMR in the retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",                   # <-- This enables MMR
        search_kwargs={"k": k, "lambda_mult": 0.5}
    )
    return retriever



def load_or_build_vector_db() -> Chroma:
    vector_db = create_vector_db()
    if not vector_db._collection.count():
        print("No documents found in the vector database. Loading and processing documents...")
        file_path = "/home/arjunverma/Coding New/Fraud Investigation copilot/data/fraud_patterns.pdf"
        docs = load_data(file_path)
        chunks = split_text_into_chunks(docs)
        add_documents(vector_db, chunks)
        print("Documents added to the vector database.")
    else:
        print("Database for RAG Initialized.")
    return vector_db

_vector_db = load_or_build_vector_db()

@tool
def search_fraud_patterns(query: str, k: int = 3) -> str:
    """
    Search for fraud patterns information in the knowledge base.
    Call this when the transaction's suspicious pattern isn't obvious from the score alone

    Args:
        query: The search query.
        k: The number of results to return.

    Returns:
        A list of relevant documents.
    """
    retriever = retrieve_with_mmr(_vector_db, k)
    results = retriever.invoke(query)
    
    return "\n\n".join([result.page_content for result in results])