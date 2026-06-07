import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# NEW IMPORTS: Bringing in the Google Embedding connector and FAISS Vector Store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from the .env file
load_dotenv()

# Ensure our Google API Key is set in the environment
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ ERROR: GOOGLE_API_KEY not found in .env file!")

def process_and_store_pdf(file_path, vector_store_path="faiss_index"):
    """
    Loads a PDF, chunks it, generates vector embeddings, and saves them into a local FAISS database.
    """
    print(f"🔄 Loading document from: {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Successfully loaded {len(documents)} pages.")
    
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    
    # 1. Initialize the Google Embedding Model
    print("🤖 Initializing Google Embedding Model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # 2. Convert text chunks to mathematical vectors and save them into FAISS
    print("🧠 Converting text chunks to vectors and building FAISS database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ FAISS index built successfully.")
    
    # 3. Save the FAISS database locally so we don't have to re-embed it every time
    print(f"💾 Saving FAISS database locally to folder: '{vector_store_path}'...")
    vector_store.save_local(vector_store_path)
    print("🎉 All steps complete! Your vector database is ready.")
    
    return vector_store

if __name__ == "__main__":
    sample_pdf_path = "data/syllabus.pdf"
    
    if os.path.exists(sample_pdf_path):
        process_and_store_pdf(sample_pdf_path)
    else:
        print(f"⚠️ Test run skipped: Please make sure 'data/syllabus.pdf' exists!")