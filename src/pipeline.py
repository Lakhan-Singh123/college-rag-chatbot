import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from our secure .env file
load_dotenv()

def format_docs(docs):
    """Combines all retrieved text chunks into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_pipeline(user_question):
    """
    Loads the local FAISS index, retrieves matching chunks, and uses Gemini to answer via LCEL.
    """
    # 1. Initialize our embedding model (Matches ingest.py exactly)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    print("📂 Loading local FAISS index...")
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Turn our vector store into a Retriever (fetches top 3 matching chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 3. Initialize Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    # 4. Create a strict System Prompt Template
    system_prompt_template = """
    You are a helpful college assistant. Answer the user's question based strictly on the provided context. 
    If the answer cannot be found in the context, say exactly: "I'm sorry, I cannot find that information in the provided document." 
    Do not try to make up answers or use outside knowledge.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=system_prompt_template,
        input_variables=["context", "question"]
    )
    
    # 5. Build the Modern LCEL RAG Chain
    # This pipeline reads: Retrieve context -> Pass question -> Format prompt -> Send to LLM -> Parse text string output
    print("⛓️ Assembling LCEL RAG Pipeline...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. Run the query through the pipeline
    print(f"🔍 Searching database and generating answer for: '{user_question}'...")
    answer = rag_chain.invoke(user_question)
    
    # We also fetch the raw docs manually just so we can show you the sources on your terminal screen!
    retrieved_docs = retriever.invoke(user_question)
    
    return {"answer": answer, "context": retrieved_docs}

if __name__ == "__main__":
    test_query = "What program or internship is this document about?"
    
    result = run_rag_pipeline(test_query)
    
    print("\n================ 🤖 CHATBOT ANSWER ================")
    print(result["answer"])
    print("===================================================\n")
    
    print("📄 SOURCE SNIPPETS RETRIEVED:")
    for i, doc in enumerate(result["context"]):
        print(f"\n[Source {i+1}] Page {doc.metadata.get('page', 'Unknown')}:")
        print(doc.page_content[:150] + "...")