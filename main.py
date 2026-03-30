from fastapi import FastAPI, UploadFile, File
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pypdf import PdfReader
from dotenv import load_dotenv
import io, os

load_dotenv()

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Use local embeddings (no API calls needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    text = "\n".join(page.extract_text() for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    vectorstore.add_texts(chunks)
    return {"message": f"Indexed {len(chunks)} chunks from {file.filename}"}

@app.post("/ask")
async def ask_question(question: str):
    # Use same local embeddings for consistency
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    # Use gemini-2.5-flash (available in your API key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create RAG chain using LCEL
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(question)
    return {"answer": answer}

