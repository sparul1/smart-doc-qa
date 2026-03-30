from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from dotenv import load_dotenv
import io, os

load_dotenv()

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": question})
    return {"answer": result["result"]}