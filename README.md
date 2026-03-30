    # Smart Document Q&A API

A serverless RAG (Retrieval-Augmented Generation) pipeline that lets you upload any PDF and ask questions about it — powered by Google Gemini and LangChain.

## How it works

1. Upload a PDF → it gets split into chunks and stored as embeddings in a vector database
2. Ask a question → relevant chunks are retrieved and sent to Gemini as context
3. Gemini answers based on your document's content

## Architecture
```
User uploads PDF
      ↓
FastAPI /upload endpoint
      ↓
PDF parsed → split into chunks → embedded via Gemini API
      ↓
Stored in ChromaDB (vector database)

User asks question
      ↓
FastAPI /ask endpoint
      ↓
Question embedded → similar chunks retrieved from ChromaDB
      ↓
Chunks + question sent to Gemini → answer returned
```
<img width="1425" height="777" alt="image" src="https://github.com/user-attachments/assets/47addd3e-28c5-415e-965b-8e258d71421d" />
<img width="1458" height="801" alt="image" src="https://github.com/user-attachments/assets/7613399a-f408-4c7c-bcb5-e354f9183f49" />



## Tech Stack

## Tech stack

| Layer | Library |
|---|---|
| **HTTP API** | FastAPI + Uvicorn (ASGI) |
| **PDF parsing** | pypdf |
| **Text splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | HuggingFace Embeddings (via LangChain) |
| **Vector store** | ChromaDB (persisted locally in `chroma_db/`) |
| **LLM** | Google Gemini 2.5 Flash (via `langchain-google-genai`) |
| **RAG chain** | LangChain Core — `ChatPromptTemplate`, `RunnablePassthrough`, `StrOutputParser` |
| **Config** | `python-dotenv` — secrets loaded from `.env` |

## Setup

1. Clone the repo
```
   git clone https://github.com/your-username/smart-doc-qa.git
   cd smart-doc-qa
```

2. Create a virtual environment
```
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```
   pip install -r requirements.txt
```

4. Add your Gemini API key
```
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
```

5. Run the app
```
   uvicorn main:app --reload
```

6. Open http://localhost:8000/docs to test the API

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload a PDF to index |
| `/ask` | POST | Ask a question about uploaded docs |

## Get a free Gemini API key
👉 https://aistudio.google.com
