import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

class Config:
    """
    Holds application configuration, reading from environment variables
    with sensible defaults.
    """
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


app = FastAPI(title="GenTaxAI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


qa_chain = None
try:
    print("Initializing RAG pipeline...")
    
    
    if not Config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

   
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    if not os.path.exists(Config.FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at '{Config.FAISS_INDEX_PATH}'. Please run the data pipeline first.")
    
    vectorstore = FAISS.load_local(
        Config.FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        temperature=0,
        model_name=Config.GROQ_MODEL,
        api_key=Config.GROQ_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    print("RAG pipeline initialized successfully.")

except Exception as e:
    print(f"FATAL: Failed to initialize RAG pipeline: {e}")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/chat", summary="Process a user query")
async def chat(request: Request):
    """
    Receives a user's message, gets an answer from the RAG pipeline,
    and saves the conversation history.
    """
    if not qa_chain:
        raise HTTPException(status_code=500, detail="RAG chain is not initialized. Please check server logs.")

    try:
        data = await request.json()
        session_id = data.get("session_id")
        user_message = data.get("message")

        if not user_message:
            raise HTTPException(status_code=400, detail="Field 'message' is required.")

      
        result = qa_chain({"query": user_message})
        bot_response = result.get("result", "I am sorry, but I could not find an answer.")

        

        return JSONResponse(content={"answer": bot_response})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    print("Warning: 'static' directory not found. The web interface will not be available.")

