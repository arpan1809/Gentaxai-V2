import os
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


# -------------------------
# Configuration
# -------------------------
class Config:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


app = FastAPI(title="GenTaxAI", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Lazy-loading for RAG (Do NOT load at import time)
# -------------------------------------------------------
qa_chain = None


def load_rag():
    """
    Loads the embeddings, FAISS index, and LLM only when /chat is called.
    Prevents high RAM usage during Render startup.
    """
    global qa_chain
    if qa_chain:
        return qa_chain

    print("⚡ Loading RAG pipeline...")

    if not Config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing.")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    if not os.path.exists(Config.FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index missing at {Config.FAISS_INDEX_PATH}"
        )

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

    print("✅ RAG loaded.")
    return qa_chain


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()

    user_message = data.get("message")
    if not user_message:
        raise HTTPException(status_code=400, detail="message required")

    qa = load_rag()
    result = qa({"query": user_message})

    answer = result.get("result", "No answer available.")

    return JSONResponse({"answer": answer})





