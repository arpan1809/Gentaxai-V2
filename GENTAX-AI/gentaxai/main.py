import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from contextlib import asynccontextmanager

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "faiss_index"
    GROQ_MODEL = "llama-3.1-8b-instant"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are GenTaxAI, an expert on Indian tax, GST, and investment regulations.

Retrieved Context:
{context}
---

Instructions:
- If the context above contains relevant information, use it to answer the question
- If the context is empty or irrelevant, use your knowledge of Indian finance to provide a helpful answer
- Be specific and detailed
- Use bullet points for lists
- Never give generic "I'm a helpful assistant" responses

Answer the question directly and completely."""

# Global state
retriever = None
llm = None
models_loaded = False

async def load_models():
    """Load models in background without blocking startup"""
    global retriever, llm, models_loaded
    
    print("🔄 Starting background model loading...")
    
    try:
        # Load LLM first (fast)
        print("📦 Loading Groq LLM...")
        llm = ChatGroq(
            model_name=Config.GROQ_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=0.3
        )
        print(f"✅ LLM ready: {Config.GROQ_MODEL}")
        
        # Load embeddings (slow - ~30-60 seconds)
        print("📦 Loading embedding model (this takes 30-60 seconds)...")
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32}
        )
        print("✅ Embeddings ready")
        
        # Load FAISS index if available
        if os.path.exists(Config.FAISS_INDEX_PATH):
            print(f"📦 Loading FAISS index...")
            vectorstore = FAISS.load_local(
                Config.FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            print("✅ FAISS index ready")
        else:
            print("⚠️  No FAISS index found - using LLM knowledge only")
        
        models_loaded = True
        print("🎉 All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        models_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager for FastAPI"""
    print("=" * 60)
    print("🚀 GenTaxAI Starting...")
    print("=" * 60)
    
    # Start loading models in background
    asyncio.create_task(load_models())
    
    print("✅ App started - models loading in background")
    print("=" * 60)
    
    yield  # Application runs here
    
    print("Shutting down...")

# Create app with lifespan
app = FastAPI(title="GenTaxAI", lifespan=lifespan)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve frontend"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    """Health check - shows model loading status"""
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "llm_ready": llm is not None,
        "retriever_ready": retriever is not None
    }

@app.post("/chat")
async def chat(request: Request):
    """Main chat endpoint"""
    
    # Return helpful message if models still loading
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "answer": "🔄 AI models are still loading (takes 30-60 seconds on first start). Please try again in a moment!",
                "loading": True
            }
        )
    
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    query = data.get("message", "")
    if not query:
        raise HTTPException(status_code=400, detail="'message' required")

    print(f"📨 Query: {query}")
    
    try:
        # Get context from FAISS
        if retriever:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content[:500] for doc in docs[:3]])
            print(f"📚 Retrieved {len(docs)} docs")
        else:
            context = "[No knowledge base available]"
            print("⚠️  Using LLM knowledge only")

        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": query, "context": context})
        
        print(f"✅ Generated answer ({len(answer)} chars)")
        return JSONResponse({"answer": answer})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

