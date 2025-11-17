import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio

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

app = FastAPI(title="GenTaxAI")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
retriever = None
llm = None
is_loading = True

@app.on_event("startup")
async def load_rag_pipeline():
    """Load models asynchronously to avoid blocking startup"""
    global retriever, llm, is_loading
    
    print("=" * 60)
    print("🚀 GenTaxAI Starting...")
    print("=" * 60)
    
    # Start loading in background
    asyncio.create_task(load_models_background())
    
    print("✅ Application ready - models loading in background")
    print("=" * 60)

async def load_models_background():
    """Load heavy models in the background"""
    global retriever, llm, is_loading
    
    try:
        print("📦 Loading Groq LLM...")
        llm = ChatGroq(
            model_name=Config.GROQ_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=0.3
        )
        print(f"✅ LLM loaded: {Config.GROQ_MODEL}")
        
        print("📦 Loading embedding model (this may take 30-60 seconds)...")
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32}  # Optimize batch size
        )
        print("✅ Embeddings loaded")
        
        if os.path.exists(Config.FAISS_INDEX_PATH):
            print(f"📦 Loading FAISS index from {Config.FAISS_INDEX_PATH}...")
            vectorstore = FAISS.load_local(
                Config.FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            print(f"✅ FAISS index loaded successfully")
        else:
            print(f"⚠️  FAISS index not found - using LLM knowledge only")
        
        is_loading = False
        print("=" * 60)
        print("🎉 All models loaded! System fully operational")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR loading models: {e}")
        is_loading = False

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": not is_loading,
        "llm": llm is not None,
        "retriever": retriever is not None
    }

@app.post("/chat")
async def chat(request: Request):
    """Main chat endpoint"""
    
    # Check if models are still loading
    if is_loading:
        raise HTTPException(
            status_code=503, 
            detail="Models are still loading. Please wait 30-60 seconds and try again."
        )
    
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    query = data.get("message", "")
    if not query:
        raise HTTPException(status_code=400, detail="'message' field required")

    print(f"\n📨 Query: {query}")
    
    try:
        # Get context
        if retriever:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content[:500] for doc in docs[:3]])
            print(f"📚 Retrieved {len(docs)} docs")
        else:
            context = "[No knowledge base available]"
            print("⚠️  Using LLM knowledge only")

        # Build and execute chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": query, "context": context})
        
        print(f"✅ Response generated ({len(answer)} chars)")
        return JSONResponse({"answer": answer})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))


