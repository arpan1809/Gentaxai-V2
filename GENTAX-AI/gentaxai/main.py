import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="GenTaxAI - Debug Version")
retriever = None
llm = None

@app.on_event("startup")
def load_rag_pipeline():
    global retriever, llm
    try:
        print("=" * 50)
        print("STARTING GenTaxAI - Debug Version")
        print("=" * 50)

        llm = ChatGroq(
            model_name=Config.GROQ_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=0.3
        )
        print(f"✓ LLM initialized: {Config.GROQ_MODEL}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        
        if os.path.exists(Config.FAISS_INDEX_PATH):
            vectorstore = FAISS.load_local(
                Config.FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            print(f"✓ FAISS index loaded from: {Config.FAISS_INDEX_PATH}")
        else:
            print(f"⚠ FAISS index NOT found at: {Config.FAISS_INDEX_PATH}")
            retriever = None

        print("=" * 50)
    except Exception as e:
        print(f"✗ FATAL ERROR: {e}")

# Serve static files and index page
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(request: Request):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized")

    data = await request.json()
    query = data.get("message", "")

    if not query:
        raise HTTPException(status_code=400, detail="'message' required")

    print("\n" + "=" * 50)
    print(f"📨 USER QUERY: {query}")
    
    try:
        # Get context
        if retriever:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content[:500] for doc in docs[:3]])
            print(f"📚 Retrieved {len(docs)} documents")
            print(f"📄 First doc preview: {context[:200]}...")
        else:
            context = "[No FAISS index available]"
            print("⚠ No retriever - using LLM knowledge only")

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # Generate answer
        print("🤖 Generating answer...")
        answer = chain.invoke({"question": query, "context": context})
        
        print(f"✅ ANSWER: {answer[:200]}...")
        print("=" * 50 + "\n")

        return JSONResponse({"answer": answer})
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
