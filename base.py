import os
import json
import re
import time
from typing import List, Dict

# Simple configuration holder for paths and model name
class Config:
    """
    Minimal Config used by this script; values can be overridden via environment variables:
      - KB_PATH: directory containing knowledge base files (default: ./knowledge_base)
      - EMBEDDING_MODEL: HuggingFace embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)
      - FAISS_INDEX_PATH: directory where FAISS index will be saved (default: ./faiss_index)
    """
    KB_PATH = os.environ.get("KB_PATH", os.path.join(os.getcwd(), "knowledge_base"))
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", os.path.join(os.getcwd(), "faiss_index"))

# LangChain components for vectorization
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def clean_text(text: str) -> str:
    """A robust text cleaner."""
    text = text.replace("\u00a0", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_documents_from_kb() -> List[Document]:
    """
    Loads, cleans, and processes documents from the knowledge_base directory,
    preparing them for vectorization.
    """
    documents: List[Document] = []
    
    if not os.path.isdir(Config.KB_PATH):
        print(f"Error: Knowledge base directory '{Config.KB_PATH}' not found.")
        return documents

    print(f"Loading files from '{Config.KB_PATH}'...")
    for filename in os.listdir(Config.KB_PATH):
        if not filename.lower().endswith(".json"):
            continue

        filepath = os.path.join(Config.KB_PATH, filename)
        corpus_text = ""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extracts text based on the JSON structure
            if isinstance(data, dict) and "text" in data:
                corpus_text = str(data["text"])
            else:
                corpus_text = json.dumps(data, ensure_ascii=False)

        except Exception as e:
            print(f"Could not process {filename} as JSON, treating as raw text. Error: {e}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    corpus_text = f.read()
            except Exception as read_e:
                print(f"Fatal: Could not read file {filename}. Skipping. Error: {read_e}")
                continue
        
        cleaned_text = clean_text(corpus_text)
        doc = Document(page_content=cleaned_text, metadata={"source": filename})
        documents.append(doc)

    print(f"Successfully loaded and cleaned {len(documents)} documents.")
    return documents

def main():
    """
    Main function to build and save the FAISS vector index.
    """
    print("--- Starting FAISS Knowledge Base Build ---")
    
    # 1. Load and process documents using your logic
    start_time = time.time()
    docs = load_documents_from_kb()
    if not docs:
        print("No documents found to index. Exiting.")
        return
    print(f"-> Document loading took: {time.time() - start_time:.2f}s")

    # 2. Split documents into chunks
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(splits)} chunks.")
    print(f"-> Document splitting took: {time.time() - start_time:.2f}s")

    # 3. Initialize embedding model
    print(f"Loading embedding model: '{Config.EMBEDDING_MODEL}'...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    print(f"-> Embedding model loaded in: {time.time() - start_time:.2f}s")

    # 4. Create FAISS index from chunks
    print("Creating FAISS index... This might take a while.")
    start_time = time.time()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print(f"-> Index creation took: {time.time() - start_time:.2f}s")

    # 5. Save the index to disk
    if os.path.exists(Config.FAISS_INDEX_PATH):
        print(f"Overwriting existing index at '{Config.FAISS_INDEX_PATH}'")
    else:
        print(f"Saving new index to '{Config.FAISS_INDEX_PATH}'")
        
    vectorstore.save_local(Config.FAISS_INDEX_PATH)
    print("--- FAISS Knowledge Base has been built successfully! ---")

if __name__ == "__main__":
    main()
