# **GenTaxAI 🇮🇳**

**GenTaxAI** is an intelligent, conversational AI chatbot designed to demystify Indian tax law and investment regulations. It provides instant, accurate, and citation-backed answers using a powerful Retrieval-Augmented Generation (RAG) pipeline, making complex financial information easy for everyone — from individual taxpayers to business owners.

---

## 📋 **Table of Contents**

* [Features](#-features)
* [How It Works](#-how-it-works)
* [Tech Stack](#-tech-stack)
* [Getting Started](#-getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#-usage)

  * [1. Data Processing Pipeline](#1-data-processing-pipeline)
  * [2. Running the Application](#2-running-the-application)
* [Deployment on Render](#-deployment-on-render)
* [Project Structure](#-project-structure)

---

## ✨ **Features**

* **Conversational Q&A:** Ask complex Indian tax and finance questions in plain English.
* **Comprehensive Knowledge Base:** Uses official documents (GST, ITR, SEBI, etc.).
* **Citation-backed Responses:** Every answer is grounded in retrieved source text.
* **High-performance Backend:** FastAPI + Groq LPU™ for ultra-fast inference.
* **End-to-End Pipeline:** Includes full data ingestion, cleaning, structuring, and vectorization steps.
* **Deploy-ready:** Easily deployable to Render with automatic build pipeline.

---

## 🧠 **How It Works**

GenTaxAI uses a robust RAG architecture:

### 1. **Data Ingestion**

Raw PDF files (GST, Income Tax, SEBI) are extracted using `01_extract_pdfs.py`.

### 2. **Text Cleaning**

`02_clean_text.py` removes noise, headers, page numbers, and formatting.

### 3. **Structuring**

`03_structure_data.py` organizes the cleaned text into structured JSON chunks.

### 4. **Vector Database Creation**

`base.py`:

* Loads JSON text chunks
* Splits them using LangChain text splitters
* Embeds them using Sentence Transformers
* Stores embeddings inside a **FAISS** vector index

### 5. **Retrieval + Generation**

* FAISS finds the most relevant text chunks
* LLM (via **Groq API**) generates accurate, citation-backed answers

---

## 🛠️ **Tech Stack**

**Backend:**

* FastAPI
* LangChain
* Groq API (LLM inference)
* FAISS (vector retrieval)
* Sentence Transformers

**Infrastructure:**

* Gunicorn + Uvicorn workers
* Python 3.11

**Frontend:**

* HTML / CSS / JS
* Fetch API for querying backend

---

## 🚀 **Getting Started**

## **Prerequisites**

* Python **3.11+**
* Groq API key → [https://console.groq.com/keys](https://console.groq.com/keys)

---

## **Installation**

### 1. Clone the repository

```sh
git clone https://github.com/arpan1809/Gentaxai-V2.git
cd Gentaxai-V2
```

### 2. Create & activate a virtual environment

```sh
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Add your `.env` file

Create `.env` in the project root:

```
GROQ_API_KEY="your_api_key_here"
```

---

# **USAGE**

GenTaxAI has **two steps**:
1️⃣ Build the knowledge base
2️⃣ Run the web server

---

## **1. Data Processing Pipeline**

### Create required folders

```sh
mkdir -p data/raw data/processed data/structured knowledge_base
```

### Place PDF documents

Put your documents inside:

```
data/raw/
```

### Run pipeline sequentially

```sh
python 01_extract_pdfs.py
python 02_clean_text.py
python 03_structure_data.py
python 04_build_kb.py
```

This generates your **FAISS index**:

```
faiss_index/
```

---

## **2. Running the Application**

Start FastAPI backend:

```sh
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000
```

Your chatbot is now live locally.

---

## ☁️ **Deployment on Render**

This project is fully compatible with Render.

### Steps:

1. Fork this repo on GitHub
2. Create a **Web Service** on Render
3. Use these settings:

| Setting        | Value                                                                          |
| -------------- | ------------------------------------------------------------------------------ |
| Runtime        | Python                                                                         |
| Build Command  | `./render-build.sh`                                                            |
| Start Command  | `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT` |
| Root Directory | *(leave empty)*                                                                |

4. Add environment variable:

```
GROQ_API_KEY = your_key
```

5. Deploy ✔
   Render will:

* Install dependencies
* Run PDF → KB pipeline
* Launch your FastAPI App

---

## 📁 **Project Structure**

```
.
├── .env                  # Secret keys
├── base.py
├── main.py               # FastAPI backend
├── requirements.txt
├── render-build.sh       # Render build script
├── static/
│   └── index.html        # Frontend UI
│   └── style.css
│   └── script.js
├── data/
│   ├── raw/
│   ├── processed/
│   └── structured/
├── knowledge_base/
├── faiss_index/
├── 01_extract_pdfs.py
├── 02_clean_text.py
├── 03_structure_data.py
├── 04_build_kb.py
├── sessions.py
└── utils.py
```

---

If you'd like, I can also generate:

✅ Beautiful project banner
✅ MIT License file
✅ Badges (Python version, backend, deploy status)
✅ Contributing guidelines
📌 Just say **"Add badges & license"** or **"Add contribution guide"**!
