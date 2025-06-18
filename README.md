# 🧠 PDF Chatbot using FastAPI + Hugging Face + Sentence Transformers

This project is a **chatbot API** that answers questions based on the contents of a PDF file using a **Retrieval-Augmented Generation (RAG)** approach.

It combines:
- 🧠 SentenceTransformers for semantic search
- 🤗 OpenAI GPT for text generation
- 🔍 FAISS for fast vector search
- ⚡ FastAPI for serving the chatbot
- 📄 PyMuPDF to read PDF content

---


## 🚀 Getting Started

### 1. Clone the repo or create the folder

If you're starting from scratch:

```bash
mkdir chatbot-backend
cd chatbot-backend
```


---

### 2. Install Dependencies

Create and activate a virtual environment (optional):

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Add Your PDF

Replace the existing `leave_policy.pdf` with your own documents.
Make sure the filename matches in `rag_bot.py`. Also create .env file and add your openai API Key: 
```bash
OPENAI_API_KEY = "sk-proj-..."
```

---

### 4. Run the Server

For generating the chunks run:

```bash
python index_builder.py 
```

```bash
uvicorn rag_bot:app --reload
```

You’ll see:

```
Uvicorn running on http://127.0.0.1:8000
```

---

## 🧪 How to Test

Visit the Swagger UI:

```
http://127.0.0.1:8000/docs
```

Use the `/chat` endpoint and test with:

```json
{
  "message": "How many casual leaves can be taken in a year?"
}
```

---

## 📌 Notes

- The document is chunked using paragraph breaks (`\n\n`).
- Embeddings are created using `all-MiniLM-L6-v2`.
- Answers are generated only from document context using openai.

---

## 📬 Contact

Feel free to modify and expand this project for company bots, document QA, or internal support systems.
Built with ❤️ using open-source tools.
