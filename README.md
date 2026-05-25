# GenAI RAG Assistant

A GenAI-powered chat assistant that answers user questions using a custom document knowledge base. Built with FastAPI, Sentence Transformers, and Groq LLM.

---

## Architecture Diagram
User Question
↓
Generate Query Embedding (Sentence Transformers - all-MiniLM-L6-v2)
↓
Cosine Similarity Search (scikit-learn)
↓
Retrieve Top 3 Relevant Chunks (threshold: 0.3)
↓
Build RAG Prompt (Context + Conversation History + Question)
↓
Groq LLM (llama-3.3-70b-versatile)
↓
Grounded Response → User

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | Cosine Similarity (scikit-learn) |
| LLM | Groq - llama-3.3-70b-versatile |
| Storage | In-memory (session history) |
| Frontend | HTML, CSS, JavaScript |

---

## RAG Workflow Explanation

1. User types a question in the chat interface
2. The question is converted into a vector embedding using Sentence Transformers
3. Cosine similarity is computed between the query embedding and all document embeddings
4. Top 3 most relevant chunks are retrieved (only if similarity score is above 0.3 threshold)
5. A structured RAG prompt is built combining context, conversation history, and the question
6. Groq LLM generates a grounded answer based only on the retrieved context
7. The answer is returned to the frontend and displayed in the chat

---

## Embedding Strategy

- Model used: `all-MiniLM-L6-v2` from Sentence Transformers
- Runs completely locally — no embedding API key required
- All documents are embedded once at server startup and stored in memory
- Each document chunk is converted to a 384-dimensional vector
- The same model is used for both document embeddings and query embeddings to ensure they exist in the same vector space

---

## Similarity Search Logic

- Method: Cosine Similarity using scikit-learn
- Score range: 0 (completely unrelated) to 1 (identical)
- Top 3 chunks are selected from all documents
- Threshold: 0.3 — if the best score is below this, the bot replies with "I do not have enough information"
- This prevents hallucination by ensuring only relevant content is passed to the LLM

---

## Prompt Design Reasoning

The prompt is structured as:
You are a helpful customer support assistant.
Answer the user question using ONLY the provided context below.
If the context does not contain enough information, say: "I do not have enough information to answer that question."
Context:
{retrieved chunks}
Conversation History:
{last 5 exchanges}
Question:
{user question}
Answer:

Key design decisions:
- "ONLY the provided context" — prevents hallucination
- Conversation history included — supports follow-up questions
- Clear fallback instruction — handles out-of-scope questions gracefully
- Low temperature equivalent — Groq model set to be factual and consistent

---

## Project Structure
GenAI-RAG-Assistant/
├── backend/
│   ├── main.py           # FastAPI server, RAG pipeline, all logic
│   ├── docs.json         # Knowledge base documents
│   ├── requirements.txt  # Python dependencies
│   └── .env              # API key (not uploaded)
├── frontend/
│   ├── index.html        # Chat UI structure
│   ├── style.css         # Chat UI styling
│   └── script.js         # Frontend logic, API calls
├── .gitignore
└── README.md

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/suryavikasindraganti/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
pip install groq
```

### 4. Get free Groq API key
- Go to https://console.groq.com
- Sign up and create an API key
- Create `backend/.env` file:
LLM_API_KEY=your_groq_api_key_here

### 5. Run the server
```bash
cd backend
uvicorn main:app --reload
```

### 6. Open the chat
http://127.0.0.1:8000/ui

---

## Example Questions to Test

| Question | Expected Response |
|---|---|
| How do I reset my password? | Explains Settings > Security > Reset Password |
| What is your refund policy? | 14 days, unused product, 5-7 business days |
| How do I enable two factor authentication? | Explains Settings > Security > 2FA |
| What is the capital of France? | I do not have enough information |

---

## Knowledge Base

The assistant answers questions based on these topics:
- Password Reset
- Account Deletion
- Billing and Subscriptions
- Two-Factor Authentication
- Data Export
- Contacting Support
- Changing Email Address
- Refund Policy

---

## Dependencies
fastapi
uvicorn
python-dotenv
sentence-transformers
scikit-learn
groq

## Author

- Name: Surya Vikas Indraganti
- GitHub: https://github.com/suryavikasindraganti
