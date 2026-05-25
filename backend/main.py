from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from groq import Groq

# Load environment variables
load_dotenv()

# Configure Groq
client_groq = Groq(api_key=os.getenv("LLM_API_KEY"))

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
docs_path = os.path.join(os.path.dirname(__file__), "docs.json")
with open(docs_path, "r") as f:
    documents = json.load(f)

# Generate embeddings for all documents at startup
doc_embeddings = []
for doc in documents:
    embedding = embedding_model.encode(doc["content"])
    doc_embeddings.append({
        "title": doc["title"],
        "content": doc["content"],
        "embedding": embedding
    })

print(f"Loaded {len(doc_embeddings)} documents.")

# Frontend path
frontend_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
print(f"Frontend path: {frontend_path}")
print(f"Frontend exists: {os.path.exists(frontend_path)}")

# Store session history
conversation_history = {}


# Request model
class ChatRequest(BaseModel):
    sessionId: str
    message: str


# ── Routes (must be defined BEFORE app.mount) ──────────────────────────────

@app.get("/")
def home():
    return {"message": "RAG Assistant Running"}


@app.get("/debug")
def debug():
    return {
        "frontend_path": frontend_path,
        "frontend_exists": os.path.exists(frontend_path),
        "files": os.listdir(frontend_path) if os.path.exists(frontend_path) else []
    }


@app.get("/ui", response_class=HTMLResponse)
def serve_frontend():
    index_file = os.path.join(frontend_path, "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(frontend_path, "style.css"), media_type="text/css")


@app.get("/script.js")
def serve_js():
    return FileResponse(os.path.join(frontend_path, "script.js"), media_type="application/javascript")


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        if not request.sessionId:
            raise HTTPException(status_code=400, detail="sessionId is required.")

        user_message = request.message
        session_id = request.sessionId

        # Generate query embedding
        query_embedding = embedding_model.encode(user_message)

        # Similarity search
        scores = []
        for doc in doc_embeddings:
            score = cosine_similarity(
                [query_embedding],
                [doc["embedding"]]
            )[0][0]
            scores.append({
                "title": doc["title"],
                "content": doc["content"],
                "score": float(score)
            })

        # Sort and take top 3
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = scores[:3]

        # Threshold check
        if top_chunks[0]["score"] < 0.3:
            return {
                "reply": "I do not have enough information to answer that question.",
                "retrievedChunks": 0
            }

        # Build context
        relevant_chunks = [c["content"] for c in top_chunks if c["score"] >= 0.3]
        context = "\n\n".join(relevant_chunks)

        # Conversation history
        history = conversation_history.get(session_id, [])
        history_text = ""
        for item in history:
            history_text += f"User: {item['user']}\nAssistant: {item['assistant']}\n"

        # Build RAG prompt
        prompt = f"""You are a helpful customer support assistant.
Answer the user question using ONLY the provided context below.
If the context does not contain enough information, say: "I do not have enough information to answer that question."

Context:
{context}

Conversation History:
{history_text}

Question:
{user_message}

Answer:"""

        # Call Groq
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

        # Save history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({
            "user": user_message,
            "assistant": answer
        })
        conversation_history[session_id] = conversation_history[session_id][-5:]

        return {
            "reply": answer,
            "retrievedChunks": len(relevant_chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}