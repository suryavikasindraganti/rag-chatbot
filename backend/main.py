from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("LLM_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

with open("docs.json", "r") as f:
    documents = json.load(f)

doc_embeddings = []
for doc in documents:
    embedding = embedding_model.encode(doc["content"])
    doc_embeddings.append({
        "title": doc["title"],
        "content": doc["content"],
        "embedding": embedding
    })

print(f"Loaded {len(doc_embeddings)} documents.")

conversation_history = {}

class ChatRequest(BaseModel):
    sessionId: str
    message: str

@app.get("/")
def home():
    return {"message": "RAG Assistant Running"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        user_message = request.message
        session_id = request.sessionId

        # Generate query embedding
        query_embedding = embedding_model.encode(user_message)

        # Similarity search against all docs
        scores = []
        for doc in doc_embeddings:
            score = cosine_similarity([query_embedding], [doc["embedding"]])[0][0]
            scores.append({"content": doc["content"], "title": doc["title"], "score": float(score)})

        # Sort and take top 3
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = scores[:3]

        # Threshold check
        if top_chunks[0]["score"] < 0.3:
            return {"reply": "I do not have enough information to answer that.", "retrievedChunks": 0}

        # Filter chunks above threshold
        relevant_chunks = [c["content"] for c in top_chunks if c["score"] >= 0.3]
        context = "\n\n".join(relevant_chunks)

        # Build conversation history text
        history = conversation_history.get(session_id, [])
        history_text = ""
        for item in history:
            history_text += f"User: {item['user']}\nAssistant: {item['assistant']}\n"

        # Build prompt
        prompt = f"""You are a helpful AI assistant.
Answer the user's question using ONLY the provided context below.
If the context does not contain the answer, say: "I do not have enough information to answer that."

Context:
{context}

Conversation History:
{history_text}

Question:
{user_message}

Answer:"""

        # ✅ Actually call Gemini
        response = model_gemini.generate_content(prompt)
        answer = response.text

        # Store history (keep last 5)
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({"user": user_message, "assistant": answer})
        conversation_history[session_id] = conversation_history[session_id][-5:]

        return {"reply": answer, "retrievedChunks": len(relevant_chunks)}

    except Exception as e:
        return {"error": str(e)}