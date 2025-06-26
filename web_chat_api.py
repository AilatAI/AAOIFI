import os
import re
from flask import Flask, request
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone

# ─── 1. Конфиг ────────────────────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV       = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX     = "aaoifi-standards"

EMBED_MODEL        = "text-embedding-ada-002"
CHAT_MODEL         = "gpt-3.5-turbo"
TOP_K              = 5

# ─── 2. Инициализация клиентов ─────────────────────────────────────────────────
openai = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)

app = Flask(__name__)
CORS(app, resources={
    r"/chat": {
      "origins": [
        "https://www.ailat.kz",
        "https://ailat.kz"
      ]
    }
})

# ─── 3. Основная логика ─────────────────────────────────────────────────────────
def answer_question(question: str) -> str:
    # 1) Embedding + Pinecone lookup
    resp = openai.embeddings.create(model=EMBED_MODEL, input=question)
    q_emb = resp.data[0].embedding

    qr = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
    contexts = [
        f"{m.metadata.get('section_title','')} (Std {m.metadata.get('standard_number','')}):\n{m.metadata.get('chunk_text','')}"
        for m in qr.matches
    ]

    # 2) One‐shot prompt: detect → translate → retrieve → answer → re‐translate
    system = {
        "role": "system",
        "content": (
            "You are an AAOIFI standards expert. When you receive a question, do the following:\n"
            "1. Detect the question’s original language (e.g. ru, kk, en, ar).\n"
            "2. If it’s not in English, translate it into English for internal use, preserving ALL technical terms exactly.\n"
            "3. Using only the provided excerpts, compose a coherent and detailed answer "
            "that explains and synthesizes the relevant sections. "
            "4. After each fact or quotation, append a clear, human-readable citation in full words— "
            "for example: (AAOIFI Standard 35, Introduction, Paragraph 3). "
            "5. If the original question was not in English, translate your English answer back into the "
            "original language, again preserving ALL technical terms — and ensure the entire output is "
            "in that language with no remaining English text (aside from citation IDs themselves).\n"
            "6. If the information is incomplete, clearly state what is missing. "
            "7. Maintain all technical terms in their original form."
            "8. Return ONLY the final answer in the user’s language—do not reveal any internal steps."
        )
    }
    user = {
        "role": "user",
        "content": (
            "Here are the relevant AAOIFI excerpts:\n\n"
            + "\n---\n".join(contexts)
            + f"\n\nQuestion: {question}"
        )
    }

    chat_resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.2,
        max_tokens=600
    )

    return chat_resp.choices[0].message.content.strip()

# ─── 4. Flask‐эндпоинт ─────────────────────────────────────────────────────────
@app.route("/chat", methods=["GET"])
def chat():
    question = request.args.get("question", "").strip()
    if not question:
        return "No question provided", 400
    try:
        answer = answer_question(question)
        return answer
    except Exception as e:
        return f"Error: {str(e)}", 500

# ─── 5. Запуск ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
