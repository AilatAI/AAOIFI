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
    # 2) Embedding + Pinecone lookup
    resp = openai.embeddings.create(model=EMBED_MODEL, input=question)
    q_emb = resp.data[0].embedding

    qr = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
    contexts = [
        f"{m.metadata.get('section_title','')} (Std {m.metadata.get('standard_number','')}):\n{m.metadata.get('chunk_text','')}"
        for m in qr.matches
    ]

    # 3) One‐shot prompt: detect → translate → retrieve → answer → re‐translate
    system = {
        "role": "system",
        "content": (
            "You are an AAOIFI standards expert. When you receive a question, do the following steps:\n"
            "1. Detect the question’s original language (e.g. ru, kk, en, ar).\n"
            "2. If it is not in English, translate it into English for internal processing, preserving ALL technical terms exactly.\n"
            "3. Perform a vector search on the Pinecone index “aaoifi-standards” and retrieve the top 5 most relevant chunks.\n"
            "4. Synthesize a coherent, detailed answer in English, appending inline citations like (AAOIFI Std X Sec Y ¶ Z) as markdown links.\n"
            "5. Finally, if the original question was not in English, translate your English answer back into the original language, again preserving ALL technical terms.\n"
            "6. Return ONLY the final answer in the user’s language—do not show any internal steps or translations."
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

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()

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
