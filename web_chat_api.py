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

# ─── 3. Языковая утилита ────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    if re.search(r"[ңғүұқәі]", text.lower()):
        return 'kk'
    elif re.search(r"[\u0500-\u052F]", text):
        return 'kk'
    elif re.search(r"[\u0600-\u06FF]", text):
        return 'ar'
    elif re.search(r"[\u0750-\u077F\uFB50-\uFDFF]", text):
        return 'ur'
    elif re.search(r"[\u0400-\u04FF]", text):
        return 'ru'
    else:
        return 'en'

# ─── 3.1. Человеко-читаемые названия языков ────────────────────────────────────
LANG_NAMES = {
    'en': 'English',
    'ru': 'Russian',
    'kk': 'Kazakh',
    'ar': 'Arabic',
    'ur': 'Urdu'
}

# ─── 4. Логика вашего answer_question ──────────────────────────────────────────
def answer_question(question: str) -> str:
    lang_code = detect_language(question)
    lang_name = LANG_NAMES.get(lang_code, 'English')

    # 1) Получаем embedding на английском
    #    (пусть GPT сам переведёт внутрь системы)
    resp = openai.embeddings.create(model=EMBED_MODEL, input=question)
    q_emb = resp.data[0].embedding
    qr = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

    contexts = []
    for m in qr.matches:
        md    = m.metadata
        txt   = md.get("chunk_text", "")
        title = md.get("section_title", "")
        num   = md.get("standard_number", "")
        contexts.append(f"{title} (Std {num}):\n{txt}")

    # 2) Один вызов: перевод вопроса → поиск ответа → перевод ответа назад
    system = {
        "role": "system",
        "content": (
            f"You are an AAOIFI standards expert. The user’s question is in {lang_name}.\n"
            "Step 1: Translate the question into English for your own processing, preserving ALL AAOIFI/Islamic-finance terms.\n"
            "Step 2: Using ONLY the provided English excerpts, compose a coherent, detailed answer in English, "
            "with citations like (AAOIFI Standard 35, Introduction, Paragraph X).\n"
            f"Step 3: Translate that English answer back into {lang_name}, keeping all technical terms in English.\n"
            "Output ONLY the final answer in the user’s original language."
        )
    }

    user = {
        "role": "user",
        "content": (
            "Here are the relevant AAOIFI excerpts:\n\n"
            + "\n---\n".join(contexts)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
    }

    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.2,
        max_tokens=512
    )

    return chat.choices[0].message.content.strip()

# ─── 5. Flask‐эндпоинт ─────────────────────────────────────────────────────────
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

# ─── 6. Запуск ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
