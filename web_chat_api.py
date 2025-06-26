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
    # 1) Определяем язык пользователя
    lang_code = detect_language(question)
    lang_name = LANG_NAMES.get(lang_code, "English")

    # 2) Создаём embedding и ищем в Pinecone
    resp = openai.embeddings.create(model=EMBED_MODEL, input=question)
    q_emb = resp.data[0].embedding
    qr    = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

    contexts = [
        f"{m.metadata.get('section_title','')} "
        f"(Std {m.metadata.get('standard_number','')}):\n"
        f"{m.metadata.get('chunk_text','')}"
        for m in qr.matches
    ]

    # 3) Улучшенный system prompt
    system_prompt = (
        f"You are an AAOIFI standards expert. The user is asking in {lang_name}.\n\n"
        "Follow these steps strictly:\n"
        "1. First understand the question's intent completely.\n"
        "2. Search for the most relevant information in the provided AAOIFI excerpts.\n"
        "3. If the question is not in English, process it internally in English but maintain all technical terms.\n"
        "4. Formulate a precise, accurate answer based ONLY on the provided excerpts.\n"
        "5. Present the answer in the user's original language ({lang_name}), keeping:\n"
        "   - All standard numbers (like 'AAOIFI Standard 12')\n"
        "   - All technical terms (like 'Murabaha', 'Sukuk')\n"
        "   - All citations (like 'Section 5.2')\n"
        "   EXACTLY as in English, without translation.\n"
        "6. Ensure the response is natural in {lang_name} while preserving untranslatable elements.\n\n"
        "Important:\n"
        "- Never invent information not present in the excerpts.\n"
        "- If unsure, say you don't know based on AAOIFI standards.\n"
        "- For Kazakh questions, pay special attention to proper terminology."
    )

    user_prompt = (
        "Relevant AAOIFI excerpts:\n\n"
        + "\n---\n".join(contexts)
        + f"\n\nUser's question ({lang_name}): {question}\n\n"
        "Provide the answer in {lang_name} following all instructions above:"
    )

    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800
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
