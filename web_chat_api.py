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
    # 4.1) Детерктируем язык
    lang_code = detect_language(question)
    lang_name = LANG_NAMES.get(lang_code, 'English')

    # 4.2) Переводим на английский для Pinecone
    if lang_code != 'en':
        tran = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content":
                    "Translate the following into English, preserving all AAOIFI/Islamic-finance terms exactly."},
                {"role": "user",   "content": question}
            ],
            temperature=0.1
        )
        eng_question = tran.choices[0].message.content.strip()
    else:
        eng_question = question

    # 4.3) Делаем эмбединг и ищем по индексу
    resp  = openai.embeddings.create(model=EMBED_MODEL, input=eng_question)
    q_emb = resp.data[0].embedding
    qr    = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

    contexts = []
    for match in qr.matches:
        md    = match.metadata
        txt   = md.get("chunk_text", "")
        title = md.get("section_title", "")
        num   = md.get("standard_number", "")
        contexts.append(f"{title} (Std {num}):\n{txt}")

    # 4.4) Один чат-вызов: внутренняя трансляция → ответ по английским фрагментам → перевод назад
    system_prompt = f"""You are an AAOIFI standards expert. The user’s question is in {lang_name} ({lang_code}).
1. Translate the question _internally_ into English, preserving all AAOIFI/Islamic-finance terms exactly.
2. Using ONLY the provided English AAOIFI excerpts, compose a detailed answer in English with citations like “(AAOIFI Standard 35, Introduction, Paragraph 3)”.
3. Translate that English answer back into {lang_name}, preserving meaning exactly and keeping all AAOIFI technical terms in English.

IMPORTANT: Your FINAL OUTPUT MUST BE 100% in {lang_name}, except for the AAOIFI terms (e.g. “murabaha”, “sukuk”, “Ijarah”) and the citations, which stay in English. Do NOT include any other English words or sentences.
"""

    user_prompt = (
        "Here are the relevant AAOIFI excerpts:\n\n"
        + "\n---\n".join(contexts)
        + f"\n\nQuestion: {question}\nAnswer:"
    )

    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
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
