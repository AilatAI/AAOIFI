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

EMBED_MODEL        = "text-embedding-3-small"
CHAT_MODEL         = "gpt-3.5-turbo"
TOP_K              = 15

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

    # 2) Предобработка короткого запроса "standard X"
    q = question.strip()
    m = re.match(r"^standard\s+(\d+)$", q, flags=re.IGNORECASE)
    if m:
        num = m.group(1)
        # развернем в более описательный запрос
        q = f"What is AAOIFI Standard {num} about?"

    # 3) Создаём embedding и ищем чуть больше, чтобы поймать редкие фрагменты
    resp = openai.embeddings.create(model=EMBED_MODEL, input=q)
    q_emb = resp.data[0].embedding
    qr    = index.query(vector=q_emb, top_k=15, include_metadata=True)

    # 4) Фильтруем для конкретного стандарта, если он упомянут
    std_matches = []
    if m:
        for match in qr.matches:
            if match.metadata.get("standard_number") == num:
                std_matches.append(match)
    if std_matches:
        matches = std_matches
    else:
        matches = qr.matches[:5]

    # 5) Если ничего не найдено — сразу отказываем
    if not matches:
        return "This isn't covered in AAOIFI standards"

    # 6) Строим контекст с полным заголовком
    contexts = []
    for m in matches:
        contexts.append(
            f"AAOIFI Standard {m.metadata['standard_number']} – {m.metadata.get('standard_name','')}\n"
            f"Section {m.metadata.get('section_number','')} ({m.metadata.get('section_title','')}):\n"
            f"{m.metadata.get('chunk_text','')}"
        )
    excerpts = "\n\n---\n\n".join(contexts)

    contexts = [
        f"{m.metadata.get('section_title','')} "
        f"(Std {m.metadata.get('standard_number','')}):\n"
        f"{m.metadata.get('chunk_text','')}"
        for m in qr.matches
    ]

    # 3) Улучшенный system prompt с явным контролем языка
    system_prompt = f"""
You are an AAOIFI standards expert. Follow these rules strictly:

1. LANGUAGE HANDLING:
   - The user's question is in {lang_name}
   - You MUST respond in {lang_name} exclusively
   - Never mix languages in your response
   - Preserve these elements in English exactly:
     * Standard numbers (e.g., "AAOIFI Standard 12")
     * Technical terms (e.g., "Murabaha", "Sukuk")
     * Citations (e.g., "Section 5.2", "Paragraph 3.1.4")

2. CONTENT REQUIREMENTS:
   - Use ONLY the provided AAOIFI excerpts below
   - If the answer isn't in the excerpts, say "This isn't covered in AAOIFI standards"
   - Be precise and factual
   - Maintain professional tone

3. FOR KAZAKH QUESTIONS:
   - Pay special attention to Islamic finance terminology
   - Use Kazakh grammar properly around English terms
   - Example: "AAOIFI Standard 12 бойынша Murabaha операциялары..."

Relevant excerpts:
{'\n---\n'.join(contexts)}
"""

    user_prompt = f"""
Question in {lang_name}: {question}

Instructions:
- Respond in {lang_name} only
- Keep technical terms in English
- Base answer strictly on AAOIFI standards above
- Format citations as: (Standard X, Section Y)
"""

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
