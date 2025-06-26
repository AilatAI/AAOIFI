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

# ─── 4. Логика вашего answer_question ──────────────────────────────────────────
def answer_question(question: str) -> str:
    lang = detect_language(question)
    
    # 1) Translate to English if needed (for knowledge base search)
    if lang != 'en':
        tran = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system", "content":
                    "Translate the following into English, preserving meaning and style. "
                    "Keep all AAOIFI/Islamic finance terms unchanged. "
                    "If unclear, keep the original term in [brackets]."},
                {"role":"user", "content": question}
            ],
            temperature=0.1
        )
        eng_question = tran.choices[0].message.content.strip()
    else:
        eng_question = question

    # 2) Embedding + Pinecone (unchanged)
    resp = openai.embeddings.create(model=EMBED_MODEL, input=eng_question)
    q_emb = resp.data[0].embedding

    qr = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)
    contexts = []
    for match in qr.matches:
        md    = match.metadata
        txt   = md.get("chunk_text","")
        title = md.get("section_title","")
        num   = md.get("standard_number","")
        contexts.append(f"{title} (Std {num}):\n{txt}")

    # 3) Generate answer in the original language
    system = {
        "role":"system",
        "content":(
            f"You are an AAOIFI standards expert. Respond in {lang} using ONLY these excerpts:\n"
            "1. Keep all AAOIFI terms in original English (e.g., 'murabaha', 'sukuk')\n"
            "2. For citations use format: (AAOIFI Standard 35, Introduction)\n"
            "3. If question isn't covered, say: \"This specific case isn't covered in available AAOIFI standards\"\n"
            "4. Maintain professional Islamic finance terminology\n\n"
            "Available excerpts:\n" + "\n---\n".join(contexts)
        )
    }
    
    user = {
        "role":"user",
        "content": f"Question: {question}\n\nAnswer in {lang} using only provided AAOIFI standards:"
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
