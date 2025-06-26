import os
import re
from flask import Flask, request
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone

# ─── 1. Конфиг ────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX   = "aaoifi-standards"

EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL  = "gpt-3.5-turbo"
TOP_K       = 5

# ─── 2. Инициализация клиентов ─────────────────────────────────────────────────
openai = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": ["https://www.ailat.kz", "https://ailat.kz"]}})

# ─── 3. Языковая утилита ────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    if re.search(r"[ңғүұқәі]|[\u0500-\u052F]", text):
        return 'kk'
    if re.search(r"[\u0600-\u06FF]", text):
        return 'ar'
    if re.search(r"[\u0750-\u077F\uFB50-\uFDFF]", text):
        return 'ur'
    if re.search(r"[\u0400-\u04FF]", text):
        return 'ru'
    return 'en'

# ─── 3.1. Читаемые имена языков ─────────────────────────────────────────────────
LANG_NAMES = {
    'en': 'English',
    'ru': 'Russian',
    'kk': 'Kazakh',
    'ar': 'Arabic',
    'ur': 'Urdu'
}

# ─── 4. Логика answer_question ─────────────────────────────────────────────────
 def answer_question(question: str) -> str:
     # 4.1) Определяем язык
     lang_code = detect_language(question)
     lang_name = LANG_NAMES.get(lang_code, 'English')

     # 4.2) Переводим вопрос на английский для поиска (если нужно)
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

     # 4.3) Создаем embedding и ищем по Pinecone
     resp  = openai.embeddings.create(model=EMBED_MODEL, input=eng_question)
     q_emb = resp.data[0].embedding
     qr    = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

     contexts = [
         f"{m.metadata.get('section_title')} (Std {m.metadata.get('standard_number')}):\n{m.metadata.get('chunk_text')}"
         for m in qr.matches
     ]

-    # 4.4) Один чат-вызов: перевод → ответ → перевод назад
-    system_prompt = f"""..."""
-    user_prompt = (
-        "Here are the relevant AAOIFI excerpts:\n\n"
-        + "\n---\n".join(contexts)
-        + f"\n\nQuestion: {question}\nAnswer:"
-    )
+    # 4.4) Один чат-вызов: ответ на eng_question → перевод на lang_code
+    system_prompt = f"""
+You are an AAOIFI standards expert.
+Step 1: Read the question in English and use ONLY the provided English AAOIFI excerpts to compose a detailed answer in English with citations (e.g. “(AAOIFI Standard 35, Introduction, Paragraph 3)”).
+Step 2: Translate that entire English answer into {lang_name}, preserving ALL AAOIFI/Islamic-finance technical terms and citations in English.
+
+IMPORTANT: Your FINAL OUTPUT MUST BE 100% in {lang_name}. Do NOT include any other English words or sentences.
+"""
+
+    user_prompt = (
+        "Relevant AAOIFI excerpts:\n\n"
+        + "\n---\n".join(contexts)
+        + f"\n\nQuestion (in English): {eng_question}\nAnswer:"
+    )
 
     chat = openai.chat.completions.create(
         model=CHAT_MODEL,
         messages=[
             {"role": "system", "content": system_prompt},
-            {"role":"user",   "content": user_prompt}
+            {"role": "user",   "content": user_prompt}
         ],
         temperature=0,
         max_tokens=600
     )
 
     return chat.choices[0].message.content.strip()

# ─── 5. Flask‐эндпоинт ─────────────────────────────────────────────────────────
@app.route("/chat", methods=["GET"])
def chat_endpoint():
    question = request.args.get("question", "").strip()
    if not question:
        return "No question provided", 400
    try:
        return answer_question(question)
    except Exception as e:
        return f"Error: {e}", 500

# ─── 6. Запуск ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
