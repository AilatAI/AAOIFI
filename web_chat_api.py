import os
import re
from flask import Flask, request
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone

# ─── 1. Configuration ─────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "aaoifi-standards")

EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
TOP_K            = int(os.getenv("TOP_K", "15"))

# ─── 2. Initialize clients ─────────────────────────────────────────────────
openai = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index  = pc.Index(PINECONE_INDEX)

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": ["https://www.ailat.kz","https://ailat.kz"]}})

# ─── 3. Language detection ──────────────────────────────────────────────────
def detect_language(text: str) -> str:
    if re.search(r"[ңғүұқәі]", text.lower()): return 'kk'
    if re.search(r"[\u0500-\u052F]", text):    return 'kk'
    if re.search(r"[\u0600-\u06FF]", text):    return 'ar'
    if re.search(r"[\u0750-\u077F\uFB50-\uFDFF]", text): return 'ur'
    if re.search(r"[\u0400-\u04FF]", text):    return 'ru'
    return 'en'

LANG_NAMES = {'en':'English','ru':'Russian','kk':'Kazakh','ar':'Arabic','ur':'Urdu'}

# ─── 4. Answer logic ───────────────────────────────────────────────────────
def answer_question(question: str) -> str:
    # 1) detect language
    lang_code = detect_language(question)
    lang_name = LANG_NAMES.get(lang_code, 'English')

    # 2) expand short "standard X" queries
    q_original = question.strip()
    m = re.match(r"^standard\s+(\d+)$", q_original, flags=re.IGNORECASE)
    if m:
        num = m.group(1)
        # build descriptive query
        q = f"What is AAOIFI Standard {num} about?"
    else:
        q = q_original
        num = None

    # 3) embedding + vector search
    resp = openai.embeddings.create(model=EMBED_MODEL, input=q)
    q_emb = resp.data[0].embedding
    qr    = index.query(vector=q_emb, top_k=TOP_K, include_metadata=True)

    # 4) optionally filter for specific standard
    if num:
        std_matches = [m for m in qr.matches if m.metadata.get('standard_number') == num]
        matches = std_matches or qr.matches[:5]
    else:
        matches = qr.matches[:5]

    # 5) no matches → fallback
    if not matches:
        return "This isn't covered in AAOIFI standards"

    # 6) build rich contexts
    contexts = []
    for m in matches:
        contexts.append(
            f"AAOIFI Standard {m.metadata.get('standard_number')} – {m.metadata.get('standard_name','')}\n"
            f"Section {m.metadata.get('section_number')} ({m.metadata.get('section_title','')}):\n"
            f"{m.metadata.get('chunk_text','')}"
        )
    excerpts = "\n\n---\n\n".join(contexts)

    # 7) system + user prompts
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

Relevant excerpts:
{excerpts}
"""

    user_prompt = f"Question: {q_original}"

    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.0,
        max_tokens=800
    )
    return chat.choices[0].message.content.strip()

# ─── 5. Flask endpoint ─────────────────────────────────────────────────────
@app.route('/chat', methods=['GET'])
def chat():
    question = request.args.get('question','').strip()
    if not question:
        return 'No question provided', 400
    try:
        return answer_question(question)
    except Exception as e:
        return f"Error: {e}", 500

# ─── 6. Run server ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
