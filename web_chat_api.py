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
    
    # 1) Better translation to English with clarification
    if lang != 'en':
        tran = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system", "content": 
                    "You are a precise financial translator. Translate to English while: "
                    "1. Keeping all AAOIFI/Islamic finance terms in original English "
                    "2. Preserving exact legal/financial meaning "
                    "3. Clarify ambiguous terms"},
                {"role":"user", "content": question}
            ],
            temperature=0.0
        )
        eng_question = tran.choices[0].message.content.strip()
    else:
        eng_question = question

    # 2) More focused embedding search
    resp = openai.embeddings.create(model=EMBED_MODEL, input=eng_question)
    q_emb = resp.data[0].embedding

    # Add filter for relevant standards (money, debt, theft)
    # Using only valid Pinecone operators
    qr = index.query(
        vector=q_emb,
        top_k=TOP_K,
        include_metadata=True,
        filter={
            "$or": [
                {"standard_number": {"$in": ["2", "3", "5", "7"]}},  # Money/debt related standards
                # Removed invalid $contains filters
            ]
        }
    )

    # Additional filtering for relevant content in Python
    contexts = []
    money_related_keywords = ["money", "debt", "loan", "theft", "steal", "borrow"]
    
    for match in qr.matches:
        if match.score < 0.7:  # Skip low-confidence matches
            continue
            
        md = match.metadata
        section_title = md.get('section_title', '').lower()
        chunk_text = md.get('chunk_text', '').lower()
        
        # Check if any keyword exists in either title or text
        if any(keyword in section_title or keyword in chunk_text 
               for keyword in money_related_keywords):
            contexts.append(
                f"{md.get('section_title','')} (Std {md.get('standard_number','')}):\n"
                f"{md.get('chunk_text','')}"
            )

    # 3) Strict answer generation with verification
    system_content = (
        f"Respond in {lang} to this Islamic finance question using ONLY the AAOIFI excerpts below. "
        "Rules:\n"
        "1. If excerpts don't match the question, say: 'This specific case isn't covered in AAOIFI standards I have access to.'\n"
        "2. Keep all AAOIFI terms in original English (e.g. 'murabaha', 'qard')\n"
        "3. For money-related questions, only use Standards 2, 3, 5, 7\n"
        "4. Never invent answers - if unsure, say you don't know"
    )
    
    if contexts:
        system_content += "\n\nExcerpts:\n" + "\n---\n".join(contexts)
    
    system = {"role": "system", "content": system_content}
    user = {"role": "user", "content": question}
    
    chat = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system, user],
        temperature=0.1,  # Lower for more factual responses
        max_tokens=512
    )
    
    answer = chat.choices[0].message.content.strip()
    
    # 4) Post-answer verification and language-specific fallback
    not_covered_phrases = ["isn't covered", "not covered", "не покрывается", "жоқ"]
    if any(phrase in answer.lower() for phrase in not_covered_phrases):
        if lang == 'kk':
            return "Кешіріңіз, AAOIFI стандарттарында ақша ұрлау мен қарыз беру туралы нақты ақпарат жоқ."
        elif lang == 'ru':
            return "Извините, в стандартах AAOIFI нет конкретной информации о краже денег и их кредитовании."
        else:
            return "Sorry, the AAOIFI standards don't contain specific information about money theft and lending."
    
    return answer

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
