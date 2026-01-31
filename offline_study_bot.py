from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests, json

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatIP(384)
full_text = ""
chunks = []

def ask_local_llama(prompt: str):
    try:
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        res = requests.post(OLLAMA_URL, json=payload, timeout=300)
        return json.loads(res.text)["response"]
    except Exception as e:
        return f"⚠️ Local AI Error: {e}\n❗ Run:  ollama run llama3"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global full_text, chunks, index
    pdf = request.files["file"]
    reader = PdfReader(pdf)
    text = []

    for p in reader.pages:
        t = p.extract_text()
        if t and t.strip():
            text.append(t)

    if not text:
        return jsonify({"error": "No readable text found"}), 400

    full_text = "\n".join(text)
    chunks = [c.strip() for c in full_text.split(".") if len(c.strip()) > 5]

    index = faiss.IndexFlatIP(384)
    vectors = model.encode(chunks)
    index.add(np.array(vectors))

    return jsonify({"msg": "PDF Loaded Successfully!"})

@app.route("/summary")
def summary():
    if not full_text: return "⚠️ Upload PDF first."
    return ask_local_llama(f"Summarize in 20 bullet points:\n{full_text[:3000]}")

@app.route("/mcq")
def mcq():
    if not full_text: return "⚠️ Upload PDF first."
    return ask_local_llama("Generate 20 MCQs (4 options + answer) based ONLY on:\n" + full_text[:3000])

@app.route("/exam")
def exam_questions():
    if not full_text: return "⚠️ Upload PDF first."
    return ask_local_llama(f"Generate 10 long + 10 short questions from:\n{full_text[:6000]}")

@app.route("/ask", methods=["POST"])
def ask():
    q = request.json.get("q")
    vec = model.encode([q])
    _, ids = index.search(np.array(vec), 5)
    context = " ".join(chunks[i] for i in ids[0])
    return ask_local_llama(f"Answer using ONLY this:\n{context}\n\nQ: {q}")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
