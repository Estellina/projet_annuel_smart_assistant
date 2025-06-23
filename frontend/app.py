from flask import Flask, render_template, request, redirect, url_for, session
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.modules.transcription import transcribe
from backend.modules.summarizer import generate_summary
from backend.modules.quiz_generator import generate_quiz
#from backend.modules.pdf_extractor import extract_with_grobid, extract_with_fitz, parse_grobid_tei

from backend.modules.rag_pipeline import retrieve_relevant
from backend.modules.save_json import save_results
from flask import send_from_directory
from werkzeug.utils import safe_join
from backend.modules.pdf_processing import process_pdf, initialize_models

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Dossier des uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# INITIALISATION DU MODÈLE
initialize_models()


@app.route("/", methods=["GET"])
def index():
    """Page d'accueil."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload du fichier, traitement du PDF, génération du résumé et des quiz."""
    file = request.files.get("file")
    question = request.form.get("question", "")

    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # ⚡️ Appel DIRECT à process_pdf
        with open(path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            result = process_pdf(pdf_bytes)

        summary = result.get("summary", "")
        quiz = result.get("quizzes", [])

        if question:
            chunks = [summary[i:i + 300] for i in range(0, len(summary), 300)]
            answer = retrieve_relevant(chunks, question)  # À définir si nécessaire
        else:
            answer = ""

        # Sauvegarde des résultats
        save_results({
            "filename": file.filename,
            "summary": summary,
            "question": question,
            "answer": answer,
            "quiz": quiz
        }, output_dir="shared/exports")

        return render_template("result.html",
                               summary=summary,
                               answer=answer,
                               quiz=quiz,
                               question=question,
                               filename=file.filename)

    return redirect(url_for("index"))


@app.route("/generate_quiz", methods=["POST"])
def generate_quiz_route():
    """Génération du quiz à partir du résumé."""
    summary = request.form.get("summary", "")
    question = request.form.get("question", "")
    quiz = []  # Ici, tu peux définir ta propre méthode de génération du quiz si nécessaire
    answer = retrieve_relevant([summary], question) if question else ""
    return render_template("quiz.html",
                           summary=summary,
                           quiz=quiz,
                           answer=answer,
                           question=question)


@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    """Récupération des réponses du quiz soumis."""
    submitted_answers = {
        key: value for key, value in request.form.items() if key.startswith("question_")
    }
    return render_template("qcm.html",
                           summary=session.get("summary", ""),
                           quiz=session.get("quiz", []),
                           answers=submitted_answers)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Accès direct aux fichiers uploadés."""
    file_path = safe_join(UPLOAD_FOLDER, filename)
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(file_path))

@app.route("/ask_question", methods=["POST"])
def ask_question():
    """Répond à une question à propos du document déjà traité."""
    filename = request.form.get("filename")
    question = request.form.get("question", "")
    if not filename or not question:
        return redirect(url_for("index"))

    # Chemin du fichier
    path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(path):
        return redirect(url_for("index"))

    #Relire le fichier pour effectuer le RAG
    with open(path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
        result = process_pdf(pdf_bytes)

    summary = result.get("summary", "")
    chunks = [summary[i:i + 300] for i in range(0, len(summary), 300)]
    answer = retrieve_relevant(chunks, question)

    return render_template("result.html",
                           summary=summary,
                           answer=answer,
                           quiz=result.get("quizzes", []),
                           question=question,
                           filename=filename)


if __name__ == "__main__":
    app.run(debug=True)