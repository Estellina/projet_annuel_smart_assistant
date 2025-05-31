from fastapi import FastAPI, UploadFile, File
import shutil, os
from transcription import transcribe
from summarizer import generate_summary
from rag_pipeline import retrieve_relevant
from quiz_generator import generate_quiz
from save_json import save_results

app = FastAPI()

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        transcript = transcribe(temp_path)
        summary = generate_summary(transcript)
        context = transcript.split(". ")  # chunks naïfs
        rag_answer = retrieve_relevant(context, "De quoi parle ce cours ?")
        quiz = generate_quiz(summary)

        results = {
            "transcription": transcript,
            "summary": summary,
            "RAG": rag_answer,
            "quiz": quiz
        }

        save_results(results)
        return results
    finally:
        os.remove(temp_path)
