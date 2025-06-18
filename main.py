from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipelines.pdf_processing import process_pdf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), mode: str = Form(...)):
    try:
        pdf_bytes = await file.read()
        result = process_pdf(pdf_bytes)

        print("📤 Résultat brut:", result)

        if mode == "summary":
            return {"result": result.get("summary", "")}

        elif mode == "quiz":
            return {"result": result.get("quizzes", [])}

        else:
            return JSONResponse(status_code=400, content={"error": "Mode invalide."})

    except Exception as e:
        print("❌ Erreur backend:", str(e))
        if mode == "summary":
            return {"result": f"Erreur : {str(e)}"}
        elif mode == "quiz":
            return {"result": []}
        else:
            return JSONResponse(status_code=500, content={"error": str(e)})
