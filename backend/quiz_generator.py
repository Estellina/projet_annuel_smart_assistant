from transformers import pipeline

quiz_gen = pipeline("text2text-generation", model="t5-base")

def generate_quiz(text: str) -> str:
    prompt = f"Génère 2 questions QCM à partir du texte suivant : {text}"
    return quiz_gen(prompt, max_length=200)[0]['generated_text']
