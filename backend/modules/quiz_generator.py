from transformers import pipeline
import re

def chunk_text(text, max_tokens=1024):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    current_len = 0
    for para in paragraphs:
        if len(para) + current_len <= max_tokens:
            current_chunk += para + "\n"
            current_len += len(para)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
            current_len = len(para)
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



def generate_summary(text: str) -> str:
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(result[0]['summary_text'])
    return "\n".join(summaries)

# Générateur de texte
quiz_generator = pipeline(
    "text-generation",
    model="openai-community/gpt2-medium",
    device=-1
)

def parse_quiz_output(output: str):
    pattern = r"Question: (.*?)\na\) (.*?)\nb\) (.*?)\nc\) (.*?)\nd\) (.*?)\nR[eé]ponse correcte: (.*?)\n?"
    matches = re.findall(pattern, output, re.DOTALL)
    questions = []
    for match in matches:
        question, a, b, c, d, correct = match
        questions.append({
            "question": question.strip(),
            "propositions": [a.strip(), b.strip(), c.strip(), d.strip()],
            "reponse_correcte": correct.strip()
        })
    return questions

def generate_quiz(summary_text: str, num_questions: int = 10):
    if len(summary_text.strip()) < 30:
        return [{
            "question": "Texte trop court pour générer un QCM.",
            "propositions": [],
            "reponse_correcte": ""
        }]

    quiz_list = []
    attempts = 0
    max_attempts = num_questions * 3

    while len(quiz_list) < num_questions and attempts < max_attempts:
        prompt = f"""
        Génère une question QCM en français à partir du texte suivant. Propose une seule bonne réponse parmi 4. Formate la sortie comme suit :

        Question: ...
        a) ...
        b) ...
        c) ...
        d) ...
        Réponse correcte: ...

        Texte: {summary_text}
        """
        result = quiz_generator(prompt, max_new_tokens=250, do_sample=False)[0]['generated_text']
        parsed = parse_quiz_output(result)
        if parsed:
            for q in parsed:
                if q not in quiz_list and len(quiz_list) < num_questions:
                    quiz_list.append(q)
        attempts += 1

    return quiz_list if quiz_list else [{
        "question": "Aucune question valide générée.",
        "propositions": [],
        "reponse_correcte": ""
    }]
