# backend/services/quiz_generation.py
import random


def generate_quiz(text: str, num_questions: int = 3) -> list:
    """
    Placeholder quiz generator - replace with your actual models
    For now, returns sample questions based on text length
    """
    # This is a placeholder implementation
    # In production, replace with your quiz generation models

    quizzes = []
    keywords = ["méthode", "résultat", "conclusion", "théorie", "modèle"]

    for i in range(num_questions):
        # Select a random keyword
        keyword = random.choice(keywords)

        # Generate question
        question = f"Quelle est la principale {keyword} présentée dans le document?"

        # Generate options
        options = [
            f"Option A concernant {keyword}",
            f"Option B concernant {keyword}",
            f"Option C concernant {keyword}",
            f"Option D concernant {keyword}"
        ]

        # Set correct answer (random for now)
        correct_index = random.randint(0, 3)

        quizzes.append({
            "question": question,
            "options": options,
            "answer": correct_index
        })

    return quizzes