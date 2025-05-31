from transformers import pipeline

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

def generate_summary(text: str) -> str:
    return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
