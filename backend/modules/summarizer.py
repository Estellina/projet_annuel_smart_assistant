"""
from transformers import pipeline
#le model t5-base est entrainé sur des textes en anglais il comprend mal le français
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

def generate_summary(text: str) -> str:
    return summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']


#modele multilingue et compatible avec HuggingFace
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Charger explicitement le tokenizer lent
from transformers import pipeline

def chunk_text(text, max_tokens=1024):
    # Simple splitting method by paragraph (can be improved with tokenizer)
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

summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum",
    framework="pt", # <-- force l'utilisation de PyTorch
    device=-1, # <-- CPU uniquement, ou device=0 pour GPU
    model_kwargs={"from_pt": True} # <-- important ici
)

def generate_summary(text: str) -> str:
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return "\n".join(summaries)
"""
from transformers import pipeline

summarizer = None

def generate_summary(text, max_length=150):
    global summarizer
    if summarizer is None:
        summarizer = pipeline(
            "summarization",
            model="moussaKam/barthez",
            tokenizer="moussaKam/barthez",
            framework="pt",
            device=-1  # CPU
        )
    result = summarizer(text, max_length=max_length, truncation=True)
    return result[0]["summary_text"]
