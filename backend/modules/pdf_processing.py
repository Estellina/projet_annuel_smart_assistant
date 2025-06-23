# backend/pipelines/pdf_processing.py
import io
import fasttext
from pdfplumber import open as pdf_open
from typing import Dict, List
import logging
import re
import torch
import numpy as np
from tqdm import tqdm

# Import internal modules
#from models.summarization.model import load_model as load_summarization_model
#from pipelines.translation import initialize_translation, translate_text
#from services import quiz_generation
#from utils.text_processing import preprocess_academic_text

from backend.model.summarization.model import load_model as load_summarization_model
from backend.modules.translation import initialize_translation, translate_text
from backend.modules import quiz_generation
from backend.model.utils.text_processing import preprocess_academic_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models instances
SUMMARIZATION_MODEL = None
SUMMARIZATION_TOKENIZER = None
LANG_DETECTOR = None
TRANSLATION_INITIALIZED = False


def initialize_models():
    """Load all required models at startup"""
    global SUMMARIZATION_MODEL, SUMMARIZATION_TOKENIZER, LANG_DETECTOR, TRANSLATION_INITIALIZED

    try:
        # 1. Load summarization models
        if SUMMARIZATION_MODEL is None:
            logger.info("Loading summarization models...")
            SUMMARIZATION_MODEL, SUMMARIZATION_TOKENIZER = load_summarization_model(
                "../backend/model/summarization/best_model.pth",
                "../backend/model/summarization"
            )
            logger.info("Summarization models loaded successfully")

        # 2. Load language detection models
        if LANG_DETECTOR is None:
            logger.info("Loading language detection models...")
            LANG_DETECTOR = fasttext.load_model('lid.176.bin')
            logger.info("Language detection models loaded successfully")

        # 3. Initialize translation
        if not TRANSLATION_INITIALIZED:
            logger.info("Initializing translation system...")
            initialize_translation()
            TRANSLATION_INITIALIZED = True
            logger.info("Translation system initialized")

    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise RuntimeError("Model initialization failed") from e


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF with layout preservation and academic element handling"""
    text = ""
    try:
        with pdf_open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                # Extract text while preserving layout
                page_text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=1)

                # Handle empty pages
                if page_text is None:
                    page_text = ""

                text += page_text + "\n\n"

                # Handle special academic elements
                if page_text.strip() == "":
                    # Check for images/tables that couldn't be extracted
                    if page.images:
                        text += "[IMAGE]\n\n"
                    if page.find_tables():
                        text += "[TABLE]\n\n"

        # Post-process extracted text
        text = re.sub(r'-\n', '', text)  # Remove hyphenated line breaks
        text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s{2,}', ' ', text)  # Reduce multiple spaces

        return text.strip()

    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise RuntimeError("PDF text extraction failed") from e


def detect_language(text: str) -> str:
    """Detect document language using FastText with fallback to English"""
    try:
        # Clean text for language detection
        clean_text = text.replace("\n", " ").replace("\t", " ").strip()
        if len(clean_text) < 10:
            return "en"  # Default to English if insufficient text

        # Predict language
        predictions = LANG_DETECTOR.predict(clean_text, k=1)
        lang_code = predictions[0][0].replace('__label__', '')

        # Validate language code
        return lang_code if lang_code in ['fr', 'en'] else 'en'

    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return 'en'  # Fallback to English


def split_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into meaningful chunks while preserving academic structure

    Strategy:
    1. First try to split by sections (if document has clear headings)
    2. Then split by paragraphs
    3. Finally split by sentence boundaries if needed
    """
    chunks = []

    # Attempt 1: Split by major academic headings
    heading_pattern = r'\n(\d+\.\d+\s+[^\n]+|\n[A-Z][A-Z0-9\s]+\n)'
    sections = re.split(heading_pattern, text)

    # If we found reasonable sections
    if len(sections) > 3:
        current_chunk = ""
        for i, section in enumerate(sections):
            # Every odd index is a heading (after first element)
            if i % 2 == 1:
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Start new chunk with heading
                current_chunk = section + "\n\n"
            else:
                current_chunk += section + "\n\n"

            # If chunk is large, split it
            if len(current_chunk) > max_chunk_size * 1.2:
                chunks.append(current_chunk[:max_chunk_size].strip())
                current_chunk = current_chunk[max_chunk_size:]

        if current_chunk:
            chunks.append(current_chunk.strip())

    # If section splitting didn't work, split by paragraphs
    if not chunks:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk) + len(para) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            current_chunk += para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

    # Final fallback: Split by sentence boundaries
    if not chunks or any(len(chunk) > max_chunk_size * 1.5 for chunk in chunks):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        current_chunk = ""

        for sent in sentences:
            if current_chunk and len(current_chunk) + len(sent) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            current_chunk += sent + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

    # Ensure no chunk is too large
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_chunk_size * 1.5:
            # Find a natural break point
            break_point = max(
                chunk.rfind('. ', 0, max_chunk_size),
                chunk.rfind('\n', 0, max_chunk_size),
                max_chunk_size
            )
            final_chunks.append(chunk[:break_point].strip())
            chunk = chunk[break_point:].strip()
        final_chunks.append(chunk)

    logger.info(f"Split document into {len(final_chunks)} chunks")
    return final_chunks


def extract_key_sentences(text: str, num_sentences: int = 5) -> str:
    """Simple TextRank implementation to extract key sentences"""
    try:
        # Split into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        if len(sentences) < 3:
            return text  # Return full text if not enough sentences

        # Calculate word frequencies
        word_freq = {}
        for sentence in sentences:
            for word in sentence.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            for word in sentence.split():
                if word in word_freq:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
            sentence_scores[sentence] /= len(sentence.split())

        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        return " ".join([s[0] for s in top_sentences])

    except Exception as e:
        logger.warning(f"Key sentence extraction failed: {str(e)}")
        return text


def nucleus_sampling(logits, top_p: float = 0.92, temperature: float = 0.85) -> int:
    """Nucleus sampling implementation for diverse generation"""
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens outside the nucleus
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()


def process_chunk(chunk: str) -> dict:
    """Process a single text chunk using the model's internal generation method"""
    try:
        # Prétraitement spécifique aux documents académiques
        processed_text = preprocess_academic_text(chunk)

        # ✅ Utilisation de la méthode interne de génération
        summary_text = SUMMARIZATION_MODEL.generate_high_quality_summary(
            tokenizer=SUMMARIZATION_TOKENIZER,
            text=processed_text,
            max_length=256,
            max_source_length=2048
        )

        # 🔄 Post-processing (optionnel ici car déjà fait dans le modèle)
        if summary_text and summary_text[-1] not in {'.', '!', '?'}:
            summary_text += '.'
        if summary_text and summary_text[0].islower():
            summary_text = summary_text[0].upper() + summary_text[1:]

        # Génération de quiz (inchangée)
        quiz = quiz_generation.generate_quiz(processed_text)

        return {
            "summary": summary_text,
            "quizzes": quiz
        }

    except Exception as e:
        logger.error(f"Chunk processing failed: {str(e)}")
        return {
            "summary": f"Summary generation error: {str(e)}",
            "quizzes": []
        }



def process_pdf(pdf_bytes: bytes) -> Dict:
    """Main PDF processing pipeline with academic-aware processing"""
    try:
        # Ensure models are loaded
        initialize_models()

        # Step 1: Extract text from PDF
        logger.info("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            return {
                "summary": "Error: No text could be extracted from the PDF",
                "quizzes": [],
                "language": "en"
            }

        # Step 2: Detect language
        logger.info("Detecting document language...")
        lang = detect_language(text)
        logger.info(f"Detected language: {lang}")

        # Step 3: Translate to English if needed
        working_text = text
        if lang == 'fr':
            logger.info("Translating document to English...")
            working_text = translate_text(text, 'fr', 'en')

        # Step 4: Split into manageable chunks
        logger.info("Splitting document into chunks...")
        chunks = split_into_chunks(working_text, max_chunk_size=1024)

        # Step 5: Process each chunk with academic-aware summarization
        logger.info(f"Processing {len(chunks)} document chunks...")
        summaries = []
        quizzes = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            result = process_chunk(chunk)
            summaries.append(result["summary"])
            quizzes.extend(result["quizzes"])

        # Combine summaries
        full_summary = "\n\n".join(summaries)

        # Step 6: Translate results back to French if needed
        if lang == 'fr':
            logger.info("Translating results back to French...")
            full_summary = translate_text(full_summary, 'en', 'fr')
            for quiz in quizzes:
                quiz["question"] = translate_text(quiz["question"], 'en', 'fr')
                quiz["options"] = [translate_text(opt, 'en', 'fr') for opt in quiz["options"]]

        logger.info("PDF processing completed successfully")
        return {
            "summary": full_summary,
            "quizzes": quizzes,
            "language": lang
        }

    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        return {
            "summary": f"Processing error: {str(e)}",
            "quizzes": [],
            "language": "en"
        }