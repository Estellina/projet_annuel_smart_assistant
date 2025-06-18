# backend/utils/academic_tokenizer.py
from tokenizers import ByteLevelBPETokenizer
import re


def load_tokenizer(vocab_path: str, merges_path: str) -> ByteLevelBPETokenizer:
    """
    Load custom tokenizer with academic special tokens
    """
    tokenizer = ByteLevelBPETokenizer(
        vocab_path,
        merges_path
    )

    # Add academic-specific tokens
    special_tokens = ["[CITE]", "[MATH]", "[EQUATION]"]
    tokenizer.add_special_tokens(special_tokens)

    return tokenizer


def preprocess_text(text: str) -> str:
    """
    Preprocess academic text by replacing special patterns with tokens
    """
    # Replace citations
    text = re.sub(r'\\cite\{.*?\}', '[CITE]', text)
    text = re.sub(r'\\citep\{.*?\}', '[CITE]', text)
    text = re.sub(r'\\citet\{.*?\}', '[CITE]', text)

    # Replace inline math
    text = re.sub(r'\$.*?\$', '[MATH]', text)

    # Replace equation blocks
    text = re.sub(
        r'\\begin\{equation\}.*?\\end\{equation\}',
        '[EQUATION]',
        text,
        flags=re.DOTALL
    )

    return text


def decode_special_tokens(text: str) -> str:
    """
    Convert academic tokens to human-readable format
    """
    replacements = {
        "[CITE]": "[citation]",
        "[MATH]": "[math]",
        "[EQUATION]": "[equation]"
    }

    for token, replacement in replacements.items():
        text = text.replace(token, replacement)

    return text