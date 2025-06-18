import re


def preprocess_academic_text(text: str) -> str:
    """Clean academic text before processing"""
    # Handle citations
    text = re.sub(r'\\cite\{.*?\}', '[CITE]', text)
    text = re.sub(r'\\citep\{.*?\}', '[CITE]', text)
    text = re.sub(r'\\citet\{.*?\}', '[CITE]', text)

    # Handle math
    text = re.sub(r'\$.*?\$', '[MATH]', text)

    # Handle equations
    text = re.sub(
        r'\\begin\{equation\}.*?\\end\{equation\}',
        '[EQUATION]',
        text,
        flags=re.DOTALL
    )

    return text