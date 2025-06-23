# backend/pipelines/translation.py
import argostranslate.package
import argostranslate.translate
import logging
import os
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global translation engines
TRANSLATOR_FR_EN = None
TRANSLATOR_EN_FR = None
TRANSLATION_INITIALIZED = False
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def initialize_translation():
    """
    Initialize translation system by:
    1. Updating package index
    2. Installing required language packages if missing
    3. Loading translation engines
    """
    global TRANSLATOR_FR_EN, TRANSLATOR_EN_FR, TRANSLATION_INITIALIZED

    if TRANSLATION_INITIALIZED:
        return

    logger.info("Initializing translation system...")

    try:
        # Step 1: Update package index
        logger.info("Updating translation package index...")
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        logger.info(f"Found {len(available_packages)} available translation packages")

        # Step 2: Install French to English if needed
        fr_en_package = next(
            (pkg for pkg in available_packages
             if pkg.from_code == 'fr' and pkg.to_code == 'en'),
            None
        )

        if fr_en_package:
            if not is_package_installed(fr_en_package):
                logger.info("Installing French to English translation package...")
                install_package_with_retry(fr_en_package)
            else:
                logger.info("French to English package already installed")
        else:
            logger.warning("French to English package not found in index")

        # Step 3: Install English to French if needed
        en_fr_package = next(
            (pkg for pkg in available_packages
             if pkg.from_code == 'en' and pkg.to_code == 'fr'),
            None
        )

        if en_fr_package:
            if not is_package_installed(en_fr_package):
                logger.info("Installing English to French translation package...")
                install_package_with_retry(en_fr_package)
            else:
                logger.info("English to French package already installed")
        else:
            logger.warning("English to French package not found in index")

            # Load translation engines
            logger.info("Loading translation engines...")
            available_languages = argostranslate.translate.get_installed_languages()

            from_lang_fr = next((lang for lang in available_languages if lang.code == "fr"), None)
            to_lang_en = next((lang for lang in available_languages if lang.code == "en"), None)

            from_lang_en = next((lang for lang in available_languages if lang.code == "en"), None)
            to_lang_fr = next((lang for lang in available_languages if lang.code == "fr"), None)

            if from_lang_fr and to_lang_en:
                TRANSLATOR_FR_EN = from_lang_fr.get_translation(to_lang_en)
            else:
                logger.error("No translation available for fr -> en.")

            if from_lang_en and to_lang_fr:
                TRANSLATOR_EN_FR = from_lang_en.get_translation(to_lang_fr)
            else:
                logger.error("No translation available for en -> fr.")

            TRANSLATION_INITIALIZED = True
            logger.info("Translation system initialized successfully.")

    except Exception as e:
        logger.error(f"Translation initialization failed: {str(e)}")
        raise RuntimeError("Translation system initialization failed") from e


def is_package_installed(package) -> bool:
    """Check if a translation package is already installed"""
    installed_packages = argostranslate.package.get_installed_packages()
    return any(
        pkg.from_code == package.from_code and
        pkg.to_code == package.to_code
        for pkg in installed_packages
    )


def install_package_with_retry(package, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Install a package with retry logic for network issues"""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Installation attempt {attempt}/{max_retries}")
            package.install()
            logger.info(f"Successfully installed {package.from_code} to {package.to_code} package")
            return True
        except Exception as e:
            logger.warning(f"Installation failed (attempt {attempt}): {str(e)}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    logger.error(f"Failed to install package after {max_retries} attempts")
    return False


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text between French and English with:
    - Error handling
    - Chunked translation for long texts
    - Fallback to original text on failure
    """
    global TRANSLATOR_FR_EN, TRANSLATOR_EN_FR

    # Return original text if no translation needed
    if source_lang == target_lang or not text.strip():
        return text

    try:
        # Validate translation direction
        if source_lang == 'fr' and target_lang == 'en' and TRANSLATOR_FR_EN:
            return chunked_translation(TRANSLATOR_FR_EN, text)
        elif source_lang == 'en' and target_lang == 'fr' and TRANSLATOR_EN_FR:
            return chunked_translation(TRANSLATOR_EN_FR, text)
        else:
            logger.warning(f"Unsupported translation direction: {source_lang} to {target_lang}")
            return text

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text


def chunked_translation(translator, text: str, max_chunk_size: int = 5000) -> str:
    """
    Split long text into chunks for translation to:
    1. Prevent resource exhaustion
    2. Improve reliability
    3. Maintain context within chunks
    """
    # Return immediately for short texts
    if len(text) <= max_chunk_size:
        return translator.translate(text)

    logger.info(f"Splitting text into chunks for translation (size: {len(text)} chars)")

    # Split at natural boundaries (paragraphs)
    paragraphs = text.split('\n\n')
    translated_chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed chunk size
        if current_chunk and len(current_chunk) + len(para) > max_chunk_size:
            translated_chunks.append(translator.translate(current_chunk))
            current_chunk = para
        else:
            current_chunk += '\n\n' + para if current_chunk else para

    # Translate remaining chunk
    if current_chunk:
        translated_chunks.append(translator.translate(current_chunk))

    # Combine translated chunks
    return '\n\n'.join(translated_chunks)


def get_translation_status() -> dict:
    """Get status of translation system for health checks"""
    return {
        "initialized": TRANSLATION_INITIALIZED,
        "fr_en_available": TRANSLATOR_FR_EN is not None,
        "en_fr_available": TRANSLATOR_EN_FR is not None
    }