import requests
import fitz  # PyMuPDF
import os
from bs4 import BeautifulSoup

def parse_grobid_tei(tei_xml: str) -> str:
    soup = BeautifulSoup(tei_xml, "lxml-xml")

    # Récupérer le titre
    title = soup.find("titleStmt").find("title").text.strip() if soup.find("titleStmt") else ""

    # Récupérer le résumé (abstract)
    abstract_tag = soup.find("abstract")
    abstract = abstract_tag.get_text(separator="\n").strip() if abstract_tag else ""

    # Récupérer le corps principal
    body_tag = soup.find("body")
    body = body_tag.get_text(separator="\n").strip() if body_tag else ""

    # Combine
    full_text = f"Titre : {title}\n\nRésumé : {abstract}\n\nCorps :\n{body}"
    return full_text

def extract_with_grobid(file_path: str, grobid_url="http://localhost:8070/api/processFulltextDocument") -> str:
    with open(file_path, "rb") as f:
        files = {"input": (os.path.basename(file_path), f, "application/pdf")}
        response = requests.post(grobid_url, files=files)
        if response.status_code == 200:
            return response.text  # C’est du XML TEI à parser ensuite
        else:
            raise Exception(f"GROBID error {response.status_code}: {response.text}")

def extract_with_fitz(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
