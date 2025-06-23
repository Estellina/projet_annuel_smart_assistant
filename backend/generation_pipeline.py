import torch
from nucleus_decoder import nucleus_sampling_decode
from pdf_pipeline import PDFProcessor
from translation import translate_fr_to_en, translate_en_to_fr, detect_lang
from utils import clean_and_format_summary
from model import TransformerSummarizer
from sentence_transformers import SentenceTransformer, util

# ---- Paramètres principaux ---- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_path = "../tokenizer"
pdf_path = "../Cours.pdf"
max_chunk_len = 1024

# ---- Initialisations ---- #
processor = PDFProcessor(tokenizer_path)
tokenizer = processor.tokenizer
vocab_size = tokenizer.get_vocab_size()
pad_id = tokenizer.token_to_id("<pad>")

# ---- Chargement du modèle entraîné ---- #
model = TransformerSummarizer(
    vocab_size,
    pad_id=pad_id,
    d_model=768,
    nhead=12,
    num_layers=6,
    dropout=0.2,
    use_checkpointing=False
).to(device)
model.load_state_dict(torch.load("../checkpoints/best_model.pth", map_location=device))
model.eval()

# ---- Chargement modèle MiniLM ---- #
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Extraction du texte ---- #
raw_text = processor.extract_text(pdf_path)
if detect_lang(raw_text) == "fr":
    raw_text = translate_fr_to_en(raw_text)

chunks = processor.chunk_and_tokenize(raw_text)

summaries = []
for i, chunk in enumerate(chunks):
    src_ids = torch.tensor([chunk], device=device)
    src_mask = (src_ids != pad_id).long()
    summary_en = nucleus_sampling_decode(model, src_ids, src_mask, tokenizer, p=0.9)

    summary_en = clean_and_format_summary(summary_en)
    summaries.append(summary_en)

# ---- Structuration thématique ---- #
def cluster_summaries(sentences, model, threshold=0.6):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    clusters = []
    assigned = [False] * len(sentences)

    for i, emb in enumerate(embeddings):
        if assigned[i]:
            continue
        group = [sentences[i]]
        assigned[i] = True
        for j in range(i + 1, len(sentences)):
            if not assigned[j]:
                sim = util.pytorch_cos_sim(emb, embeddings[j]).item()
                if sim > threshold:
                    group.append(sentences[j])
                    assigned[j] = True
        clusters.append(group)
    return clusters

clustered = cluster_summaries(summaries, embedder)

# ---- Reconstruction et traduction ---- #
structured_en = "\n\n".join([" ".join(group) for group in clustered])
structured_fr = translate_en_to_fr(structured_en)

# ---- Affichage final ---- #
print("\n📚 Résumé structuré final (français) :\n")
print(structured_fr)
