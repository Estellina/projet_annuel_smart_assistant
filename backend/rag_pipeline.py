from sentence_transformers import SentenceTransformer, util

# Vecteur + recherche naïve
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant(text_chunks, question):
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)
    best_idx = scores.argmax()
    return text_chunks[best_idx]
