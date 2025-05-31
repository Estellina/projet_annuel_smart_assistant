import streamlit as st
import requests
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API backend fonctionnelle"}

st.set_page_config(page_title="Assistant IA - Prototype", layout="centered")
st.title("Assistant IA - Analyse de contenu")

st.header("Envoyer un document")
uploaded_file = st.file_uploader("Choisir un fichier PDF ou vidéo", type=["pdf", "mp4"])
question = st.text_input("Posez une question sur le contenu (optionnel)")

if st.button("Analyser") and uploaded_file:
    with st.spinner("Analyse en cours..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"question": question}
        response = requests.post("http://localhost:8000/analyze", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            st.subheader("Résumé")
            st.write(result["summary"])
            if result["answer"]:
                st.subheader("Réponse IA")
                st.write(result["answer"])

            st.subheader("QCM généré")
            for q in result["quiz"]:
                st.write("**" + q["question"] + "**")
                st.radio("", q["propositions"], index=0, key=q["question"])
        else:
            st.error("Erreur lors de l'analyse. Vérifie le backend.")


