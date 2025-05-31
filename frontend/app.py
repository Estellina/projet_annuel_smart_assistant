import streamlit as st
import fitz


# --- Titre ---
st.set_page_config(page_title="Smart Assistant IA", layout="centered")
st.title(" Smart Assistant IA - Prototype")

# --- Choix des sources ---
st.header("1. Choisissez vos sources")

with st.form("source_form"):
    pdf_option = st.checkbox(" Envoyer un document PDF")
    video_option = st.checkbox(" Envoyer une vidéo")
    live_option = st.checkbox(" Captation en direct")
    submitted = st.form_submit_button("Valider")

# --- Interface dynamique après validation ---
if submitted:
    st.success("Sources validées ")

    # === PDF ===
    if pdf_option:
        st.subheader(" Uploader votre document PDF")
        uploaded_pdf = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

    # === Vidéo ===
    if video_option:
        st.subheader(" Uploader votre vidéo")
        uploaded_video = st.file_uploader("Choisissez une vidéo", type=["mp4", "avi", "mov"])

    # === Enregistrement Live (Simulation) ===
    if live_option:
        st.subheader(" Captation en direct (simulation)")
        if st.button(" ▶ Démarrer l'enregistrement"):
            st.info(" Enregistrement en cours... (simulation uniquement)")

    # === Résumé simulé ===
    st.header("📝 Résumé généré")

    if pdf_option and uploaded_pdf:
        st.subheader("📖 Contenu du document PDF")

        # Lire le PDF avec PyMuPDF
        pdf_bytes = uploaded_pdf.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        doc.close()

        # Afficher un extrait du texte
        if full_text.strip():
            st.text_area("Aperçu du contenu extrait", value=full_text[:1500], height=300)
        else:
            st.warning("Aucun texte lisible n'a été extrait du PDF.")

    if video_option and uploaded_video:
        st.subheader("Résumé de la vidéo")
        st.write("La vidéo explique les avantages de l'intelligence artificielle dans l'éducation...")

    if live_option:
        st.subheader("Résumé du live")
        st.write("Le discours en live aborde les tendances 2025 en formation professionnelle...")

    # === Quiz Simulé ===
    st.header("QCM Interactif")

    with st.form("quiz_form"):
        question = st.radio(
            "Question 1 : Quel est l'avantage principal de l'IA selon la vidéo ?",
            ("Automatisation", "Personnalisation", "Gain de temps", "Coût élevé")
        )
        quiz_submit = st.form_submit_button("Valider ma réponse")

    if quiz_submit:
        if question == "Personnalisation":
            st.success(" Bonne réponse !")
        else:
            st.warning(" Mauvaise réponse. La bonne réponse était : Personnalisation.")
