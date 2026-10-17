
import streamlit as st
import torch
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adaptation pédagogique EIAH",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CHEMINS
# ============================================================

MODEL_REPO = "Ichraf-Ben-Chaabane/modernbert-eiah-sentiment"

# Répertoire de l'application
BASE_DIR = Path(__file__).resolve().parent

CHEMIN_HISTORIQUE = (
    BASE_DIR / "data" / "evolution_5_sessions.csv"
)

CHEMIN_FIGURE_SENTIMENTS = (
    BASE_DIR / "figures" / "evolution_sentiments_par_session.png"
)

CHEMIN_FIGURE_STYLE = (
    BASE_DIR / "figures" / "evolution_style_enseignement.png"
)


# ============================================================
# PROFIL EXPERIMENTAL
# ============================================================

PROFIL_EXPERIMENTAL = (
    "ACTIF",
    "SENSORIEL",
    "VISUEL",
    "SEQUENTIEL"
)


# ============================================================
# BASE DE REGLES PEDAGOGIQUES
# ============================================================

BASE_REGLES = {

    # Style actuel : COMMANDEMENT

    ("POSITIF", *PROFIL_EXPERIMENTAL, "COMMANDEMENT"):
        "PRATIQUE",

    ("NEUTRE", *PROFIL_EXPERIMENTAL, "COMMANDEMENT"):
        "COMMANDEMENT",

    ("NEGATIF", *PROFIL_EXPERIMENTAL, "COMMANDEMENT"):
        "COMMANDEMENT",


    # Style actuel : PRATIQUE

    ("POSITIF", *PROFIL_EXPERIMENTAL, "PRATIQUE"):
        "INCLUSION",

    ("NEUTRE", *PROFIL_EXPERIMENTAL, "PRATIQUE"):
        "PRATIQUE",

    ("NEGATIF", *PROFIL_EXPERIMENTAL, "PRATIQUE"):
        "COMMANDEMENT",


    # Style actuel : INCLUSION

    ("POSITIF", *PROFIL_EXPERIMENTAL, "INCLUSION"):
        "DECOUVERTE_GUIDEE",

    ("NEUTRE", *PROFIL_EXPERIMENTAL, "INCLUSION"):
        "INCLUSION",

    ("NEGATIF", *PROFIL_EXPERIMENTAL, "INCLUSION"):
        "PRATIQUE",


    # Style actuel : DECOUVERTE GUIDEE

    ("POSITIF", *PROFIL_EXPERIMENTAL, "DECOUVERTE_GUIDEE"):
        "DECOUVERTE_GUIDEE",

    ("NEUTRE", *PROFIL_EXPERIMENTAL, "DECOUVERTE_GUIDEE"):
        "DECOUVERTE_GUIDEE",

    ("NEGATIF", *PROFIL_EXPERIMENTAL, "DECOUVERTE_GUIDEE"):
        "INCLUSION",
}


def recommander_style(sentiment, profil, style_actuel):

    cle = (
        sentiment,
        *profil,
        style_actuel
    )

    return BASE_REGLES.get(cle)


# ============================================================
# CHARGEMENT MODERNBERT
# ============================================================

@st.cache_resource
def charger_modernbert():

    # Token Hugging Face stocké dans les secrets Streamlit
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        token=hf_token
    )

    modele = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO,
        token=hf_token
    )

    modele.to("cpu")
    modele.eval()

    return tokenizer, modele


# ============================================================
# ANALYSE SENTIMENTALE
# ============================================================

def analyser_sentiment(commentaire):

    tokenizer, modele = charger_modernbert()

    entrees = tokenizer(
        commentaire,
        return_tensors="pt",
