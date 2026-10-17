
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
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        sorties = modele(**entrees)

        probabilites = torch.softmax(
            sorties.logits,
            dim=-1
        )

    classe = int(
        torch.argmax(
            probabilites,
            dim=-1
        ).item()
    )

    confiance = float(
        probabilites[0][classe].item()
    )

    # Mapping vérifié du checkpoint expérimental
    mapping = {
        0: "NEGATIF",
        1: "NEUTRE",
        2: "POSITIF"
    }

    sentiment = mapping[classe]

    return sentiment, confiance


# ============================================================
# FONCTION D'AFFICHAGE DES STYLES
# ============================================================

def afficher_style(style):

    noms = {
        "COMMANDEMENT": "Commandement",
        "PRATIQUE": "Pratique",
        "INCLUSION": "Inclusion",
        "DECOUVERTE_GUIDEE": "Découverte guidée"
    }

    return noms.get(style, style)


# ============================================================
# EN-TETE
# ============================================================

st.title("🎓 Adaptation pédagogique dans un EIAH")

st.write(
    "Prototype fonctionnel d'orientation dynamique du style "
    "d'enseignement à partir de l'analyse sentimentale "
    "des commentaires d'un apprenant."
)


# ============================================================
# ONGLETS
# ============================================================

tab_apprenant, tab_enseignant, tab_adaptation = st.tabs(
    [
        "👤 Espace Apprenant",
        "👨‍🏫 Espace Enseignant",
        "🔄 Adaptation dynamique"
    ]
)



# ============================================================
# ESPACE APPRENANT
# ============================================================

def reinitialiser_commentaire():
    # Effacer la saisie et le résultat précédent
    st.session_state["commentaire_apprenant"] = ""
    st.session_state.pop("resultat_analyse", None)


with tab_apprenant:

    st.header("Espace Apprenant")

    st.write(
        "Cet espace permet de saisir un commentaire relatif "
        "à l'expérience d'apprentissage."
    )

    commentaire = st.text_area(
        "Commentaire de l'apprenant",
        height=140,
        placeholder="Saisir un commentaire...",
        key="commentaire_apprenant"
    )

           # Deux petits boutons de même largeur, alignés à gauche
    col_analyser, col_reinitialiser, espace = st.columns([1.7, 1.7, 5])

    with col_analyser:
        analyser = st.button(
            "Analyser le commentaire",
            type="primary",
            use_container_width=True
        )

    with col_reinitialiser:
        st.button(
            "🔄 Nouvelle analyse",
            type="secondary",
            on_click=reinitialiser_commentaire,
            use_container_width=True
        )

    if analyser:

        if commentaire.strip():

            with st.spinner(
                "Analyse du commentaire avec ModernBERT..."
            ):
                sentiment, confiance = analyser_sentiment(
                    commentaire
                )

            style_actuel = "COMMANDEMENT"

            style_recommande = recommander_style(
                sentiment,
                PROFIL_EXPERIMENTAL,
                style_actuel
            )

            # Conserver le résultat entre les interactions
            st.session_state["resultat_analyse"] = {
                "commentaire": commentaire.strip(),
                "sentiment": sentiment,
                "confiance": confiance,
                "style_actuel": style_actuel,
                "style_recommande": style_recommande
            }

        else:
            st.session_state.pop("resultat_analyse", None)
            st.warning(
                "Veuillez saisir un commentaire avant "
                "de lancer l'analyse."
            )

    # Afficher le résultat enregistré
    if "resultat_analyse" in st.session_state:

        resultat = st.session_state["resultat_analyse"]

        st.success(
            f"Sentiment détecté : {resultat['sentiment']}"
        )

        st.metric(
            "Confiance du modèle",
            f"{resultat['confiance'] * 100:.2f} %"
        )

        st.divider()

        st.subheader("Orientation pédagogique")

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                "**Style actuel :** "
                + afficher_style(resultat["style_actuel"])
            )

        with col2:
            st.write(
                "**Style recommandé :** "
                + afficher_style(resultat["style_recommande"])
            )

        st.caption(
            "La recommandation est issue de la base "
            "expérimentale de règles pour le profil "
            "Actif–Sensoriel–Visuel–Séquentiel."
        )




# ============================================================
# ESPACE ENSEIGNANT
# ============================================================

with tab_enseignant:

    st.header("Espace Enseignant")

    # --------------------------------------------------------
    # 1. PROFIL DE L'APPRENANT SIMULÉ
    # --------------------------------------------------------

    st.subheader("Profil de l'apprenant simulé")

    st.write("**Apprenant simulé :** Apprenant 1")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dimension 1 :** Actif")
        st.write("**Dimension 2 :** Sensoriel")

    with col2:
        st.write("**Dimension 3 :** Visuel")
        st.write("**Dimension 4 :** Séquentiel")

    st.info(
        "Le profil Felder–Silverman est fixé pour le scénario "
        "expérimental. Il n'est pas déduit des commentaires "
        "ni du dataset Coursera."
    )

    st.divider()

    # --------------------------------------------------------
    # 2. PARAMÈTRES DU SCÉNARIO EXPÉRIMENTAL
    # --------------------------------------------------------

    st.subheader("Paramètres du scénario expérimental")

    st.write("**Style initial :** Commandement")
    st.write("**Nombre de sessions simulées :** 5")
    st.write("**Nombre de commentaires par session :** 6")
    st.write("**Nombre total de commentaires :** 30")
    st.write("**Intervalle temporel simulé :** 30 minutes")

    st.caption(
        "Les sessions et les intervalles temporels sont "
        "simulés. Ils ne correspondent pas à une expérimentation "
        "réalisée auprès de 30 apprenants réels."
    )

    st.divider()

    # --------------------------------------------------------
    # 3. SUIVI DU DERNIER COMMENTAIRE ANALYSÉ
    # --------------------------------------------------------

    st.subheader("Suivi pédagogique du dernier commentaire")

    resultat = st.session_state.get("resultat_analyse")

    if not isinstance(resultat, dict):

        st.info(
            "Aucun commentaire analysé pour le moment. "
            "Veuillez saisir un commentaire dans l'Espace "
            "Apprenant et lancer son analyse."
        )

    else:

        # Les clés sont lues uniquement lorsqu'elles existent.
        # Aucun sentiment ou style n'est inventé.

        def lire_champ(donnees, *cles):
            for cle in cles:
                valeur = donnees.get(cle)
                if valeur is not None and valeur != "":
                    return valeur
            return None

        commentaire = lire_champ(
            resultat,
            "commentaire",
            "texte",
            "text"
        )

        sentiment = lire_champ(
            resultat,
            "sentiment",
            "sentiment_predit",
            "prediction"
        )

        confiance = lire_champ(
            resultat,
            "confiance",
            "score_confiance",
            "confidence"
        )

        style_actuel = lire_champ(
            resultat,
            "style_actuel",
            "style_courant"
        )

        style_recommande = lire_champ(
            resultat,
            "style_recommande",
            "recommandation"
        )

        regle = lire_champ(
            resultat,
            "regle",
            "regle_appliquee",
            "regle_id"
        )

        # ----------------------------------------------------
        # AFFICHAGE DES RÉSULTATS DISPONIBLES
        # ----------------------------------------------------

        if commentaire is not None:
            st.write("**Commentaire analysé :**")
            st.write(str(commentaire))
        else:
            st.caption(
                "Le texte du commentaire n'est pas encore "
                "enregistré dans le résultat d'analyse."
            )

        if sentiment is not None:
            st.write("**Sentiment détecté :**", str(sentiment))
        else:
            st.caption("Sentiment non disponible.")

        if confiance is not None:
            try:
                valeur_confiance = float(confiance)

                # Les probabilités entre 0 et 1 sont
                # converties en pourcentage.
                if 0 <= valeur_confiance <= 1:
                    valeur_confiance *= 100

                st.write(
                    f"**Score de confiance :** "
                    f"{valeur_confiance:.2f} %"
                )

            except (TypeError, ValueError):
                st.write(
                    "**Score de confiance :**",
                    str(confiance)
                )

        else:
            st.caption("Score de confiance non disponible.")

        if style_actuel is not None:
            st.write(
                "**Style d'enseignement actuel :**",
                str(style_actuel)
            )
        else:
            st.caption(
                "Style actuel non enregistré dans ce résultat."
            )

        if style_recommande is not None:
            st.write(
                "**Style recommandé :**",
                str(style_recommande)
            )
        else:
            st.caption("Style recommandé non disponible.")

        if regle is not None:
            st.write(
                "**Règle pédagogique appliquée :**",
                str(regle)
            )
        else:
            st.caption(
                "L'identifiant de la règle appliquée "
                "n'est pas disponible."
            )

        # ----------------------------------------------------
        # 4. INTERPRÉTATION DE LA RECOMMANDATION
        # ----------------------------------------------------

        st.divider()
        st.subheader("Interprétation pédagogique")

        if style_actuel is not None and style_recommande is not None:

            actuel = str(style_actuel).strip().upper()
            recommande = str(style_recommande).strip().upper()

            if actuel == recommande:
                st.success(
                    "La règle pédagogique recommande de "
                    "maintenir le style d'enseignement actuel."
                )
            else:
                st.info(
                    "La règle pédagogique recommande un "
                    "changement de style d'enseignement : "
                    f"{style_actuel} vers {style_recommande}."
                )

        else:
            st.info(
                "L'interprétation sera disponible lorsque "
                "le style actuel et le style recommandé "
                "seront tous les deux enregistrés."
            )

        st.caption(
            "Cette recommandation concerne le dernier "
            "commentaire analysé. Elle ne signifie pas "
            "qu'un changement de style a déjà été appliqué "
            "à une session d'enseignement."
        )



# ============================================================
# ADAPTATION DYNAMIQUE
# ============================================================

with tab_adaptation:

    st.header(
        "Adaptation dynamique"
    )

    st.subheader(
        "Principe de fonctionnement"
    )

    st.write(
        "Commentaire → Analyse sentimentale → "
        "Profil + style actuel → Règle d'orientation → "
        "Style recommandé → Décision de régulation "
        "entre les sessions."
    )

    st.info(
        "Simulation contrôlée : 5 sessions, "
        "6 commentaires par session et un intervalle "
        "simulé de 30 minutes."
    )

    # Commandes d'affichage de la simulation déjà calculée
    if "simulation_visible" not in st.session_state:
        st.session_state["simulation_visible"] = True

    col_reset, col_afficher = st.columns(2)

    with col_reset:
        if st.button("🔄 Réinitialiser l'affichage", key="reset_simulation"):
            st.session_state["simulation_visible"] = False
            st.rerun()

    with col_afficher:
        if st.button("▶️ Afficher les résultats", key="show_simulation"):
            st.session_state["simulation_visible"] = True
            st.rerun()

    if st.session_state["simulation_visible"]:
        # --------------------------------------------------------
        # SENTIMENTS
        # --------------------------------------------------------

        st.divider()

        st.subheader(
            "Évolution des sentiments détectés"
        )

        if CHEMIN_FIGURE_SENTIMENTS.exists():

            st.image(
                str(CHEMIN_FIGURE_SENTIMENTS),
                caption=(
                    "Évolution des sentiments prédits par "
                    "ModernBERT au cours des cinq sessions."
                ),
                use_container_width=True
            )

            st.caption(
                "Chaque session comporte six commentaires. "
                "Les résultats correspondent aux prédictions "
                "obtenues sur les trente commentaires "
                "de l'expérimentation contrôlée."
            )

        else:

            st.warning(
                "Le graphique des sentiments est introuvable."
            )


        # --------------------------------------------------------
        # EVOLUTION DU STYLE
        # --------------------------------------------------------

        st.divider()

        st.subheader(
            "Évolution du style d'enseignement"
        )

        if CHEMIN_FIGURE_STYLE.exists():

            st.image(
                str(CHEMIN_FIGURE_STYLE),
                caption=(
                    "Évolution du style d'enseignement "
                    "au cours des sessions simulées."
                ),
                use_container_width=True
            )

            st.caption(
                "Le graphique représente uniquement les cinq "
                "sessions effectivement simulées. La décision "
                "prise à l'issue de la session 5 concerne le "
                "style qui serait appliqué lors d'une éventuelle "
                "session 6, non simulée dans cette expérimentation."
            )

        else:

            st.warning(
                "Le graphique d'évolution du style est introuvable."
            )


        # --------------------------------------------------------
        # HISTORIQUE
        # --------------------------------------------------------

        st.divider()

        st.subheader(
            "Historique des sessions simulées"
        )

        if CHEMIN_HISTORIQUE.exists():

            historique = pd.read_csv(
                CHEMIN_HISTORIQUE
            )

            st.dataframe(
                historique,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Historique des décisions de régulation "
                "obtenues au cours des cinq sessions simulées."
            )

            st.info(
                "La décision prise à la fin de la session 5 "
                "propose le style de la session suivante. "
                "La session 6 n'a pas été simulée."
            )

        else:

            st.warning(
                "Le fichier d'historique des sessions "
                "est introuvable."
            )
    else:
        st.info("Affichage réinitialisé : aucun résultat expérimental affiché.")

# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Prototype développé dans le cadre du mémoire de recherche : "
    "analyse sentimentale et orientation dynamique "
    "du style d'enseignement dans un EIAH."
)


