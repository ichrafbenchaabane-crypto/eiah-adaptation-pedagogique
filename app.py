"Profil + style actuel → Règle d'orientation → "
        "Style recommandé → Décision de régulation "
        "entre les sessions."
    )

    st.info(
        "Simulation contrôlée : 5 sessions, "
        "6 commentaires par session et un intervalle "
        "simulé de 30 minutes."
    )

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


# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Prototype développé dans le cadre du mémoire de recherche : "
    "analyse sentimentale et orientation dynamique "
    "du style d'enseignement dans un EIAH."
)
