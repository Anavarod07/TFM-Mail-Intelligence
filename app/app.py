import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Permitir importar src desde la raíz del proyecto
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import predecir_pipeline


# ---------------------------------------------------------
# Configuración de la aplicación
# ---------------------------------------------------------

st.set_page_config(
    page_title="CFPB Complaint Classifier",
    page_icon="📩",
    layout="centered"
)


# ---------------------------------------------------------
# Interfaz
# ---------------------------------------------------------

st.title("📩 CFPB Complaint Classifier")

st.write(
    """
    Esta aplicación clasifica automáticamente una reclamación
    de consumidor utilizando un pipeline jerárquico de NLP.
    
    Primero predice el **Product** mediante DistilBERT y,
    posteriormente, predice el **Issue** mediante un modelo
    específico para el Product identificado.
    """
)

st.divider()

texto = st.text_area(
    "Consumer complaint narrative",
    placeholder="Write or paste a consumer complaint here...",
    height=200
)

if st.button("Classify complaint", type="primary"):

    if not texto.strip():

        st.warning(
            "Please enter a complaint before classifying."
        )

    else:

        with st.spinner("Classifying complaint..."):

            try:

                resultado = predecir_pipeline(texto)

                st.success("Classification completed")

                st.subheader("Prediction")

                st.write("**Product**")
                st.info(resultado["Product"])

                st.write("**Issue**")
                st.info(resultado["Issue"])

            except Exception as e:

                st.error(
                    f"An error occurred during prediction: {e}"
                )