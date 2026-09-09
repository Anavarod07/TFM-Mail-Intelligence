import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import (
    predecir_pipeline,
    modelos_issue,
    resolver_componentes_issue,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Consumer Complaint Intelligence",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""

if "resultado" not in st.session_state:
    st.session_state.resultado = None


# --------------------------------------------------
# CUSTOM STYLES
# --------------------------------------------------

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.3rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(29, 78, 216, 0.10),
            rgba(124, 58, 237, 0.06)
        );
        border: 1px solid rgba(80, 100, 150, 0.14);
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-size: 2.65rem;
        font-weight: 750;
        line-height: 1.1;
        margin-bottom: 0.7rem;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.6;
        opacity: 0.78;
        max-width: 850px;
    }

    .flow {
        margin-top: 1.5rem;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.55);
        font-size: 0.95rem;
        font-weight: 600;
        text-align: center;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(80, 100, 150, 0.15);
        background: rgba(80, 100, 150, 0.045);
        min-height: 150px;
    }

    .result-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.62;
        margin-bottom: 0.7rem;
        font-weight: 650;
    }

    .result-value {
        font-size: 1.18rem;
        font-weight: 700;
        line-height: 1.45;
    }

    .section-note {
        opacity: 0.7;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HERO
# --------------------------------------------------

st.markdown(
    """
<div class="hero">
<div class="hero-title">🧠 Consumer Complaint Intelligence</div>
<div class="hero-subtitle">
AI-powered hierarchical classification of consumer complaints.
Transform free-text narratives into structured <b>Product</b> and <b>Issue</b> categories.
</div>
<div class="flow">
Complaint &nbsp; → &nbsp; DistilBERT &nbsp; → &nbsp; Product
&nbsp; → &nbsp; Product-specific classifier &nbsp; → &nbsp; Issue
</div>
</div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------

st.subheader("How it works")

st.markdown(
    """
<div class="section-note">
The system processes each complaint through two consecutive classification stages.
</div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.caption("STEP 01")
        st.markdown("### Read the complaint")
        st.write(
            "The free-text narrative is processed as natural language."
        )

with col2:
    with st.container(border=True):
        st.caption("STEP 02 · PRODUCT")
        st.markdown("### Understand the context")
        st.write(
            "DistilBERT identifies the financial product involved."
        )

with col3:
    with st.container(border=True):
        st.caption("STEP 03 · ISSUE")
        st.markdown("### Identify the problem")
        st.write(
            "A specialized classifier predicts the Issue within that Product."
        )

st.write("")

# --------------------------------------------------
# SYNTHETIC EXAMPLES
# --------------------------------------------------

EXAMPLES = {
    "Option 1": (
        "Last month I paid my credit card balance in full before the due date. "
        "However, my latest statement shows that I was charged a late payment fee. "
        "I contacted the card issuer and provided proof that the payment was made "
        "on time, but they refused to remove the fee and told me that I still have "
        "to pay it."
    ),

    "Option 2": (
        "I recently sold my home and the mortgage was paid off at closing. "
        "Three weeks later, I noticed that my mortgage servicer had withdrawn "
        "another monthly payment from my bank account. I contacted them to request "
        "a refund, but they told me the payment was still being processed and "
        "could not tell me when I would get my money back."
    ),

    "Option 3": (
        "I sent $1,200 to my sister through an online money transfer service. "
        "The money was immediately taken from my account, but five days later "
        "she still had not received it. The company keeps telling me that the "
        "transaction is pending and has not been able to explain where the money "
        "is or when it will arrive."
    ),
}


# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

def load_example(example_name):
    st.session_state.complaint_text = EXAMPLES[example_name]
    st.session_state.resultado = None


def clear_complaint():
    st.session_state.complaint_text = ""
    st.session_state.resultado = None


# --------------------------------------------------
# INPUT AREA
# --------------------------------------------------

st.subheader("Analyze a complaint")

st.markdown(
    """
<div class="section-note">
Paste a consumer complaint below or load one of the synthetic examples.
</div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):

    st.text_area(
        "Consumer complaint narrative",
        key="complaint_text",
        placeholder=(
            "Write or paste a consumer complaint here...\n\n"
            "The model will first identify the Product and then predict "
            "the corresponding Issue."
        ),
        height=190,
    )

    st.caption(
    "Synthetic examples · Select an option to test the pipeline"
)

    ex_col1, ex_col2, ex_col3 = st.columns(3)

with ex_col1:
    st.button(
        "Option 1",
        use_container_width=True,
        on_click=load_example,
        args=("Option 1",),
    )

with ex_col2:
    st.button(
        "Option 2",
        use_container_width=True,
        on_click=load_example,
        args=("Option 2",),
    )

with ex_col3:
    st.button(
        "Option 3",
        use_container_width=True,
        on_click=load_example,
        args=("Option 3",),
    )

    col_clear, col_analyze = st.columns([1, 2])

    with col_clear:
        st.button(
            "Clear",
            use_container_width=True,
            on_click=clear_complaint,
        )

    with col_analyze:
        analyze = st.button(
            "Analyze complaint →",
            type="primary",
            use_container_width=True,
        )
# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if analyze:

    texto = st.session_state.complaint_text.strip()

    if not texto:

        st.warning(
            "Enter a consumer complaint before running the classifier.",
            icon="⚠️",
        )

    else:

        with st.status(
            "Analyzing complaint...",
            expanded=True,
        ) as status:

            try:

                st.write("Reading complaint narrative...")
                st.write("Predicting Product with DistilBERT...")
                st.write("Routing to the corresponding Issue classifier...")

                resultado = predecir_pipeline(texto)

                st.session_state.resultado = resultado

                status.update(
                    label="Classification completed",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:

                st.session_state.resultado = None

                status.update(
                    label="Classification failed",
                    state="error",
                    expanded=True,
                )

                st.error(
                    f"An error occurred during prediction: {e}"
                )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if st.session_state.resultado is not None:

    resultado = st.session_state.resultado

    st.write("")
    st.subheader("Classification result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    📦 Detected Product
                </div>
                <div class="result-value">
                    {resultado["Product"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    🔎 Detected Issue
                </div>
                <div class="result-value">
                    {resultado["Issue"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "The Issue prediction depends on the Product identified during "
        "the first stage of the hierarchical pipeline.",
        icon="ℹ️",
    )


# --------------------------------------------------
# ABOUT THE MODEL
# --------------------------------------------------

with st.expander("About the model"):

    st.markdown(
        """
        ### Model architecture

        This application uses a hierarchical NLP pipeline:

        `Complaint → DistilBERT → Product → Product-specific LinearSVC → Issue`

        The first model predicts the financial **Product**.
        The second stage then uses a classifier specialized for that Product
        to predict the corresponding **Issue**.
        """
    )

    st.divider()

    # --------------------------------------------------
    # PRODUCTS
    # --------------------------------------------------

    st.markdown("### Products covered by the model")

    products = sorted(modelos_issue.keys())

    st.caption(
        f"The hierarchical classifier currently includes {len(products)} Product categories."
    )

    products_df = {
        "Product": products
    }

    st.dataframe(
        products_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # ISSUES BY PRODUCT
    # --------------------------------------------------

    st.markdown("### Issues by Product")

    st.write(
        "Select a Product to inspect the Issue categories available "
        "for that specific branch of the hierarchical classifier."
    )

    selected_product = st.selectbox(
        "Select Product",
        products,
        key="taxonomy_product"
    )

    entrada_issue = modelos_issue[selected_product]

    vectorizador_issue, modelo_issue = resolver_componentes_issue(
        entrada_issue
    )

    issues = sorted(modelo_issue.classes_)

    st.caption(
        f"{selected_product} contains {len(issues)} Issue categories."
    )

    issues_df = {
        "Issue": issues
    }

    st.dataframe(
        issues_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------

    st.markdown("### Technical details")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric(
            "Product model",
            "DistilBERT"
        )

    with col_b:
        st.metric(
            "Issue models",
            f"{len(products)} classifiers"
        )

    st.caption(
        "The Product taxonomy comes from the DistilBERT classifier, while "
        "the Issue taxonomy is organized hierarchically by Product."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Master's Thesis · Hierarchical NLP Classification · "
    "DistilBERT + TF-IDF / LinearSVC"
)