from pathlib import Path

import joblib
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT / "models"

DISTILBERT_PATH = MODELS_DIR / "distilbert_cfpb_250k"
ISSUE_MODEL_PATH = MODELS_DIR / "issue_hierarchical_svm.pkl"


# ---------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------

tokenizer_product = AutoTokenizer.from_pretrained(
    DISTILBERT_PATH
)

model_product_transformer = (
    AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_PATH
    )
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_product_transformer.to(device)
model_product_transformer.eval()


artefacto_issue = joblib.load(
    ISSUE_MODEL_PATH
)

modelos_issue = artefacto_issue[
    "modelos_por_producto"
]


# ---------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------

def resolver_componentes_issue(entrada):
    """
    Extrae el vectorizador y el modelo de Issue
    asociados a un Product.
    """

    if isinstance(entrada, dict):

        vec_keys = [
            "vectorizer",
            "vectorizador",
            "tfidf",
            "tfidf_vectorizer"
        ]

        mod_keys = [
            "model",
            "modelo",
            "classifier",
            "clasificador",
            "svm"
        ]

        vectorizador = next(
            (
                entrada[k]
                for k in vec_keys
                if k in entrada
            ),
            None
        )

        modelo = next(
            (
                entrada[k]
                for k in mod_keys
                if k in entrada
            ),
            None
        )

        if (
            vectorizador is not None
            and modelo is not None
        ):
            return vectorizador, modelo

    if (
        isinstance(entrada, (tuple, list))
        and len(entrada) >= 2
    ):
        return entrada[0], entrada[1]

    raise TypeError(
        "No se pudo interpretar la estructura "
        "del modelo de Issue."
    )


def predecir_issue(texto, product):
    """
    Predice Issue a partir de una narrativa
    y de un Product previamente predicho.
    """

    if product not in modelos_issue:
        raise ValueError(
            f"No existe modelo de Issue para: {product}"
        )

    entrada = modelos_issue[product]

    vectorizador, modelo = resolver_componentes_issue(
        entrada
    )

    X = vectorizador.transform([texto])

    return modelo.predict(X)[0]


# ---------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------

def predecir_pipeline(texto):
    """
    Pipeline completo:

    narrativa
        -> DistilBERT
        -> Product
        -> modelo jerárquico
        -> Issue
    """

    if not isinstance(texto, str):
        raise TypeError(
            "El texto debe ser una cadena."
        )

    texto = texto.strip()

    if not texto:
        raise ValueError(
            "El texto no puede estar vacío."
        )

    # -------------------------
    # Product con DistilBERT
    # -------------------------

    inputs = tokenizer_product(
        [texto],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model_product_transformer(
            **inputs
        )

    pred_id = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    product_pred = (
        model_product_transformer
        .config
        .id2label[pred_id]
    )

    # -------------------------
    # Issue jerárquico
    # -------------------------

    issue_pred = predecir_issue(
        texto,
        product_pred
    )

    return {
        "Product": product_pred,
        "Issue": issue_pred
    }