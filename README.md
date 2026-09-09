# TFM: Clasificación Jerárquica de Reclamaciones de Consumidores mediante NLP

Trabajo Fin de Máster del **Máster en Data Science, Big Data & Business Analytics**.

## Descripción

Este proyecto desarrolla un sistema de clasificación automática de texto basado en técnicas de **Natural Language Processing (NLP)** y **Machine Learning**, capaz de transformar el texto libre de una reclamación de consumidor en información estructurada.

El sistema implementa una arquitectura jerárquica de clasificación:

**Consumer Complaint Narrative → Product → Issue**

A partir del texto escrito por el consumidor, el sistema identifica primero el producto financiero relacionado con la reclamación y utiliza posteriormente esta información para determinar el tipo de problema (`Issue`).

El proyecto surge de una problemática empresarial real: la necesidad de clasificar grandes volúmenes de comunicaciones escritas y convertir información no estructurada en categorías útiles para facilitar su posterior gestión.

Debido a las limitaciones para disponer de un histórico corporativo suficientemente amplio y etiquetado, el desarrollo y la evaluación se realizan utilizando el **Consumer Complaint Database** del Consumer Financial Protection Bureau (CFPB), un dataset público de reclamaciones de consumidores.

---

## Objetivos

El objetivo principal es diseñar, entrenar y evaluar un sistema de clasificación jerárquica de texto capaz de predecir automáticamente las variables `Product` e `Issue` a partir de una reclamación escrita en lenguaje natural.

Los objetivos específicos son:

- Analizar y preparar un dataset real de texto no estructurado.
- Estudiar la distribución y el desequilibrio de las clases.
- Comparar diferentes algoritmos clásicos de clasificación de texto.
- Evaluar el impacto del tamaño del conjunto de entrenamiento.
- Aplicar modelos Transformer mediante fine-tuning.
- Diseñar una estrategia jerárquica para la clasificación de `Issue`.
- Analizar la propagación de errores entre los diferentes niveles de clasificación.
- Construir un pipeline de inferencia end-to-end.
- Desarrollar una aplicación interactiva que permita demostrar el funcionamiento del sistema.

---

## Dataset

El proyecto utiliza el **Consumer Complaint Database** publicado por el Consumer Financial Protection Bureau (CFPB).

El dataset contiene reclamaciones reales de consumidores sobre productos y servicios financieros e incluye, entre otras variables:

- `Consumer complaint narrative`: texto libre de la reclamación.
- `Product`: categoría del producto financiero.
- `Issue`: problema asociado a la reclamación.
- `Company`: compañía relacionada con la reclamación.
- `State`: estado desde el que se presenta.
- `Date received`: fecha de recepción.

Para trabajar con una taxonomía reciente y consistente, el análisis principal utiliza reclamaciones recibidas desde el **1 de septiembre de 2023**.

Durante el procesamiento se eliminaron registros sin narrativa útil, duplicados y casos ambiguos en los que una misma narrativa aparecía asociada a diferentes categorías.

Los archivos de datos no se incluyen en el repositorio debido a su tamaño.

---

## Arquitectura del sistema

La arquitectura final utiliza dos niveles de clasificación:

```text
Consumer Complaint Narrative
            │
            ▼
      DistilBERT
            │
            ▼
         Product
            │
            ▼
Modelo específico del Product
   TF-IDF + LinearSVC
            │
            ▼
          Issue
```

### Nivel 1 — Clasificación de Product

Para la predicción de `Product` se compararon inicialmente diferentes modelos clásicos utilizando representaciones TF-IDF:

- Linear SVM
- Logistic Regression
- Logistic Regression con balanceo de clases
- SGDClassifier
- Complement Naive Bayes

El mejor modelo clásico fue **Linear SVM**.

Posteriormente se realizó fine-tuning de **DistilBERT**, obteniendo una mejora respecto al mejor modelo clásico. Por este motivo, DistilBERT fue seleccionado como modelo final para la primera etapa del pipeline.

### Nivel 2 — Clasificación de Issue

La clasificación de `Issue` presenta una mayor complejidad debido al elevado número de categorías y al fuerte desequilibrio entre ellas.

Se evaluaron dos estrategias:

1. Un único clasificador global de `Issue`.
2. Una arquitectura jerárquica con un modelo específico para cada `Product`.

La segunda estrategia obtuvo mejores resultados, por lo que el sistema final utiliza modelos **TF-IDF + LinearSVC específicos para cada Product**.

---

## Resultados principales

### Clasificación de Product

| Modelo | Accuracy | Macro F1 |
|---|---:|---:|
| Linear SVM (100k) | 0.8554 | 0.7118 |
| Linear SVM (250k) | 0.8590 | 0.7185 |
| Linear SVM (500k) | 0.8643 | 0.7271 |
| Linear SVM (dataset completo) | 0.8693 | 0.7366 |
| DistilBERT (250k) | **0.8755** | **0.7525** |

Los resultados muestran que el aumento del volumen de entrenamiento mejora progresivamente el rendimiento del modelo clásico y que el Transformer obtiene el mejor resultado global para `Product`.

### Clasificación de Issue

| Estrategia | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Clasificador global | 0.6160 | 0.2785 | 0.6025 |
| Clasificación jerárquica con Product real | **0.6920** | **0.4279** | **0.6830** |

La arquitectura jerárquica mejora el Macro F1 aproximadamente un **53,7 %** respecto al clasificador global.

---

## Evaluación end-to-end

Evaluar únicamente el segundo nivel utilizando el `Product` real proporciona una estimación optimista del rendimiento, ya que en una aplicación real esta variable también debe ser predicha.

Por ello, se construyó y evaluó el pipeline completo:

```text
Narrative
    ↓
DistilBERT
    ↓
Predicted Product
    ↓
Hierarchical Issue Model
    ↓
Predicted Issue
```

Sobre un holdout común de **5.086 reclamaciones**, el modelo de Product obtuvo:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8748 |
| Macro F1 | 0.7594 |
| Weighted F1 | 0.8725 |

El rendimiento final de `Issue` fue:

| Evaluación | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Product real (oracle) | 0.6897 | 0.4801 | 0.6803 |
| Pipeline end-to-end | **0.6258** | **0.3745** | **0.6157** |

La diferencia entre ambos escenarios permite cuantificar la **propagación de errores** entre los dos niveles.

Cuando `Product` se predice correctamente, la accuracy de `Issue` alcanza aproximadamente el **70,8 %**. Cuando `Product` se predice incorrectamente, la accuracy de `Issue` cae hasta aproximadamente el **5,2 %**.

Este resultado muestra que, en una arquitectura jerárquica, la calidad de la clasificación del primer nivel condiciona significativamente el rendimiento del segundo.

---

## Pipeline de inferencia

La lógica de inferencia se encuentra implementada en:

```text
src/pipeline.py
```

La función principal es:

```python
predecir_pipeline(texto)
```

Ejemplo:

```python
from src.pipeline import predecir_pipeline

resultado = predecir_pipeline(
    "There is an account on my credit report that does not belong to me."
)

print(resultado)
```

Salida esperada:

```python
{
    "Product": "Credit reporting or other personal consumer reports",
    "Issue": "Incorrect information on your report"
}
```

---

## Aplicación Streamlit

Se ha desarrollado una aplicación interactiva con **Streamlit** para demostrar el funcionamiento del pipeline.

La aplicación permite introducir una reclamación en texto libre y obtener automáticamente:

- Product predicho.
- Issue predicho.

Para ejecutar la aplicación localmente:

```bash
streamlit run app/app.py
```

---

## Estructura del repositorio

```text
TFM-Mail-Intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── modelos entrenados
│
├── notebooks/
│   ├── 01_EDA_Enron.ipynb
│   ├── 02_EDA_CFPB.ipynb
│   ├── 03_Modelado_Clasificacion.ipynb
│   ├── 04_Transformer_DistilBERT.ipynb
│   ├── 05_Modelado_Issue.ipynb
│   └── 06_Pipeline_End_to_End.ipynb
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── __init__.py
│   └── pipeline.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Los directorios `data/` y `models/` se mantienen fuera del control de versiones debido al tamaño de los datasets y de los modelos entrenados.

---

## Flujo experimental

Los notebooks documentan la evolución completa del proyecto:

**01 — EDA Enron**  
Evaluación inicial del Enron Email Dataset y análisis de su adecuación al problema.

**02 — EDA CFPB**  
Exploración, limpieza y preparación del Consumer Complaint Database.

**03 — Modelado de clasificación**  
Comparación de modelos clásicos para la predicción de `Product`.

**04 — Transformer DistilBERT**  
Fine-tuning y evaluación de DistilBERT para la clasificación de `Product`.

**05 — Modelado de Issue**  
Desarrollo y comparación de la clasificación global y jerárquica de `Issue`.

**06 — Pipeline End-to-End**  
Integración de ambos niveles y análisis de la propagación de errores.

---

## Instalación

Crear y activar un entorno virtual de Python:

```bash
python -m venv .venv
```

En Windows:

```bash
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

## Modelos y datos

Los datasets y los modelos entrenados no están incluidos en el repositorio debido a su tamaño.

El pipeline requiere los siguientes artefactos locales:

```text
models/
├── issue_hierarchical_svm.pkl
└── distilbert_cfpb_250k/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

Estos modelos son generados durante las etapas de entrenamiento documentadas en los notebooks correspondientes.

---

## Aplicabilidad empresarial

Aunque el sistema se desarrolla y valida utilizando reclamaciones financieras del CFPB, la metodología puede adaptarse a otros escenarios empresariales en los que sea necesario transformar comunicaciones escritas en categorías estructuradas.

Una posible adaptación sería:

```text
Texto recibido
      ↓
Categoría
      ↓
Subcategoría
      ↓
Reglas de negocio
      ↓
Equipo responsable
```

Para una implantación real sería necesario entrenar los modelos utilizando textos y una taxonomía específicos de la organización.

La capa de routing no tendría necesariamente que resolverse mediante Machine Learning. Las categorías predichas por el sistema podrían combinarse con reglas internas de negocio que relacionasen cada combinación de categorías con el equipo responsable.

De esta forma, cambios organizativos en los equipos o responsables podrían gestionarse modificando las reglas de routing sin necesidad de volver a entrenar los modelos NLP.

---

## Limitaciones y líneas futuras

Entre las principales líneas de evolución del proyecto se encuentran:

- Mejorar el rendimiento de las clases minoritarias.
- Evaluar otros modelos Transformer.
- Explorar arquitecturas jerárquicas entrenadas conjuntamente.
- Incorporar estimaciones de confianza en las predicciones.
- Implementar mecanismos de revisión humana para predicciones de baja confianza.
- Adaptar la metodología a datos corporativos reales.
- Integrar las predicciones con sistemas de routing y workflows empresariales.

---

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- DistilBERT
- TF-IDF
- LinearSVC
- Streamlit
- Jupyter Notebook
- Git / GitHub

---

## Estado del proyecto

Pipeline de clasificación jerárquica desarrollado, evaluado e integrado en un prototipo funcional.