# 🫀 Asistente Cardiológico — Predicción de Enfermedad Cardíaca con IA

Modelo de Machine Learning + asistente conversacional que estima el riesgo de enfermedad cardíaca a partir de variables clínicas, explica **por qué** con SHAP, entrega una **probabilidad calibrada** y se despliega como **API web**.

> ⚠️ **Aviso médico:** este proyecto es educativo/demostrativo. **No sustituye** el diagnóstico ni la consulta de un profesional de la salud.

---

## 📋 Tabla de contenidos
- [Descripción](#-descripción)
- [Dataset](#-dataset)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [Resultados](#-resultados)
- [Características diferenciales](#-características-diferenciales)
- [El agente conversacional](#-el-agente-conversacional)
- [Despliegue (API)](#-despliegue-api)
- [Uso de la API](#-uso-de-la-api)
- [Configuración](#-configuración)
- [Estructura del repo](#-estructura-del-repo)
- [Tecnologías](#-tecnologías)

---

## 🎯 Descripción

La enfermedad cardíaca es una de las principales causas de mortalidad mundial. Este proyecto entrena varios modelos para predecir si un paciente presenta enfermedad cardíaca (`1`) o no (`0`) y construye un **asistente clínico conversacional** sobre el mejor modelo (Random Forest calibrado).

El asistente no es un clasificador "caja negra": para cada predicción devuelve **probabilidad calibrada + nivel de confianza + explicación en lenguaje natural** de qué variables empujaron el riesgo.

---

## 📊 Dataset

[Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle) — 918 registros, 11 variables clínicas:

| Variable | Descripción |
|---|---|
| `age` | Edad (años) |
| `sex` | Sexo (M/F) |
| `chestpaintype` | Tipo de dolor de pecho (TA, ATA, NAP, ASY) |
| `restingbp` | Presión arterial en reposo (mm Hg) |
| `cholesterol` | Colesterol sérico (mg/dl) |
| `fastingbs` | Glucemia en ayunas > 120 mg/dl (1/0) |
| `restingecg` | ECG en reposo (Normal, ST, LVH) |
| `maxhr` | Frecuencia cardíaca máxima |
| `exerciseangina` | Angina por ejercicio (Y/N) |
| `oldpeak` | Depresión del segmento ST |
| `st_slope` | Pendiente del ST (Up, Flat, Down) |
| `heartdisease` | **Objetivo:** 1 = enfermedad, 0 = sano |

---

## 🔬 Pipeline de Machine Learning

1. **Preprocesamiento**
   - Renombrado de columnas a minúsculas.
   - `cholesterol = 0` tratado como nulo e imputado con la **mediana** (robusta a outliers).
   - Codificación de categóricas con `LabelEncoder`.
   - Escalado con `StandardScaler`.
   - Eliminación de `restingbp` y `restingecg` (baja correlación con el objetivo).
2. **Modelos entrenados**
   - Regresión Logística (línea base).
   - **Random Forest** ajustado (`max_depth=5`, `min_samples_leaf=5`) — modelo final.
   - Red neuronal (Keras) con regularización L2 + Dropout + EarlyStopping.
3. **Optimización del umbral de decisión**
   - El umbral por defecto (0.5) se **baja a un umbral clínico** para **reducir los falsos negativos** (el error más grave: declarar "sano" a un enfermo). El umbral se elige automáticamente como el más sensible que mantiene precisión ≥ 0.80 y se **persiste** en `umbral_clinico.joblib`.
4. **Calibración de probabilidades**
   - `CalibratedClassifierCV` (sigmoide/Platt, `cv=5`) para que "70% de riesgo" signifique de verdad un 70%.
5. **Explicabilidad**
   - `shap.TreeExplainer` sobre el Random Forest ajustado. La probabilidad que se reporta viene del modelo calibrado (5 copias del mismo RF + calibración sigmoide), así que SHAP explica la dirección del riesgo del RF base, no el valor calibrado exacto.

---

## 📈 Resultados

<!-- TODO: completar con los valores del notebook Heart_disease_prediction.ipynb (conjunto de prueba). -->

| Modelo | Umbral | Recall (enfermos) | Precisión | ROC-AUC | Brier score |
|---|---|---|---|---|---|
| Regresión logística (base) | 0.50 | | | | |
| Red neuronal (Keras) | 0.50 | | | | |
| Random Forest ajustado | 0.50 | | | | |
| **Random Forest calibrado** | **0.40 (clínico)** | | | | |

**Auditoría de equidad (modelo final, umbral clínico):**

| Subgrupo | n | Recall | Tasa de falsos negativos |
|---|---|---|---|
| Mujeres | | | |
| Hombres | | | |
| < 50 años | | | |
| ≥ 50 años | | | |

---

## ✨ Características diferenciales

- 🎯 **Umbral clínico** que prioriza no dejar pasar enfermos (menos falsos negativos).
- 📊 **Probabilidad calibrada** + **nivel de confianza** (Alta/Media/Baja).
- 🧠 **Explicabilidad SHAP**: qué variables aumentan/reducen el riesgo de *ese* paciente, en lenguaje natural.
- ⚖️ **Auditoría de equidad/sesgo** por sexo y grupos de edad (recall, FPR, precisión, tasa de falsos negativos por subgrupo).
- 💬 **Memoria conversacional** con ventana acotada (no agota tokens).

---

## 🤖 El agente conversacional

Construido con **LangChain** + **Groq** (`openai/gpt-oss-120b`, gratuito en el plan free de Groq). Decide en cada turno si:
- **Llama a la herramienta** `predecir_enfermedad_cardiaca` (cuando hay datos clínicos), o
- **Responde directamente** (preguntas informativas sobre exámenes, prevención, etc.).

Tras una predicción explica el resultado, comunica probabilidad/confianza/factores SHAP y agrega recomendaciones, recordando los datos ya mencionados en la conversación.

> **Cambiar de modelo:** Groq retira modelos con frecuencia ([lista de retiros](https://console.groq.com/docs/deprecations)). El modelo se define con la variable de entorno `GROQ_MODEL`, así que basta con cambiarla en Render, sin tocar el código. Elige uno que soporte *tool use*.

---

## 🚀 Despliegue (API)

Stack: **FastAPI** + **uvicorn** + **Redis** (memoria persistente), pensado para **Render**.

### Configuración en Render
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Versión de Python:** 3.12, fijada en el archivo `.python-version`.

---

## 🔌 Uso de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/consultar` | Chat con el agente (con memoria) |
| `POST` | `/predecir` | Predicción directa del modelo (JSON) |
| `DELETE` | `/limpiar/{session_id}` | Borra la memoria de una conversación |

### Chat con el agente
```bash
curl -X POST https://tu-servicio.onrender.com/consultar \
  -H "Content-Type: application/json" \
  -d '{"mensaje": "Paciente 65 años, hombre, dolor NAP, presión 130, colesterol 200, glucemia 0, ECG Normal, maxhr 135, angina N, oldpeak 0.5, ST_slope Up. ¿Tiene riesgo?"}'
```
Respuesta:
```json
{ "session_id": "abc-123", "respuesta": "Según el modelo...", "mensajes_en_memoria": 2 }
```
> `session_id` es opcional: si no lo envías, la API genera uno. Guárdalo y reenvíalo en cada mensaje para conservar la memoria.

### Predicción directa
```bash
curl -X POST https://tu-servicio.onrender.com/predecir \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":"F","chestpaintype":"ASY","restingbp":145,"cholesterol":0,"fastingbs":1,"restingecg":"ST","maxhr":120,"exerciseangina":"Y","oldpeak":2.1,"st_slope":"Flat"}'
```
Respuesta:
```json
{
  "prediccion": "Enfermedad cardíaca",
  "clase": 1,
  "probabilidad": 0.919,
  "confianza": "Alta",
  "factores_aumentan": ["pendiente del ST", "tipo de dolor de pecho", "oldpeak (depresión del ST)"],
  "factores_reducen": ["sexo"],
  "umbral": 0.4
}
```

> Documentación interactiva en `/docs` (Swagger UI).

> Reemplaza `https://tu-servicio.onrender.com` por la URL de tu servicio en Render, o por `http://127.0.0.1:8000` si lo corres localmente.

### Validación de entradas
`/predecir` valida cada campo antes de llegar al modelo y responde **422** con el detalle si algo no cumple:

| Campo | Valores aceptados |
|---|---|
| `sex` | `M`, `F` |
| `chestpaintype` | `TA`, `ATA`, `NAP`, `ASY` |
| `restingecg` | `Normal`, `ST`, `LVH` |
| `exerciseangina` | `Y`, `N` |
| `st_slope` | `Up`, `Flat`, `Down` |
| `fastingbs` | `0`, `1` |
| `age` | 18 – 100 |
| `restingbp` | 0 – 250 mm Hg |
| `cholesterol` | 0 – 700 mg/dl (0 se imputa con la mediana, 237) |
| `maxhr` | 50 – 220 |
| `oldpeak` | -3 – 7 |

### Errores
| Código | Cuándo |
|---|---|
| **422** | Datos de entrada inválidos (ver tabla anterior). |
| **503** | El LLM (Groq) o la memoria (Redis) no están disponibles: clave inválida, modelo retirado, límite de uso o variable de entorno faltante. El detalle queda en los logs de Render. `/predecir` no depende de ninguno de los dos y sigue funcionando. |

---

## ⚙️ Configuración

### Variables de entorno

| Variable | ¿Requerida? | Valor | Dónde obtenerlo |
|---|---|---|---|
| `GROQ_API_KEY` | Sí (para `/consultar`) | `gsk_...` | [console.groq.com/keys](https://console.groq.com/keys) (gratis) |
| `REDIS_URL` | Sí (para `/consultar` y `/limpiar`) | `redis://default:password@host:port` | Redis Cloud / Upstash / Render |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` (por defecto) | [Modelos de Groq](https://console.groq.com/docs/models) |
| `ALLOWED_ORIGINS` | No | `https://misitio.com,https://www.misitio.com` (por defecto `*`) | El dominio de tu sitio web |

> `openai/gpt-oss-120b` es un modelo abierto de OpenAI **servido por Groq**: se usa con la API key de Groq, no con una de OpenAI.

- **En Google Colab:** panel 🔑 **Secrets** → añade ambas y activa *Notebook access*.
- **En Render:** **Environment → Environment Variables**.

### Instalación local
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 📁 Estructura del repo

```
heart_disease_prediction/
├── agente.py          # Modelo + tool de predicción + agente con memoria Redis
├── main.py            # API FastAPI (endpoints y validación de entradas)
├── requirements.txt   # Dependencias con versiones fijas
├── .python-version    # Versión de Python para Render (3.12)
├── README.md          # Este archivo
└── *.joblib           # Modelo y objetos de preprocesamiento serializados
```

El entrenamiento completo está en el notebook `Heart_disease_prediction.ipynb`.

> **Nota:** los `.joblib` se entrenaron con **scikit-learn 1.6.1**, y `requirements.txt` fija esa versión. Si vuelves a entrenar con otra versión, actualiza el pin para que los modelos carguen sin errores.

---

## 🛠️ Tecnologías

`Python` · `scikit-learn` · `Keras/TensorFlow` · `SHAP` · `pandas` · `LangChain` · `Groq (gpt-oss-120b)` · `FastAPI` · `uvicorn` · `Redis` · `Render`

---

## ⚠️ Aviso

Proyecto con fines educativos y de investigación. Las predicciones **no constituyen un diagnóstico médico**. Ante cualquier síntoma o duda, consulta a un profesional de la salud.

---
