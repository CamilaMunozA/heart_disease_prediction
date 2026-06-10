# 🫀 Asistente Cardiológico — Predicción de Enfermedad Cardíaca con IA

Modelo de Machine Learning + asistente conversacional que estima el riesgo de enfermedad cardíaca a partir de variables clínicas, explica **por qué** con SHAP, entrega una **probabilidad calibrada** y se despliega como **API web**.

> ⚠️ **Aviso médico:** este proyecto es educativo/demostrativo. **No sustituye** el diagnóstico ni la consulta de un profesional de la salud.

---

## 📋 Tabla de contenidos
- [Descripción](#-descripción)
- [Dataset](#-dataset)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
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
   - `CalibratedClassifierCV` (sigmoide/Platt) para que "70% de riesgo" signifique de verdad un 70%.

---

## ✨ Características diferenciales

- 🎯 **Umbral clínico** que prioriza no dejar pasar enfermos (menos falsos negativos).
- 📊 **Probabilidad calibrada** + **nivel de confianza** (Alta/Media/Baja).
- 🧠 **Explicabilidad SHAP**: qué variables aumentan/reducen el riesgo de *ese* paciente, en lenguaje natural.
- ⚖️ **Auditoría de equidad/sesgo** por sexo y grupos de edad (recall, FPR, precisión, tasa de falsos negativos por subgrupo).
- 💬 **Memoria conversacional** con ventana acotada (no agota tokens).

---

## 🤖 El agente conversacional

Construido con **LangChain** + **Groq** (`llama-3.3-70b-versatile`). Decide en cada turno si:
- **Llama a la herramienta** `predecir_enfermedad_cardiaca` (cuando hay datos clínicos), o
- **Responde directamente** (preguntas informativas sobre exámenes, prevención, etc.).

Tras una predicción explica el resultado, comunica probabilidad/confianza/factores SHAP y agrega recomendaciones, recordando los datos ya mencionados en la conversación.

---

## 🚀 Despliegue (API)

Stack: **FastAPI** + **uvicorn** + **Redis** (memoria persistente), pensado para **Render**.

### Configuración en Render
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

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

---

## ⚙️ Configuración

### Variables de entorno requeridas

| Variable | Valor | Dónde obtenerlo |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | [console.groq.com](https://console.groq.com) |
| `REDIS_URL` | `redis://default:password@host:port` | Redis Cloud / Upstash / Render |

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
agente_cardiaco/
├── agente.py          # Modelo + tool de predicción + agente con memoria Redis
├── main.py            # API FastAPI (endpoints)
├── requirements.txt   # Dependencias
├── README.md          # Este archivo
└── *.joblib           # Modelo y objetos de preprocesamiento serializados
```

El entrenamiento completo está en el notebook `Heart_disease_prediction.ipynb`.

> **Nota:** fija `scikit-learn` en `requirements.txt` a la misma versión usada al entrenar (`import sklearn; print(sklearn.__version__)`) para que los `.joblib` carguen sin errores.

---

## 🛠️ Tecnologías

`Python` · `scikit-learn` · `Keras/TensorFlow` · `SHAP` · `pandas` · `LangChain` · `Groq (Llama 3.3 70B)` · `FastAPI` · `uvicorn` · `Redis` · `Render`

---

## ⚠️ Aviso

Proyecto con fines educativos y de investigación. Las predicciones **no constituyen un diagnóstico médico**. Ante cualquier síntoma o duda, consulta a un profesional de la salud.

---
