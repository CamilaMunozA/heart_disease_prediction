
# ─────────────────────────────────────────────────────────────────────
# agente.py
# Lógica del asistente cardiológico: carga el modelo Random Forest
# calibrado + el preprocesamiento + SHAP, define la tool de predicción
# y el agente con memoria persistente en Redis.
#
# Es independiente de la API: si cambias FastAPI por otro framework,
# este archivo no se toca.
# ─────────────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
import shap
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory


# ── Carga de modelos y preprocesamiento (UNA sola vez al arrancar) ─────
# Las rutas son relativas a este archivo, así funciona en cualquier servidor.
BASE = os.path.dirname(os.path.abspath(__file__))
def _ruta(f):
    return os.path.join(BASE, f)

mediana_chol       = joblib.load(_ruta('mediana_chol.joblib'))
label_encoders     = joblib.load(_ruta('label_encoders.joblib'))
scaler_rf          = joblib.load(_ruta('scaler_rf.joblib'))
feature_names      = joblib.load(_ruta('feature_names.joblib'))
model_rf           = joblib.load(_ruta('random_forest_tuned_model.joblib'))
model_rf_calibrado = joblib.load(_ruta('rf_calibrado.joblib'))
UMBRAL_CLINICO     = joblib.load(_ruta('umbral_clinico.joblib'))

# Explicador SHAP sobre el RF base (rápido para árboles)
explainer_shap = shap.TreeExplainer(model_rf)

ETIQUETAS_FEATURES = {
    'age': 'edad', 'sex': 'sexo', 'chestpaintype': 'tipo de dolor de pecho',
    'cholesterol': 'colesterol', 'fastingbs': 'glucemia en ayunas',
    'maxhr': 'frecuencia cardíaca máxima', 'exerciseangina': 'angina por ejercicio',
    'oldpeak': 'oldpeak (depresión del ST)', 'st_slope': 'pendiente del ST',
    'restingbp': 'presión en reposo', 'restingecg': 'ECG en reposo',
}


def nivel_confianza(prob):
    """Confianza según cuán lejos esté la probabilidad de 0.5 (zona de duda)."""
    margen = abs(prob - 0.5) * 2.0
    if margen >= 0.6:
        return "Alta"
    if margen >= 0.3:
        return "Media"
    return "Baja"


def predecir_riesgo(age, sex, chestpaintype, restingbp, cholesterol, fastingbs,
                    restingecg, maxhr, exerciseangina, oldpeak, st_slope):
    """Aplica TODO el pipeline (imputación, encoding, escalado, modelo calibrado,
    umbral clínico y SHAP) y devuelve un diccionario estructurado."""
    raw = pd.DataFrame({
        'age': [age], 'sex': [sex], 'chestpaintype': [chestpaintype],
        'restingbp': [restingbp], 'cholesterol': [cholesterol], 'fastingbs': [fastingbs],
        'restingecg': [restingecg], 'maxhr': [maxhr], 'exerciseangina': [exerciseangina],
        'oldpeak': [oldpeak], 'st_slope': [st_slope],
    })

    # Imputar colesterol = 0 con la mediana guardada
    raw['cholesterol'] = raw['cholesterol'].replace(0, np.nan).fillna(mediana_chol)

    # Codificar categóricas con los LabelEncoders guardados
    for col in ['sex', 'chestpaintype', 'restingecg', 'exerciseangina', 'st_slope']:
        raw[col] = label_encoders[col].transform(raw[col])

    # Eliminar columnas no usadas y ordenar como en el entrenamiento
    raw = raw.drop(columns=[c for c in ['restingbp', 'restingecg'] if c in raw.columns])
    raw = raw.reindex(columns=feature_names, fill_value=0)

    # Escalar las numéricas
    num_cols = raw.select_dtypes(include=np.number).columns
    raw[num_cols] = scaler_rf.transform(raw[num_cols])

    # Probabilidad calibrada + decisión con umbral clínico
    proba = float(model_rf_calibrado.predict_proba(raw)[0][1])
    pred = int(proba >= UMBRAL_CLINICO)

    # SHAP: contribución de cada variable a ESTA predicción
    vals = np.array(explainer_shap(raw).values)[0]
    if vals.ndim == 2:        # (n_features, n_clases) -> clase 1
        vals = vals[:, 1]
    contrib = sorted(zip(raw.columns, vals), key=lambda x: abs(x[1]), reverse=True)

    return {
        "prediccion": "Enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca",
        "clase": pred,
        "probabilidad": round(proba, 4),
        "confianza": nivel_confianza(proba),
        "factores_aumentan": [ETIQUETAS_FEATURES.get(f, f) for f, v in contrib if v > 0][:3],
        "factores_reducen": [ETIQUETAS_FEATURES.get(f, f) for f, v in contrib if v < 0][:3],
        "umbral": UMBRAL_CLINICO,
    }


@tool
def predecir_enfermedad_cardiaca(
    age: int, sex: str, chestpaintype: str, restingbp: int, cholesterol: int,
    fastingbs: int, restingecg: str, maxhr: int, exerciseangina: str,
    oldpeak: float, st_slope: str,
) -> str:
    """Predice el riesgo de enfermedad cardíaca de un paciente.

    Args:
        age: edad en años. sex: 'M' o 'F'.
        chestpaintype: 'TA','ATA','NAP','ASY'. restingbp: presión en reposo mm Hg.
        cholesterol: colesterol mg/dl (0 se imputa). fastingbs: 1 si >120 mg/dl, si no 0.
        restingecg: 'Normal','ST','LVH'. maxhr: frecuencia cardíaca máxima.
        exerciseangina: 'Y' o 'N'. oldpeak: depresión del ST. st_slope: 'Up','Flat','Down'.

    Returns:
        Texto con predicción, probabilidad calibrada, confianza y factores SHAP.
    """
    try:
        r = predecir_riesgo(age, sex, chestpaintype, restingbp, cholesterol, fastingbs,
                            restingecg, maxhr, exerciseangina, oldpeak, st_slope)
    except ValueError as e:
        return f"Error: valor inválido en algún campo. Detalle: {e}"

    return (
        "RESULTADO DEL MODELO (Random Forest calibrado):\n"
        f"- Predicción: {r['prediccion']} (clase={r['clase']}; umbral clínico={r['umbral']}).\n"
        f"- Probabilidad calibrada de enfermedad: {r['probabilidad']:.1%}.\n"
        f"- Confianza del modelo: {r['confianza']}.\n"
        f"- Factores que MÁS AUMENTAN el riesgo: {', '.join(r['factores_aumentan']) or 'ninguno destacado'}.\n"
        f"- Factores que REDUCEN el riesgo: {', '.join(r['factores_reducen']) or 'ninguno destacado'}."
    )


# ── LLM + agente con tool-calling ─────────────────────────────────────
# GROQ_API_KEY se lee automáticamente de las variables de entorno (Render).
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=1024,
)
tools = [predecir_enfermedad_cardiaca]
tools_by_name = {t.name: t for t in tools}
modelo_con_tools = modelo.bind_tools(tools)

SYSTEM = """Eres un asistente médico especializado en cardiología. Ayudas a interpretar
predicciones de riesgo cardíaco y a entender exámenes relacionados.

Tienes una herramienta `predecir_enfermedad_cardiaca` que ejecuta un Random Forest calibrado.

Reglas:
1. Si el usuario entrega datos clínicos completos y quiere una predicción, usa la herramienta.
   Si falta algún campo, pídelo amablemente antes de llamarla.
2. Si la pregunta es informativa (qué es el colesterol, oldpeak, ST_slope, prevención...),
   responde directamente sin la herramienta.
3. Tras una predicción, comunica SIEMPRE la probabilidad calibrada, la confianza y los
   factores SHAP que aumentan/reducen el riesgo, en lenguaje claro, y agrega 2-3 consejos.
4. Aprovecha la memoria: recuerda los datos ya mencionados y no los vuelvas a pedir.
5. SIEMPRE aclara que NO sustituyes la consulta con un profesional de la salud.
6. Responde en español, con tono cercano y empático."""

# ── Memoria en Redis ──────────────────────────────────────────────────
VENTANA = 10   # últimos 10 mensajes (5 turnos) que se envían al modelo
MAX_ITER = 4   # máximo de ciclos LLM->tool->LLM por mensaje


def obtener_historial(session_id: str) -> RedisChatMessageHistory:
    """Historial por usuario en Redis. ttl=3600 -> se borra tras 1h sin actividad."""
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.environ["REDIS_URL"],
        ttl=3600,
    )


def chatear(session_id: str, mensaje: str) -> dict:
    """Procesa un mensaje del usuario manteniendo la memoria en Redis."""
    historial = obtener_historial(session_id)

    # system + memoria reciente (Redis) + mensaje actual
    mensajes = [SystemMessage(content=SYSTEM)]
    mensajes.extend(list(historial.messages)[-VENTANA:])
    mensajes.append(HumanMessage(content=mensaje))

    for _ in range(MAX_ITER):
        respuesta = modelo_con_tools.invoke(mensajes)
        mensajes.append(respuesta)

        # Sin tool_calls -> respuesta final
        if not getattr(respuesta, "tool_calls", None):
            historial.add_user_message(mensaje)
            historial.add_ai_message(respuesta.content)
            return {
                "respuesta": respuesta.content,
                "session_id": session_id,
                "mensajes_en_memoria": len(historial.messages),
            }

        # Ejecutar cada tool pedida (no se persiste en Redis)
        for call in respuesta.tool_calls:
            tool_fn = tools_by_name[call["name"]]
            resultado = tool_fn.invoke(call["args"])
            mensajes.append(ToolMessage(content=str(resultado), tool_call_id=call["id"]))

    salida = "No pude completar la respuesta en el número máximo de pasos."
    historial.add_user_message(mensaje)
    historial.add_ai_message(salida)
    return {
        "respuesta": salida,
        "session_id": session_id,
        "mensajes_en_memoria": len(historial.messages),
    }
