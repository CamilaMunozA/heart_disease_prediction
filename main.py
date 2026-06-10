
# ─────────────────────────────────────────────────────────────────────
# main.py — La API. Define los endpoints que el sitio web puede llamar.
# Render ejecuta este archivo con: uvicorn main:app --host 0.0.0.0 --port $PORT
# ─────────────────────────────────────────────────────────────────────

import uuid
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agente import chatear, obtener_historial, predecir_riesgo


app = FastAPI(
    title="Asistente Cardiológico",
    description="API: Random Forest calibrado + explicación SHAP + memoria en Redis",
    version="1.0.0",
)

# CORS: permite que el navegador llame la API desde cualquier dominio.
# En producción reemplaza ["*"] por el dominio exacto de tu sitio.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos de datos (validación automática con Pydantic) ─────────────
class Consulta(BaseModel):
    session_id: Optional[str] = None   # opcional: None si es usuario nuevo
    mensaje: str                       # requerido


class RespuestaChat(BaseModel):
    session_id: str
    respuesta: str
    mensajes_en_memoria: int


class Paciente(BaseModel):
    """Datos clínicos para una predicción directa (sin pasar por el LLM)."""
    age: int
    sex: str
    chestpaintype: str
    restingbp: int
    cholesterol: int
    fastingbs: int
    restingecg: str
    maxhr: int
    exerciseangina: str
    oldpeak: float
    st_slope: str


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def raiz():
    """Health check: confirma que la API está activa."""
    return {"estado": "activo", "agente": "Asistente Cardiológico"}


@app.post("/consultar", response_model=RespuestaChat)
def consultar(consulta: Consulta):
    """Chat con el agente (usa la tool del modelo cuando hace falta).
    Si no llega session_id se genera uno; el sitio web debe guardarlo
    y reenviarlo en cada mensaje para conservar la memoria."""
    session_id = consulta.session_id or str(uuid.uuid4())
    r = chatear(session_id=session_id, mensaje=consulta.mensaje)
    return RespuestaChat(
        session_id=session_id,
        respuesta=r["respuesta"],
        mensajes_en_memoria=r["mensajes_en_memoria"],
    )


@app.post("/predecir")
def predecir(paciente: Paciente):
    """Predicción directa del modelo (probabilidad calibrada + confianza + SHAP),
    sin conversación. Útil para integrar el modelo en un formulario."""
    return predecir_riesgo(**paciente.model_dump())


@app.delete("/limpiar/{session_id}")
def limpiar(session_id: str):
    """Borra el historial de un usuario en Redis (empezar conversación nueva)."""
    obtener_historial(session_id).clear()
    return {"mensaje": f"Historial '{session_id}' eliminado"}
