
# ─────────────────────────────────────────────────────────────────────
# main.py — La API. Define los endpoints que el sitio web puede llamar.
# Render ejecuta este archivo con: uvicorn main:app --host 0.0.0.0 --port $PORT
# ─────────────────────────────────────────────────────────────────────

import logging
import os
import uuid
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agente import chatear, obtener_historial, predecir_riesgo

log = logging.getLogger("asistente")

app = FastAPI(
    title="Asistente Cardiológico",
    description="API: Random Forest calibrado + explicación SHAP + memoria en Redis",
    version="1.0.0",
)

# CORS: dominios que pueden llamar la API desde el navegador.
# En Render define ALLOWED_ORIGINS con el dominio de tu sitio, por ejemplo:
#   ALLOWED_ORIGINS=https://misitio.com,https://www.misitio.com
# Si no se define, se permite cualquier dominio ("*").
ORIGENES = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGENES,
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
    """Datos clínicos para una predicción directa (sin pasar por el LLM).
    Los valores categóricos son exactamente los que vieron los LabelEncoders
    al entrenar; cualquier otro valor se rechaza con 422 antes de llegar al modelo."""
    age: int = Field(ge=18, le=100, description="Edad en años")
    sex: Literal["M", "F"]
    chestpaintype: Literal["TA", "ATA", "NAP", "ASY"]
    restingbp: int = Field(ge=0, le=250, description="Presión arterial en reposo (mm Hg)")
    cholesterol: int = Field(ge=0, le=700, description="Colesterol sérico (mg/dl); 0 se imputa con la mediana")
    fastingbs: Literal[0, 1] = Field(description="1 si la glucemia en ayunas es > 120 mg/dl")
    restingecg: Literal["Normal", "ST", "LVH"]
    maxhr: int = Field(ge=50, le=220, description="Frecuencia cardíaca máxima")
    exerciseangina: Literal["Y", "N"]
    oldpeak: float = Field(ge=-3, le=7, description="Depresión del segmento ST")
    st_slope: Literal["Up", "Flat", "Down"]


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
    try:
        r = chatear(session_id=session_id, mensaje=consulta.mensaje)
    except KeyError as e:
        # Falta una variable de entorno (REDIS_URL)
        log.exception("Configuración incompleta")
        raise HTTPException(status_code=503, detail=f"Servicio mal configurado: falta la variable {e}.")
    except Exception as e:
        # Errores de Groq (clave inválida, modelo retirado, límite de uso) o de Redis.
        # Se registra el detalle en los logs de Render y se responde 503 en vez de un 500 opaco.
        log.exception("Error al consultar el agente")
        raise HTTPException(
            status_code=503,
            detail=f"El asistente no está disponible en este momento ({type(e).__name__}). Intenta de nuevo más tarde.",
        )
    return RespuestaChat(
        session_id=session_id,
        respuesta=r["respuesta"],
        mensajes_en_memoria=r["mensajes_en_memoria"],
    )


@app.post("/predecir")
def predecir(paciente: Paciente):
    """Predicción directa del modelo (probabilidad calibrada + confianza + SHAP),
    sin conversación. Útil para integrar el modelo en un formulario."""
    try:
        return predecir_riesgo(**paciente.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Valor inválido: {e}")


@app.delete("/limpiar/{session_id}")
def limpiar(session_id: str):
    """Borra el historial de un usuario en Redis (empezar conversación nueva)."""
    try:
        obtener_historial(session_id).clear()
    except Exception as e:
        log.exception("Error al limpiar el historial")
        raise HTTPException(status_code=503, detail=f"No se pudo acceder a la memoria ({type(e).__name__}).")
    return {"mensaje": f"Historial '{session_id}' eliminado"}
