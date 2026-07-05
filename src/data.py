"""Carga y preparación de datos de la competición.

Esquema de train.csv:
    id, model_a, model_b, prompt, response_a, response_b,
    winner_model_a, winner_model_b, winner_tie   (las 3 últimas = target one-hot)

IMPORTANTE — formato real de los datos (verificado sobre el CSV):
`prompt`, `response_a` y `response_b` NO son texto plano. Cada celda es una lista
JSON serializada de turnos de conversación (Chatbot Arena permite multi-turno):

    '["primera pregunta","segunda pregunta"]'

Un turno puede ser `null` (el modelo no respondió). Si no se parsea, el modelo
entrena sobre texto contaminado con corchetes, comillas y escapes literales.
Verificado: las 57,477 filas x 3 columnas parsean con json.loads sin excepción,
por eso NO hay fallback defensivo — un dato malformado debe fallar ruidosamente,
no colarse en silencio.
"""
from __future__ import annotations

import json

import pandas as pd

# Orden fijo de clases. Lo fijamos explícitamente para que el índice de clase sea
# estable en todo el pipeline (entrenamiento, submission, app). El orden del
# sample_submission de Kaggle es: winner_model_a, winner_model_b, winner_tie.
LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
CLASS_NAMES = ["gana_A", "gana_B", "empate"]


def load_train(path: str) -> pd.DataFrame:
    """Carga train.csv tal cual."""
    return pd.read_csv(path)


def make_label(df: pd.DataFrame) -> pd.Series:
    """Convierte las 3 columnas one-hot en un único entero de clase (0/1/2).

    Por qué: HuggingFace y sklearn esperan una etiqueta entera, no 3 columnas.
    """
    return df[LABEL_COLS].values.argmax(axis=1)


def parse_turns(raw: str) -> list[str]:
    """Parsea una celda cruda ('["...","..."]') a lista de turnos.

    Los turnos `null` se convierten en cadena vacía: preservan la POSICIÓN del turno
    (la conversación sigue alineada con el prompt) sin inventar contenido. Que una
    respuesta quede vacía es en sí una señal — suele correlacionar con perder/empatar.
    """
    return ["" if t is None else t for t in json.loads(raw)]


def join_turns(turns: list[str]) -> str:
    """Une los turnos de una conversación en un solo texto.

    Con un solo turno (el caso típico) devuelve el texto tal cual, sin ruido.
    Con varios, usa un marcador [Turno i] para que el modelo pueda alinear
    la i-ésima pregunta con la i-ésima respuesta entre prompt y responses.
    """
    if len(turns) == 1:
        return turns[0]
    return "\n\n".join(f"[Turno {i + 1}] {t}" for i, t in enumerate(turns))


TEXT_COLS = ["prompt", "response_a", "response_b"]


def add_parsed_text(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia del df con las columnas de texto parseadas y unidas.

    Se hace en una copia (no in-place) para poder comparar crudo vs limpio y
    porque mutar el argumento de entrada es una fuente clásica de bugs sorpresa.
    """
    out = df.copy()
    for col in TEXT_COLS:
        out[col] = out[col].map(lambda s: join_turns(parse_turns(s)))
    return out


def format_pair(prompt: str, response_a: str, response_b: str) -> str:
    """Une prompt + 2 respuestas en un solo texto para el modelo.

    Espera texto YA parseado (salida de add_parsed_text), no las celdas crudas.
    Formato explícito con marcadores para que el modelo distinga las tres partes.
    La estrategia de truncado (recortar respuestas largas) se decide en la Fase 2/3
    a la hora de tokenizar; aquí solo construimos el texto crudo.
    """
    return (
        f"{prompt}\n\n"
        f"Respuesta A: {response_a}\n\n"
        f"Respuesta B: {response_b}"
    )
