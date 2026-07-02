"""Carga y preparación de datos de la competición.

NOTA: código escrito contra el esquema conocido de Kaggle, pero AÚN NO VERIFICADO
sobre los datos reales. Lo probamos en la Fase 1 con el CSV descargado.

Esquema de train.csv:
    id, model_a, model_b, prompt, response_a, response_b,
    winner_model_a, winner_model_b, winner_tie   (las 3 últimas = target one-hot)
"""
from __future__ import annotations

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


def format_pair(prompt: str, response_a: str, response_b: str) -> str:
    """Une prompt + 2 respuestas en un solo texto para el modelo.

    Formato explícito con marcadores para que el modelo distinga las tres partes.
    La estrategia de truncado (recortar respuestas largas) se decide en la Fase 2/3
    a la hora de tokenizar; aquí solo construimos el texto crudo.
    """
    return (
        f"{prompt}\n\n"
        f"Respuesta A: {response_a}\n\n"
        f"Respuesta B: {response_b}"
    )
