"""Tests de las funciones puras de src/train.py (truncado, ensamblaje, aumento A/B).

Corren en local sin GPU: el objetivo es no descubrir bugs de lógica a mitad de un
entrenamiento de 2 horas en Kaggle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

import train


def _df():
    return pd.DataFrame({
        "prompt": ["p1", "p2", "p3"],
        "response_a": ["a1", "a2", "a3"],
        "response_b": ["b1", "b2", "b3"],
        "_y": [0, 1, 2],   # gana A, gana B, empate
    })


def test_swap_ab_intercambia_texto_y_etiqueta():
    out = train.swap_ab(_df())
    assert list(out["response_a"]) == ["b1", "b2", "b3"]
    assert list(out["response_b"]) == ["a1", "a2", "a3"]
    assert list(out["_y"]) == [1, 0, 2]        # 0<->1, empate igual
    assert list(out["prompt"]) == ["p1", "p2", "p3"]


def test_swap_ab_no_muta_el_original():
    df = _df()
    train.swap_ab(df)
    assert list(df["response_a"]) == ["a1", "a2", "a3"]
    assert list(df["_y"]) == [0, 1, 2]


def test_augment_with_swap_duplica():
    out = train.augment_with_swap(_df())
    assert len(out) == 6
    assert list(out["_y"]) == [0, 1, 2, 1, 0, 2]
