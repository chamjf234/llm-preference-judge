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


def test_head_tail_corto_queda_igual():
    assert train.head_tail(list(range(10)), budget=20) == list(range(10))


def test_head_tail_recorta_a_presupuesto_exacto():
    ids = list(range(100))
    out = train.head_tail(ids, budget=20, tail_frac=0.25)
    assert len(out) == 20
    assert out[:15] == list(range(15))          # 75% inicio
    assert out[15:] == list(range(95, 100))     # 25% final


def test_head_tail_tail_frac_cero_es_solo_head():
    assert train.head_tail(list(range(100)), budget=10, tail_frac=0.0) == list(range(10))


import pytest


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")


def test_encode_pair_corto_sin_truncar(tok):
    enc = train.encode_pair(tok, "hola", "respuesta a", "respuesta b")
    assert len(enc["input_ids"]) <= train.MAX_LEN
    assert enc["input_ids"][0] == tok.cls_token_id
    assert enc["input_ids"].count(tok.sep_token_id) == 3
    assert len(enc["attention_mask"]) == len(enc["input_ids"])


def test_encode_pair_largo_respeta_max_len(tok):
    largo = "palabra " * 3000
    enc = train.encode_pair(tok, largo, largo, largo)
    assert len(enc["input_ids"]) == train.MAX_LEN


def test_encode_pair_cede_presupuesto_a_la_respuesta_larga(tok):
    corta, larga = "ok", "palabra " * 3000
    enc = train.encode_pair(tok, "hola", corta, larga)
    # si B es larga y A corta, B debe poder usar el sobrante de A:
    # el total llega (casi) al máximo en vez de quedarse a mitad
    assert len(enc["input_ids"]) > train.MAX_LEN - 10


import numpy as np


class _FakeTrainer:
    """Devuelve logits fijos: simula un modelo con sesgo posicional puro hacia A."""
    def predict(self, ds):
        logits = np.tile(np.array([[2.0, 0.0, 0.0]]), (len(ds), 1))  # siempre "gana A"
        return type("P", (), {"predictions": logits})()


def test_tta_anula_el_sesgo_posicional(tok):
    df = pd.DataFrame({
        "prompt": ["p"], "response_a": ["aaa"], "response_b": ["bbb"], "_y": [0],
    })
    proba = train.predict_proba_tta(_FakeTrainer(), tok, df, max_len=64)
    # un modelo que SIEMPRE dice "gana A" en ambos órdenes debe quedar 50/50 tras TTA
    assert proba.shape == (1, 3)
    assert abs(proba[0, 0] - proba[0, 1]) < 1e-9
    assert abs(proba.sum() - 1.0) < 1e-6
