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


def test_encode_pair_max_len_pequeno_no_desborda(tok):
    """Regresión: con max_len=64 (modo smoke) y prompt largo, prompt_budget=128 dejaba
    presupuestos NEGATIVOS y el slicing negativo devolvía la respuesta casi entera
    (secuencias de ~3700 tokens → OOM en el smoke test)."""
    largo = "palabra " * 3000
    enc = train.encode_pair(tok, largo, largo, largo, max_len=64)
    assert len(enc["input_ids"]) <= 64


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


def test_grid_blend_prefiere_al_modelo_perfecto():
    import ensemble
    y = np.array([0, 1, 2, 0, 1, 2])
    perfecto = np.eye(3)[y] * 0.94 + 0.02          # casi one-hot correcto
    uniforme = np.full((6, 3), 1 / 3)
    best_w, best_ll, tabla = ensemble.grid_blend(perfecto, uniforme, y)
    assert best_w == 1.0                            # todo el peso al modelo bueno
    assert best_ll < 0.2
    assert len(tabla) == 11                         # grid de 0.0 a 1.0 en pasos de 0.1


def test_grid_blend_mezcla_gana_a_los_extremos():
    """La mezcla gana cuando la reducción de varianza (errores decorrelacionados
    GRANDES) supera la brecha de calidad entre los modelos. Con ruido chico el
    mejor modelo se lleva todo el peso — y eso también es comportamiento correcto."""
    import ensemble
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    # dos "modelos" con MUCHO ruido independiente y el mismo sesgo leve a la verdad
    def modelo_ruidoso(seed):
        r = np.random.default_rng(seed)
        p = np.full((300, 3), 1 / 3) + r.normal(0, 0.25, (300, 3))
        p[np.arange(300), y] += 0.10
        p = np.clip(p, 1e-3, None)
        return p / p.sum(axis=1, keepdims=True)
    best_w, best_ll, tabla = ensemble.grid_blend(modelo_ruidoso(1), modelo_ruidoso(2), y)
    ll_solo_a, ll_solo_b = tabla[-1][1], tabla[0][1]
    assert best_ll < min(ll_solo_a, ll_solo_b)      # la mezcla gana a ambos extremos
    assert 0.0 < best_w < 1.0                       # y el óptimo es una mezcla real


def test_tta_anula_el_sesgo_posicional(tok):
    df = pd.DataFrame({
        "prompt": ["p"], "response_a": ["aaa"], "response_b": ["bbb"], "_y": [0],
    })
    proba = train.predict_proba_tta(_FakeTrainer(), tok, df, max_len=64)
    # un modelo que SIEMPRE dice "gana A" en ambos órdenes debe quedar 50/50 tras TTA
    assert proba.shape == (1, 3)
    assert abs(proba[0, 0] - proba[0, 1]) < 1e-9
    assert abs(proba.sum() - 1.0) < 1e-6
