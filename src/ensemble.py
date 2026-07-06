"""Ensemble de probabilidades: baseline clásico + DeBERTa (Fase 2).

Por qué funciona: los dos modelos cometen errores DISTINTOS — el baseline mide sesgos
de forma (longitud, identidad A==B) y DeBERTa lee contenido. Cuando dos predictores
decorrelacionados se promedian, sus errores se cancelan parcialmente y el log loss
del conjunto baja. Es la mejora más barata que existe: cero entrenamiento adicional.

El peso w se explora en grid sobre validación. OJO: elegir w mirando val introduce
optimismo leve (mismo caveat que la C del baseline); por eso se reporta también el
blend 50/50 "sin tunear", que es el número más honesto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import baseline
import data


def grid_blend(p_model: np.ndarray, p_base: np.ndarray, y_va,
               weights=None) -> tuple[float, float, list[tuple[float, float]]]:
    """Explora w*modelo + (1-w)*baseline y devuelve (mejor_w, mejor_ll, tabla).

    La tabla completa se devuelve (no solo el óptimo) para poder VER la curva:
    un óptimo en w=0.5 con curva plana dice "los dos aportan"; un óptimo pegado
    a un extremo dice que uno de los dos casi no suma.
    """
    if weights is None:
        weights = np.linspace(0.0, 1.0, 11)   # 0.0, 0.1, ..., 1.0
    tabla = []
    for w in weights:
        ll = log_loss(y_va, w * p_model + (1 - w) * p_base, labels=[0, 1, 2])
        tabla.append((float(w), float(ll)))
    best_w, best_ll = min(tabla, key=lambda t: t[1])
    return best_w, best_ll, tabla


def baseline_val_proba(train_df, val_df, y_tr, y_va) -> np.ndarray:
    """Probabilidades de la variante [4] del baseline (diferencia + numéricas) en val.

    Reutiliza las funciones e hiperparámetros de baseline.py (C=0.1) para que sea
    EXACTAMENTE el mismo modelo de la tabla del README, no una variante accidental.
    """
    from scipy.sparse import hstack

    D_tr, D_va = baseline.tfidf_diff_features(train_df, val_df)
    N_tr, N_va = baseline.scaled_numeric(train_df, val_df)
    _, clf = baseline._fit_eval(hstack([D_tr, N_tr]).tocsr(), y_tr,
                                hstack([D_va, N_va]).tocsr(), y_va)
    return clf.predict_proba(hstack([D_va, N_va]).tocsr())


def blend_with_baseline(train_csv: str, p_model: np.ndarray) -> dict:
    """Pipeline completo para el notebook: baseline en el fold canónico + blend.

    p_model son las probabilidades TTA de DeBERTa sobre el fold canónico (el dict
    que devuelve train.main, o el val_proba_tta.npy guardado).
    """
    df = data.load_train(train_csv)
    df = data.add_parsed_text(df)
    df = df.assign(_y=data.make_label(df))
    tr_idx, va_idx = data.grouped_folds(df)[data.CANONICAL_FOLD]
    train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
    y_tr, y_va = train_df["_y"].to_numpy(), val_df["_y"].to_numpy()

    p_base = baseline_val_proba(train_df, val_df, y_tr, y_va)

    ll_base = log_loss(y_va, p_base, labels=[0, 1, 2])
    ll_model = log_loss(y_va, p_model, labels=[0, 1, 2])
    ll_5050 = log_loss(y_va, 0.5 * p_model + 0.5 * p_base, labels=[0, 1, 2])
    best_w, best_ll, tabla = grid_blend(p_model, p_base, y_va)

    print(f"baseline solo:        {ll_base:.4f}")
    print(f"DeBERTa solo (TTA):   {ll_model:.4f}")
    print(f"blend 50/50:          {ll_5050:.4f}   <- número honesto sin tunear")
    print(f"mejor blend (w={best_w:.1f}): {best_ll:.4f}   (w = peso de DeBERTa)")
    print("\ncurva w -> log loss:")
    for w, ll in tabla:
        print(f"  w={w:.1f}  {ll:.4f}")

    return {"ll_base": ll_base, "ll_model": ll_model, "ll_5050": ll_5050,
            "best_w": best_w, "best_ll": best_ll, "tabla": tabla}
