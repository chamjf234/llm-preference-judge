"""Baseline clásico: TF-IDF + Regresión Logística (Fase 1).

Objetivo: fijar un PISO honesto de log loss contra el cual medir si el transformer
(DeBERTa, Fase 2) realmente aporta. Si el fine-tuning no le gana a esto, algo va mal.

Se comparan DOS formas de construir las features, para mostrar por qué la ESTRUCTURA
del problema (dos respuestas que se comparan) importa:

  1. "combinado" — TF-IDF sobre prompt+A+B juntos. Bag-of-words: el modelo NO puede
     distinguir qué palabra vino de A vs de B, así que solo pesca señal global (longitud,
     ciertas palabras). Es el piso más ingenuo.

  2. "diferencia" — TF-IDF de A y de B por separado con el MISMO vocabulario, y como
     features la resta tfidf(A) - tfidf(B). Esa resta captura "qué tiene A que no tiene
     B", que es justo la señal de preferencia. Respeta la simetría A/B del problema.

Evaluación: split AGRUPADO por prompt (ningún prompt cae a la vez en train y val, para no
inflar la métrica — lo detectamos en el EDA) y métrica log loss (la oficial de Kaggle).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

import data  # src/data.py (Python añade el dir del script a sys.path al ejecutarlo)

# Rutas resueltas respecto al proyecto, no al cwd: así el script corre desde donde sea.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"

RANDOM_STATE = 42
# Parámetros del vectorizador. Comentados porque son las palancas típicas del baseline:
#   - max_features acota el vocabulario (memoria/velocidad).
#   - ngram_range (1,2) añade bigramas: capturan algo de contexto local ("no es bueno").
#   - min_df=5 descarta términos rarísimos (ruido, erratas).
#   - sublinear_tf amortigua términos muy frecuentes (log(tf) en vez de tf).
TFIDF_KW = dict(max_features=50_000, ngram_range=(1, 2), min_df=5, sublinear_tf=True)


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """Une prompt + A + B en un solo string (versión vectorizada de data.format_pair).

    Se usa la versión con operaciones de pandas (no data.format_pair fila a fila) por
    velocidad: son ~46k filas y el .str es mucho más rápido que un apply en Python.
    """
    return (
        df["prompt"].fillna("")
        + "\n\nRespuesta A: " + df["response_a"].fillna("")
        + "\n\nRespuesta B: " + df["response_b"].fillna("")
    )


def split_by_prompt(df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Split agrupado por prompt.

    GroupShuffleSplit garantiza que todas las filas que comparten un mismo `prompt` caen
    del mismo lado (train O val). Sin esto, el modelo podría "memorizar" un prompt visto en
    train y encontrárselo en val → métrica optimista y engañosa.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["prompt"]))
    return df.iloc[tr_idx], df.iloc[va_idx]


def _fit_eval(X_tr, y_tr, X_va, y_va) -> tuple[float, LogisticRegression]:
    """Entrena LogReg multinomial y devuelve (log_loss en val, modelo).

    No fijamos multi_class: en sklearn>=1.5 el default con solver lbfgs ya es multinomial,
    que es lo correcto para 3 clases (una sola frontera softmax, no 3 one-vs-rest).
    max_iter alto porque en matrices TF-IDF grandes lbfgs a veces necesita más iteraciones
    para converger.
    """
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_va)
    # labels explícito para que el orden de columnas de proba coincida con 0/1/2 siempre.
    return log_loss(y_va, proba, labels=[0, 1, 2]), clf


def run_combined(train_df, val_df, y_tr, y_va) -> float:
    """Baseline 1: TF-IDF sobre el texto combinado (ciego a la estructura A/B)."""
    vec = TfidfVectorizer(**TFIDF_KW)
    X_tr = vec.fit_transform(build_combined_text(train_df))
    X_va = vec.transform(build_combined_text(val_df))
    ll, _ = _fit_eval(X_tr, y_tr, X_va, y_va)
    return ll


def run_difference(train_df, val_df, y_tr, y_va) -> float:
    """Baseline 2: features = tfidf(A) - tfidf(B), con vocabulario único (consciente de A/B).

    El vocabulario se ajusta sobre TODAS las respuestas del train (A y B apiladas) para que
    ambas se proyecten al mismo espacio y la resta tenga sentido dimensión a dimensión.
    """
    vec = TfidfVectorizer(**TFIDF_KW)
    vec.fit(pd.concat([train_df["response_a"], train_df["response_b"]]).fillna(""))

    def feats(d):
        a = vec.transform(d["response_a"].fillna(""))
        b = vec.transform(d["response_b"].fillna(""))
        return a - b  # sparse - sparse sigue siendo sparse

    ll, _ = _fit_eval(feats(train_df), y_tr, feats(val_df), y_va)
    return ll


def main() -> None:
    df = data.load_train(str(TRAIN_CSV))
    df = df.assign(_y=data.make_label(df))

    train_df, val_df = split_by_prompt(df)
    y_tr, y_va = train_df["_y"].to_numpy(), val_df["_y"].to_numpy()

    floor = float(np.log(3))
    print(f"train={len(train_df):,}  val={len(val_df):,}")
    print(f"piso aleatorio informado (ln 3) = {floor:.4f}\n")

    ll_comb = run_combined(train_df, val_df, y_tr, y_va)
    print(f"[1] combinado (prompt+A+B, ciego A/B):   log loss = {ll_comb:.4f}")

    ll_diff = run_difference(train_df, val_df, y_tr, y_va)
    print(f"[2] diferencia tfidf(A)-tfidf(B):        log loss = {ll_diff:.4f}")

    print(f"\nmejor baseline = {min(ll_comb, ll_diff):.4f}  "
          f"(mejora vs piso: {floor - min(ll_comb, ll_diff):+.4f})")


if __name__ == "__main__":
    main()
