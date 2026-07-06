"""Baseline clásico: TF-IDF + Regresión Logística (Fase 1).

Objetivo: fijar un PISO honesto de log loss contra el cual medir si el transformer
(DeBERTa, Fase 2) realmente aporta. Si el fine-tuning no le gana a esto, algo va mal.

Se comparan CUATRO variantes de features, en progresión, para mostrar DÓNDE vive la señal:

  1. "combinado" — TF-IDF sobre prompt+A+B juntos. Bag-of-words: el modelo NO puede
     distinguir qué palabra vino de A vs de B, así que solo pesca señal global (longitud,
     ciertas palabras). Es el piso más ingenuo.

  2. "diferencia" — TF-IDF de A y de B por separado con el MISMO vocabulario, y como
     features la resta tfidf(A) - tfidf(B). Esa resta captura "qué tiene A que no tiene
     B", que es justo la señal de preferencia. Respeta la simetría A/B del problema.

  3. "numéricas" — SOLO un puñado de features artesanales que codifican sesgos humanos
     documentados en Chatbot Arena: verbosidad (los humanos prefieren la respuesta más
     larga: P(gana A | A más larga) ≈ 0.62 en este dataset), respuestas idénticas
     (A == B → 90% empate) y respuestas vacías. Sin leer ni una palabra del contenido.

  4. "diferencia + numéricas" — la unión de 2 y 3: el piso final a batir por DeBERTa.

Todo se calcula sobre texto PARSEADO (data.add_parsed_text): las columnas crudas son
listas JSON de turnos, no texto plano (ver EDA sección 0).

Evaluación: GroupKFold de 5 folds AGRUPADO por prompt (ningún prompt cae a la vez en train
y val, para no inflar la métrica — lo detectamos en el EDA), métrica log loss (la oficial
de Kaggle). Se reporta media ± desviación entre folds: un solo split tiene varianza de
muestreo y diferencias de ±0.005 podrían ser ruido. El fold data.CANONICAL_FOLD es el que
usará la Fase 2 (DeBERTa se entrena una sola vez), así la comparación es sobre el mismo val.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import data  # src/data.py (Python añade el dir del script a sys.path al ejecutarlo)

# Rutas resueltas respecto al proyecto, no al cwd: así el script corre desde donde sea.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data" / "train.csv"

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


def _fit_eval(X_tr, y_tr, X_va, y_va) -> tuple[float, LogisticRegression]:
    """Entrena LogReg multinomial y devuelve (log_loss en val, modelo).

    No fijamos multi_class: en sklearn>=1.5 el default con solver lbfgs ya es multinomial,
    que es lo correcto para 3 clases (una sola frontera softmax, no 3 one-vs-rest).
    max_iter alto porque en matrices TF-IDF grandes lbfgs a veces necesita más iteraciones
    para converger.

    C=0.1 (más regularización que el default 1.0): con 50k features TF-IDF y ~46k filas
    hay más dimensiones que datos, y sin regularización fuerte el modelo se sobreajusta
    a n-gramas raros. Mini-grid sobre val {0.03, 0.1, 0.3, 1.0}: 0.1 gana en las
    variantes con TF-IDF (1.094 → 1.054 en la [2]) y no afecta a las numéricas (6
    features: nada que regularizar). Elegir C mirando val introduce un optimismo leve,
    aceptable para un baseline; el fold es el mismo para todas las comparaciones.
    """
    clf = LogisticRegression(max_iter=1000, C=0.1)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_va)
    # labels explícito para que el orden de columnas de proba coincida con 0/1/2 siempre.
    return log_loss(y_va, proba, labels=[0, 1, 2]), clf


def numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Features artesanales que codifican sesgos de preferencia humana, sin leer contenido.

    - log1p de longitudes y su diferencia: el sesgo de verbosidad es la señal individual
      más fuerte del dataset. log1p en vez de longitud cruda porque la diferencia entre
      100 y 200 palabras importa mucho más que entre 5000 y 5100 (rendimientos
      decrecientes), y porque comprime la cola larga que vimos en el EDA.
    - A == B idénticas: 90% de esas filas son empate (regla casi determinista).
    - Respuesta vacía (turno null): "no contestó" suele significar perder o empatar.
    """
    wa = df["response_a"].str.split().str.len().to_numpy(dtype=float)
    wb = df["response_b"].str.split().str.len().to_numpy(dtype=float)
    return np.column_stack([
        np.log1p(wa),
        np.log1p(wb),
        np.log1p(wa) - np.log1p(wb),   # >0 si A es más larga; simétrica respecto a A/B
        (df["response_a"] == df["response_b"]).to_numpy(dtype=float),
        (df["response_a"].str.strip() == "").to_numpy(dtype=float),
        (df["response_b"].str.strip() == "").to_numpy(dtype=float),
    ])


def scaled_numeric(train_df, val_df) -> tuple[np.ndarray, np.ndarray]:
    """Numéricas estandarizadas. El scaler se ajusta SOLO con train: si usara también
    val, información de validación se filtraría al entrenamiento (fuga sutil pero real)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(numeric_features(train_df))
    X_va = scaler.transform(numeric_features(val_df))
    return X_tr, X_va


def run_combined(train_df, val_df, y_tr, y_va) -> float:
    """Baseline 1: TF-IDF sobre el texto combinado (ciego a la estructura A/B)."""
    vec = TfidfVectorizer(**TFIDF_KW)
    X_tr = vec.fit_transform(build_combined_text(train_df))
    X_va = vec.transform(build_combined_text(val_df))
    ll, _ = _fit_eval(X_tr, y_tr, X_va, y_va)
    return ll


def tfidf_diff_features(train_df, val_df):
    """Features tfidf(A) - tfidf(B) con vocabulario único (consciente de A/B).

    El vocabulario se ajusta sobre TODAS las respuestas del train (A y B apiladas) para que
    ambas se proyecten al mismo espacio y la resta tenga sentido dimensión a dimensión.
    """
    vec = TfidfVectorizer(**TFIDF_KW)
    vec.fit(pd.concat([train_df["response_a"], train_df["response_b"]]).fillna(""))

    def feats(d):
        a = vec.transform(d["response_a"].fillna(""))
        b = vec.transform(d["response_b"].fillna(""))
        return a - b  # sparse - sparse sigue siendo sparse

    return feats(train_df), feats(val_df)


def run_fold(train_df, val_df, y_tr, y_va) -> dict[str, float]:
    """Corre las 4 variantes sobre un fold y devuelve {nombre: log loss}."""
    lls = {}
    lls["[1] combinado (ciego A/B)"] = run_combined(train_df, val_df, y_tr, y_va)

    # Las features de 2, 3 y 4 se calculan una sola vez y se reutilizan (el fit del
    # TF-IDF es lo caro; repetirlo daría el mismo resultado gastando el doble de tiempo).
    D_tr, D_va = tfidf_diff_features(train_df, val_df)
    N_tr, N_va = scaled_numeric(train_df, val_df)

    lls["[2] diferencia tfidf(A)-tfidf(B)"], _ = _fit_eval(D_tr, y_tr, D_va, y_va)
    lls["[3] solo numéricas"], _ = _fit_eval(N_tr, y_tr, N_va, y_va)

    # hstack sparse + denso: scipy convierte el bloque denso y concatena columnas.
    # csr = formato eficiente para el acceso por filas que hace el entrenamiento.
    lls["[4] diferencia + numéricas"], _ = _fit_eval(hstack([D_tr, N_tr]).tocsr(), y_tr,
                                                     hstack([D_va, N_va]).tocsr(), y_va)
    return lls


def main() -> None:
    df = data.load_train(str(TRAIN_CSV))
    df = data.add_parsed_text(df)  # las columnas crudas son listas JSON, no texto plano
    df = df.assign(_y=data.make_label(df))

    floor = float(np.log(3))
    print(f"piso aleatorio informado (ln 3) = {floor:.4f}")

    resultados: dict[str, list[float]] = {}
    for k, (tr_idx, va_idx) in enumerate(data.grouped_folds(df)):
        train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr, y_va = train_df["_y"].to_numpy(), val_df["_y"].to_numpy()

        marca = "  <- fold canónico (Fase 2)" if k == data.CANONICAL_FOLD else ""
        print(f"\n--- fold {k}: train={len(train_df):,} val={len(val_df):,}{marca}")
        for nombre, ll in run_fold(train_df, val_df, y_tr, y_va).items():
            resultados.setdefault(nombre, []).append(ll)
            print(f"  {nombre:<34} log loss = {ll:.4f}")

    print(f"\n=== resumen 5-fold (media ± std) ===")
    for nombre, lls in resultados.items():
        media, std = float(np.mean(lls)), float(np.std(lls))
        canon = lls[data.CANONICAL_FOLD]
        print(f"  {nombre:<34} {media:.4f} ± {std:.4f}   (fold canónico: {canon:.4f})")

    best = min(float(np.mean(v)) for v in resultados.values())
    print(f"\nmejor baseline (media 5-fold) = {best:.4f}  (mejora vs piso: {floor - best:+.4f})")


if __name__ == "__main__":
    main()
