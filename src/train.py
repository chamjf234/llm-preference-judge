"""Fine-tuning de DeBERTa-v3 como clasificador de 3 clases (Fase 2).

Las funciones puras (aumento A/B, truncado, ensamblaje de tokens) viven aquí y se
testean en local sin GPU (tests/test_train.py). El main() de entrenamiento corre en
Kaggle vía notebooks/02_finetune_deberta.ipynb, que clona este repo.

Decisiones clave (ver docs/superpowers/plans/2026-07-05-fase2-deberta.md):
truncado estructurado prompt-cap + head/tail, aumento A<->B, label smoothing 0.1,
TTA al evaluar. Val = fold canónico (data.CANONICAL_FOLD); baseline a batir: 1.0451.
"""
from __future__ import annotations

import pandas as pd

LABEL_SWAP = {0: 1, 1: 0, 2: 2}  # gana_A <-> gana_B; el empate es simétrico


def swap_ab(df: pd.DataFrame, label_col: str = "_y") -> pd.DataFrame:
    """Copia del df con las respuestas intercambiadas y la etiqueta re-mapeada.

    Doble uso: (1) aumento de datos en train — el modelo ve cada par en ambos
    órdenes y no puede aprender un sesgo posicional; (2) TTA en inferencia.
    """
    out = df.copy()
    out["response_a"] = df["response_b"].to_numpy()
    out["response_b"] = df["response_a"].to_numpy()
    if label_col in out:
        out[label_col] = df[label_col].map(LABEL_SWAP).to_numpy()
    return out


def augment_with_swap(df: pd.DataFrame, label_col: str = "_y") -> pd.DataFrame:
    """Train duplicado: original + intercambiado. SOLO para el fold de train."""
    return pd.concat([df, swap_ab(df, label_col)], ignore_index=True)


def head_tail(ids: list[int], budget: int, tail_frac: float = 0.25) -> list[int]:
    """Trunca a `budget` tokens conservando inicio y FINAL de la secuencia.

    Por qué no solo head: el final de una respuesta suele traer la conclusión,
    que pesa en la preferencia humana. 75/25 y no 50/50 porque el planteamiento
    inicial también importa; es la palanca a experimentar si hay tiempo de GPU.
    """
    if len(ids) <= budget:
        return ids
    n_tail = int(budget * tail_frac)
    n_head = budget - n_tail
    return ids[:n_head] + (ids[len(ids) - n_tail:] if n_tail else [])


MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN = 512
PROMPT_BUDGET = 128  # el prompt es corto (mediana ~23 tokens); 128 cubre >85% completo


def encode_pair(tok, prompt: str, resp_a: str, resp_b: str,
                max_len: int = MAX_LEN, prompt_budget: int = PROMPT_BUDGET) -> dict:
    """[CLS] prompt [SEP] resp_a [SEP] resp_b [SEP], con presupuesto por campo.

    Ensamblamos a nivel de tokens (no tokenizamos un string formateado) porque el
    truncado es POR CAMPO: prompt cap fijo, y el resto repartido entre respuestas.
    Si una respuesta no usa su mitad, la otra hereda el sobrante (dos pasadas).
    NO se hace padding aquí: el DataCollatorWithPadding rellena por batch al vuelo
    (dinámico = menos cómputo desperdiciado que rellenar todo a 512).
    """
    ids = lambda s: tok.encode(s, add_special_tokens=False)
    p = head_tail(ids(prompt), prompt_budget)

    rest = max_len - 4 - len(p)          # 4 especiales: [CLS] + 3 [SEP]
    a_raw, b_raw = ids(resp_a), ids(resp_b)
    budget_a = budget_b = rest // 2
    if len(a_raw) < budget_a:            # A corta: B hereda el sobrante
        budget_b = rest - len(a_raw)
    elif len(b_raw) < budget_b:          # B corta: A hereda el sobrante
        budget_a = rest - len(b_raw)
    a, b = head_tail(a_raw, budget_a), head_tail(b_raw, budget_b)

    input_ids = ([tok.cls_token_id] + p + [tok.sep_token_id]
                 + a + [tok.sep_token_id] + b + [tok.sep_token_id])
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
