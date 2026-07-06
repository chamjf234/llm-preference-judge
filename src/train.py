"""Fine-tuning de DeBERTa-v3 como clasificador de 3 clases (Fase 2).

Las funciones puras (aumento A/B, truncado, ensamblaje de tokens) viven aquí y se
testean en local sin GPU (tests/test_train.py). El main() de entrenamiento corre en
Kaggle vía notebooks/02_finetune_deberta.ipynb, que clona este repo.

Decisiones clave (ver docs/superpowers/plans/2026-07-05-fase2-deberta.md):
truncado estructurado prompt-cap + head/tail, aumento A<->B, label smoothing 0.1,
TTA al evaluar. Val = fold canónico (data.CANONICAL_FOLD); baseline a batir: 1.0451.
"""
from __future__ import annotations

import numpy as np
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
    if budget < 1:
        # Presupuesto inválido = bug del llamador. Fallar ruidosamente: con budget
        # negativo el slicing de Python devuelve casi toda la secuencia en silencio
        # (nos pasó: OOM de 5.3GB en el smoke test por secuencias sin truncar).
        raise ValueError(f"budget debe ser >= 1, llegó {budget}")
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
    # El cap del prompt escala con max_len (128 = 512//4): si se entrena con una
    # longitud menor (p. ej. smoke con 64), el prompt no puede comerse el presupuesto
    # entero y dejar 'rest' negativo — el bug que causó el OOM del smoke test.
    prompt_budget = min(prompt_budget, max_len // 4)

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


def build_dataset(df: pd.DataFrame, tok, max_len: int = MAX_LEN, with_labels: bool = True):
    """DataFrame -> datasets.Dataset tokenizado (columnas: input_ids, attention_mask[, labels])."""
    from datasets import Dataset

    keep = ["prompt", "response_a", "response_b"] + (["_y"] if with_labels else [])
    ds = Dataset.from_pandas(df[keep], preserve_index=False)

    def encode(ex):
        enc = encode_pair(tok, ex["prompt"], ex["response_a"], ex["response_b"], max_len=max_len)
        if with_labels:
            enc["labels"] = ex["_y"]
        return enc

    return ds.map(encode, remove_columns=keep)


def compute_metrics(eval_pred):
    """Log loss puro sobre las probabilidades (la loss de train lleva smoothing y NO
    es comparable con el baseline; esta métrica sí)."""
    from scipy.special import softmax
    from sklearn.metrics import log_loss

    logits, labels = eval_pred
    return {"log_loss": log_loss(labels, softmax(logits, axis=-1), labels=[0, 1, 2])}


def predict_proba_tta(trainer, tok, df: pd.DataFrame, max_len: int = MAX_LEN) -> np.ndarray:
    """Promedia proba(A,B) con proba(B,A) reordenada — anula el sesgo posicional.

    Al predecir con el orden invertido, la columna 'gana A' del modelo corresponde a
    'gana B' real (y viceversa): por eso el [:, [1, 0, 2]].
    """
    from scipy.special import softmax

    def probs(d):
        ds = build_dataset(d, tok, max_len, with_labels=False)
        return softmax(trainer.predict(ds).predictions, axis=-1)

    p_ab = probs(df)
    p_ba = probs(swap_ab(df))[:, [1, 0, 2]]
    return (p_ab + p_ba) / 2


def main(train_csv: str, out_dir: str = "deberta_out", smoke: bool = False) -> None:
    import data
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    df = data.load_train(train_csv)
    df = data.add_parsed_text(df)
    df = df.assign(_y=data.make_label(df))

    tr_idx, va_idx = data.grouped_folds(df)[data.CANONICAL_FOLD]
    train_df = augment_with_swap(df.iloc[tr_idx])   # aumento SOLO en train
    val_df = df.iloc[va_idx]

    max_len = MAX_LEN
    if smoke:  # smoke: verificar el wiring completo en CPU en ~2 min, no aprender nada
        train_df, val_df, max_len = train_df.head(64), val_df.head(64), 64

    import torch

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_tr = build_dataset(train_df, tok, max_len)
    ds_va = build_dataset(val_df, tok, max_len)
    # dtype explícito: el checkpoint de deberta-v3-small está guardado en fp16 y
    # transformers v5 conserva el dtype del checkpoint por default ("auto"). La
    # precisión mixta (fp16=True) exige pesos maestros en fp32; con el modelo ya
    # en fp16 el GradScaler truena: "Attempting to unscale FP16 gradients".
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, dtype=torch.float32)
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=2e-5, warmup_ratio=0.06, weight_decay=0.01,
        num_train_epochs=2,
        per_device_train_batch_size=8, gradient_accumulation_steps=4,  # efectivo 32
        per_device_eval_batch_size=32,
        label_smoothing_factor=0.1,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="log_loss",
        greater_is_better=False,
        logging_steps=100, report_to="none", seed=42,
        max_steps=8 if smoke else -1,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out_dir)

    proba = predict_proba_tta(trainer, tok, val_df, max_len)
    from sklearn.metrics import log_loss
    ll = log_loss(val_df["_y"], proba, labels=[0, 1, 2])
    print(f"\nlog loss en fold canónico CON TTA = {ll:.4f}  (baseline a batir: 1.0451)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/train.csv")
    ap.add_argument("--out-dir", default="deberta_out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.train_csv, a.out_dir, a.smoke)
