# Fase 2: Fine-tuning de DeBERTa-v3 — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Afinar `microsoft/deberta-v3-small` como clasificador de 3 clases (gana A / gana B / empate) y batir el baseline de **log loss 1.0451** en el fold canónico.

**Architecture:** Funciones puras y testeables en `src/train.py` (truncado head+tail, ensamblaje de tokens, aumento A↔B, TTA), desarrolladas con TDD en local (CPU); el entrenamiento real corre en Kaggle (GPU) vía un notebook delgado que clona este repo. Validación = fold canónico (`data.CANONICAL_FOLD`), el mismo del baseline.

**Tech Stack:** transformers (Trainer), datasets, pytest, DeBERTa-v3-small, Kaggle GPU, HuggingFace Hub para los pesos.

---

## Decisiones de diseño (el porqué)

| Decisión | Elección | Por qué / alternativas |
|---|---|---|
| Modelo | `deberta-v3-small` | Rápido en GPU gratis; si la época sale < 1h, probar `base` después como experimento separado. |
| Longitud máx. | 512 tokens | DeBERTa acepta más (posiciones relativas) pero tiempo/memoria ~cuadrático en atención; primero 512, medir, luego decidir. |
| Truncado | prompt cap 128 (head+tail) + resto repartido entre respuestas, head+tail 75/25 | EDA: el largo vive en las respuestas; el final de una respuesta contiene la conclusión (pesa en la preferencia). El presupuesto no usado por una respuesta corta se cede a la otra. |
| Ensamblaje | a nivel de TOKENS: `[CLS] p [SEP] a [SEP] b [SEP]` | Tokenizar el string formateado impediría presupuestos por campo. DeBERTa-v3 no tiene segment embeddings (type_vocab_size=0): los `[SEP]` son los delimitadores. |
| Aumento | duplicar train con A↔B intercambiados (etiqueta 0↔1) | Dobla datos y elimina el sesgo posicional. Solo el fold de train — val queda intacto. |
| TTA | promediar proba(A,B) y proba(B,A) reordenada | Simetriza la predicción; mejora log loss y calibración casi gratis. |
| Label smoothing | 0.1 | Etiquetas de preferencia son ruidosas (humanos no coinciden entre sí); evita sobre-confianza que el log loss castiga. |
| Hiperparámetros | lr 2e-5, warmup 6%, wd 0.01, 2 épocas, batch efectivo 32, fp16 | Defaults conocidos para fine-tuning de encoders; no se tunean hasta tener la primera medición. |
| Val / métrica | fold canónico, log loss vía `compute_metrics` | La loss de train incluye smoothing → NO es comparable; la métrica de eval es log loss puro contra 1.0451. |

**Prerrequisito del usuario (Task 8):** cuenta de HuggingFace + token de escritura guardado como Kaggle Secret (`HF_TOKEN`), y el repo pusheado a GitHub.

---

### Task 0: Dependencias locales y esqueleto de tests

**Files:**
- Modify: `requirements.txt`
- Create: `tests/test_train.py` (vacío por ahora)

- [ ] **Step 1: Instalar dependencias de desarrollo local**

Run: `pip install transformers sentencepiece datasets accelerate pytest`

(torch CPU ya está instalado; esto añade ~200MB, no GPU. `tiktoken`/`protobuf` pueden venir como deps del tokenizer.)

- [ ] **Step 2: Verificar que el tokenizer de DeBERTa carga**

Run: `python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small'); print(t.cls_token_id, t.sep_token_id, len(t))"`
Expected: tres números sin error (descarga ~2.5MB de spm la primera vez).

- [ ] **Step 3: Fijar versiones en requirements.txt**

Sustituir la sección de transformers en `requirements.txt` por las versiones exactas instaladas (`pip show transformers datasets accelerate sentencepiece | grep -E "Name|Version"`), y añadir sección dev:

```
# === Dev / tests ===
pytest
```

- [ ] **Step 4: Crear tests/test_train.py con el header**

```python
"""Tests de las funciones puras de src/train.py (truncado, ensamblaje, aumento A/B).

Corren en local sin GPU: el objetivo es no descubrir bugs de lógica a mitad de un
entrenamiento de 2 horas en Kaggle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_train.py
git commit -m "Fase 2: deps locales (transformers/pytest) y esqueleto de tests"
```

---

### Task 1: Aumento A↔B (`swap_ab`, `augment_with_swap`)

**Files:**
- Modify: `src/train.py` (reemplaza el stub actual)
- Test: `tests/test_train.py`

- [ ] **Step 1: Escribir los tests que fallan**

Añadir a `tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL / ERROR con `AttributeError: module 'train' has no attribute 'swap_ab'`

- [ ] **Step 3: Implementar en src/train.py**

Reemplazar el contenido stub de `src/train.py` por:

```python
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
```

- [ ] **Step 4: Verificar que pasan**

Run: `python -m pytest tests/test_train.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Fase 2: aumento A<->B con intercambio de etiquetas (TDD)"
```

---

### Task 2: Truncado head+tail

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Tests que fallan**

Añadir a `tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `python -m pytest tests/test_train.py -v -k head_tail`
Expected: FAIL con `AttributeError`

- [ ] **Step 3: Implementar**

Añadir a `src/train.py`:

```python
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
```

- [ ] **Step 4: Verificar que pasan**

Run: `python -m pytest tests/test_train.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Fase 2: truncado head+tail con presupuesto (TDD)"
```

---

### Task 3: Ensamblaje de tokens (`encode_pair`)

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Tests que fallan (con el tokenizer real)**

Añadir a `tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Verificar que fallan**

Run: `python -m pytest tests/test_train.py -v -k encode`
Expected: FAIL con `AttributeError`

- [ ] **Step 3: Implementar**

Añadir a `src/train.py`:

```python
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
```

- [ ] **Step 4: Verificar que pasan**

Run: `python -m pytest tests/test_train.py -v`
Expected: 9 passed

- [ ] **Step 5: Medición real en tokens (promesa del EDA)**

Run:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import data, train
from transformers import AutoTokenizer
import numpy as np

tok = AutoTokenizer.from_pretrained(train.MODEL_NAME)
df = data.add_parsed_text(data.load_train('data/train.csv')).sample(2000, random_state=0)
lens = [len(train.encode_pair(tok, r.prompt, r.response_a, r.response_b)['input_ids'])
        for r in df.itertuples()]
full = [len(tok.encode(r.prompt)) + len(tok.encode(r.response_a)) + len(tok.encode(r.response_b))
        for r in df.itertuples()]
print('tokens SIN truncar p50/p90/p99:', np.percentile(full, [50, 90, 99]).astype(int))
print('% filas truncadas por encode_pair:', np.mean(np.array(full) > 512) * 100)
print('longitud media resultante:', np.mean(lens))
"
```

Expected: imprime percentiles (~50% truncadas, consistente con el EDA) sin errores. Anotar los números en el commit.

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Fase 2: encode_pair con presupuesto por campo y cesion de sobrante (TDD)"
```

---

### Task 4: Dataset HF + `compute_metrics` + `main()` de entrenamiento

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Implementar build_dataset y compute_metrics**

Añadir a `src/train.py`:

```python
import numpy as np


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
```

- [ ] **Step 2: Implementar main() con flag --smoke**

Añadir a `src/train.py`:

```python
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

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_tr = build_dataset(train_df, tok, max_len)
    ds_va = build_dataset(val_df, tok, max_len)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    import torch
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
```

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "Fase 2: main() de entrenamiento con Trainer, label smoothing y fold canonico"
```

---

### Task 5: TTA (`predict_proba_tta`)

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Test que falla (con un "trainer" falso — no necesitamos GPU para probar la LÓGICA del TTA)**

Añadir a `tests/test_train.py`:

```python
import numpy as np


class _FakeTrainer:
    """Devuelve logits fijos: simula un modelo con sesgo posicional puro hacia A."""
    def __init__(self):
        self.calls = []

    def predict(self, ds):
        self.calls.append(len(ds))
        n = len(ds)
        logits = np.tile(np.array([[2.0, 0.0, 0.0]]), (n, 1))  # siempre "gana A"
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
```

- [ ] **Step 2: Verificar que falla**

Run: `python -m pytest tests/test_train.py -v -k tta`
Expected: FAIL con `AttributeError`

- [ ] **Step 3: Implementar**

Añadir a `src/train.py`:

```python
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
```

- [ ] **Step 4: Verificar que pasan todos**

Run: `python -m pytest tests/test_train.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Fase 2: TTA con intercambio A<->B y reordenado de probabilidades (TDD)"
```

---

### Task 6: Smoke test local (CPU) del pipeline completo

**Files:** ninguno nuevo (verificación)

- [ ] **Step 1: Correr el smoke**

Run: `python src/train.py --smoke`
Expected: entrena 8 steps en CPU (~2-4 min), evalúa, imprime `log loss en fold canónico CON TTA = ...` sin excepción. El número será malo (~1.1): el objetivo es verificar el WIRING, no aprender.

- [ ] **Step 2: Commit de cualquier fix que haya salido del smoke**

```bash
git add src/train.py
git commit -m "Fase 2: smoke test local del pipeline completo en CPU"
```

---

### Task 7: Notebook de Kaggle

**Files:**
- Create: `notebooks/02_finetune_deberta.ipynb`

- [ ] **Step 1: Crear el notebook con este script**

```bash
python -c "
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        '# Fase 2 — Fine-tuning DeBERTa-v3-small (Kaggle GPU)\n\n'
        'Notebook DELGADO: la lógica vive en src/ del repo (testeada en local). '
        'Aquí solo: clonar, instalar, entrenar, evaluar con TTA, subir pesos a HF Hub.\n\n'
        '**Setup requerido:** GPU T4/P100 activada; secret HF_TOKEN en Add-ons > Secrets.'),
    nbf.v4.new_code_cell(
        '!pip -q install -U transformers datasets accelerate sentencepiece\n'
        '!git clone https://github.com/chamjf234/llm-preference-judge.git repo\n'
        'import sys; sys.path.insert(0, \'repo/src\')'),
    nbf.v4.new_code_cell(
        'import train\n'
        'train.main(train_csv=\'/kaggle/input/llm-classification-finetuning/train.csv\',\n'
        '           out_dir=\'/kaggle/working/deberta_out\')'),
    nbf.v4.new_code_cell(
        '# Subir pesos a HF Hub (NO al repo git: regla del portafolio)\n'
        'from kaggle_secrets import UserSecretsClient\n'
        'from transformers import AutoModelForSequenceClassification, AutoTokenizer\n'
        'token = UserSecretsClient().get_secret(\'HF_TOKEN\')\n'
        'repo_id = \'chamjf234/llm-preference-judge-deberta-v3-small\'\n'
        'AutoModelForSequenceClassification.from_pretrained(\'/kaggle/working/deberta_out\').push_to_hub(repo_id, token=token)\n'
        'AutoTokenizer.from_pretrained(\'microsoft/deberta-v3-small\').push_to_hub(repo_id, token=token)'),
]
nbf.write(nb, 'notebooks/02_finetune_deberta.ipynb')
print('ok')
"
```

- [ ] **Step 2: Push del repo (el notebook lo clona desde GitHub)**

```bash
git add notebooks/02_finetune_deberta.ipynb
git commit -m "Fase 2: notebook Kaggle delgado (clona el repo, entrena, sube a HF Hub)"
git push origin main
```

- [ ] **Step 3 (USUARIO, en kaggle.com): correr el notebook**

1. Crear notebook en la competición *LLM Classification Finetuning* (así `train.csv` queda montado en `/kaggle/input/`).
2. Subir `notebooks/02_finetune_deberta.ipynb` (File > Import).
3. Settings: GPU T4 x2 o P100; Internet ON (necesario para clonar y para HF Hub).
4. Add-ons > Secrets: `HF_TOKEN` = token de escritura de huggingface.co/settings/tokens.
5. Run all. Estimación: ~1h/época con small; si la primera época tarda >2h, bajar a 1 época o recortar MAX_LEN.

Expected: imprime `log loss en fold canónico CON TTA = X` con X < 1.0451.

---

### Task 8: Resultados al README

**Files:**
- Modify: `README.md` (tabla de resultados + fila DeBERTa)
- Modify: `notebooks/02_finetune_deberta.ipynb` (versión con outputs, exportada de Kaggle)

- [ ] **Step 1: Actualizar la fila `DeBERTa-v3 (fine-tuned)` de la tabla con el log loss del fold canónico (con y sin TTA), y añadir 2-3 líneas de lectura: cuánto batió al baseline y por qué (o por qué no).**

- [ ] **Step 2: Reemplazar el notebook local por la versión ejecutada exportada de Kaggle (File > Download).**

- [ ] **Step 3: Commit**

```bash
git add README.md notebooks/02_finetune_deberta.ipynb
git commit -m "Fase 2: resultados de DeBERTa-v3-small en el fold canonico"
git push origin main
```
