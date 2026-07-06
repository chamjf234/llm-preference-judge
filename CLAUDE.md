# llm-preference-judge — Contexto del proyecto

## Qué es
Proyecto de **AI engineering** para el portafolio. Entrena un "juez de preferencia" que, dado un
`prompt` y dos respuestas de LLMs (`response_a`, `response_b`), predice qué respuesta prefiere un
humano: **gana A / gana B / empate** (clasificación de 3 clases).

Basado en la competición de Kaggle **LLM Classification Finetuning** (datos de Chatbot Arena /
LMSYS). Conceptualmente es un **reward/preference model** — el componente central de RLHF.

- Competición: https://www.kaggle.com/competitions/llm-classification-finetuning
- Métrica oficial: **log loss multiclase** (importa la probabilidad calibrada, no solo la clase).
  Baseline aleatorio ≈ ln(3) ≈ 1.0986.

## Datos (esquema)
- `train.csv` (~57k filas): `id, model_a, model_b, prompt, response_a, response_b,
  winner_model_a, winner_model_b, winner_tie` (las 3 últimas son one-hot del target).
- `test.csv`: `id, prompt, response_a, response_b` (sin nombres de modelo ni target).
- Los datos NO se versionan en git — se descargan de Kaggle a `data/` (ver `.gitignore`).

## Decisiones tomadas (y el porqué)
- **Modelo:** DeBERTa-v3 (encoder) como clasificador de 3 clases primero. Encoder = "lector/juez",
  encaja con clasificación; pequeño, entrena rápido, bajo riesgo en GPU gratis. Stretch: LoRA sobre
  un decoder pequeño (Llama-3.2 / Qwen2.5) para una historia de progresión "encoder → decoder+LoRA".
- **Compute:** entrenamiento en **notebooks de Kaggle** (GPU gratis, datos ya montados, submission
  directa). El código se exporta a `src/` para no dejar solo notebooks desordenados.
- **Pesos entrenados → HuggingFace Hub**, NO al repo git (regla del portafolio: no versionar
  artefactos pesados). La app los carga desde HF Hub.
- **App:** Gradio en HuggingFace Spaces (alternativa: Streamlit Cloud, que ya conoce del proyecto
  de forecasting). Decisión final en Fase 4.

## Reto técnico #1
Texto **largo**: prompt + 2 respuestas suele exceder el límite de tokens de DeBERTa (~512).
La **estrategia de truncado** es la decisión técnica clave a documentar (truncar respuestas,
head+tail, etc.). Vigilar también los **empates** (clase difícil, que el modelo no la ignore).

## Estado actual
- **Fases 0-1 completas.** Hallazgo clave del EDA: las columnas de texto son LISTAS JSON de
  turnos (13% multi-turno, turnos null) — SIEMPRE parsear con `data.add_parsed_text` antes de
  medir o modelar. Baseline: TF-IDF diferencia + 6 features numéricas de sesgos humanos,
  log loss **1.0445 ± 0.0025** (GroupKFold 5-fold agrupado por prompt).
- **Validación:** fold canónico = `data.CANONICAL_FOLD` (fold 0). Baseline ahí: **1.0451**.
  Toda comparación con transformers se hace sobre ese mismo fold.
- **Fase 2 (en curso):** pipeline DeBERTa completo con TDD (`src/train.py`, 11 tests) —
  truncado head+tail con presupuesto por campo, aumento A↔B, label smoothing, TTA.
  Run 1 en Kaggle (deberta-v3-small, 2 épocas, lr 2e-5): **1.0714** con TTA — no bate al
  baseline; la época 2 sobreajusta (1.073 → 1.134). Plan de la fase:
  `docs/superpowers/plans/2026-07-05-fase2-deberta.md`.
- Gotchas aprendidos: checkpoint de deberta-v3-small está en fp16 → cargar con
  `dtype=torch.float32` explícito (transformers v5 conserva el dtype del checkpoint);
  el Training Loss reportado por el Trainer con 2 GPUs + grad accum viene inflado ~8x
  (suma en vez de promediar) — la métrica fiable es `log_loss` de compute_metrics.
- Siguiente: iterar Fase 2 (lr menor / deberta-v3-base / ensemble con baseline) y
  luego Fase 3 (calibración temperature scaling + submission).

## Convenciones
- Explicaciones en **español**, con el *porqué* de cada decisión no obvia (el usuario está
  aprendiendo AI engineering / software; primer proyecto con transformers).
- Git: modo "mezcla" — Claude ejecuta operaciones rutinarias pero explica qué pasa y por qué.
  No `git add .` / `-A`; agregar archivos por nombre. Nunca force push a main.
