# 🧑‍⚖️ LLM Preference Judge

> Un modelo que predice **qué respuesta de un LLM prefiere un humano**, dado un prompt y dos
> respuestas candidatas. En esencia, un *reward / preference model* — el componente central de
> RLHF (aprendizaje por refuerzo con feedback humano).

Basado en la competición de Kaggle
[**LLM Classification Finetuning**](https://www.kaggle.com/competitions/llm-classification-finetuning)
(datos de Chatbot Arena / LMSYS).

> 🚧 **En construcción.** Este README se irá completando conforme avanza el proyecto.

---

## El problema
Dado un `prompt` y dos respuestas (`response_a`, `response_b`) de LLMs anónimos, predecir la
preferencia humana entre tres clases: **gana A / gana B / empate**. Se evalúa con **log loss**
multiclase, así que no basta con acertar la clase: importa dar **probabilidades bien calibradas**.

## Enfoque
1. **Baseline** — TF-IDF + Regresión Logística (piso de referencia).
2. **Fine-tuning de DeBERTa-v3** — encoder afinado como clasificador de 3 clases.
3. **Calibración** de probabilidades (clave por la métrica log loss).
4. *(Stretch)* **LoRA/QLoRA** sobre un decoder pequeño (Llama-3.2 / Qwen2.5).

## Resultados
| Método (LogReg, C=0.1) | Log loss (5-fold agrupado por prompt, media ± std) |
|---|---|
| Aleatorio (ln 3) | 1.099 |
| TF-IDF texto combinado (ciego a A/B) | 1.083 ± 0.002 |
| Solo 6 features numéricas (longitud, A==B, vacías) | 1.071 ± 0.003 |
| TF-IDF diferencia tfidf(A)−tfidf(B) | 1.054 ± 0.002 |
| TF-IDF diferencia + numéricas | **1.045 ± 0.003** |
| DeBERTa-v3-small — run 1: 2 épocas, lr 2e-5, TTA (fold canónico) | 1.0714 |

> Dos lecciones del baseline. **(1) La estructura importa:** las features que respetan la
> simetría A/B (diferencia) ganan a las ciegas (combinado). **(2) Los sesgos humanos son señal
> barata:** 6 números sin leer contenido — longitudes (los humanos prefieren la respuesta más
> larga: P(gana A | A más larga) ≈ 0.62), respuestas idénticas (A==B → 90% empate) y vacías —
> casi empatan con 50k features de TF-IDF. Lo que el baseline *no* puede medir es calidad y
> coherencia: ese es el trabajo de DeBERTa en la Fase 2, y ahora tiene un piso honesto que batir.
> Validación: **GroupKFold de 5 folds agrupado por prompt** (evita fuga entre train y val;
> la ± std dice cuándo una mejora es real y cuándo es ruido de muestreo). DeBERTa se entrena
> una sola vez sobre el **fold canónico** (fold 0, fijado en `data.CANONICAL_FOLD`), donde el
> mejor baseline marca **1.0451** — ese es el número exacto a batir. Texto **parseado** de las
> listas JSON multi-turno antes de todo (ver EDA sección 0).
>
> **Run 1 de DeBERTa (no bate al baseline aún):** log loss 1.0730 en la época 1 y 1.1341 en
> la 2 (**sobreajuste**; se rescata el mejor checkpoint). Con TTA: **1.0714** ≈ el nivel de las
> features numéricas solas — en una época el modelo aprendió el sesgo de verbosidad pero aún no
> la señal fina de calidad. Siguiente iteración: lr menor y/o más capacidad (`base`), y ensemble
> con el baseline.

## Demo
_Pendiente: enlace a la app en HuggingFace Spaces + captura._

## Estructura del repo
```
notebooks/   Exploración y entrenamiento (versiones Kaggle)
src/         Código reutilizable: datos, baseline, entrenamiento, inferencia
app/         App de demo (Gradio)
data/        Datos de Kaggle (no versionados)
```

## Cómo reproducir
_Pendiente: pasos para descargar datos, entrenar y correr la app._

## Aprendizajes
_Pendiente: notas sobre truncado de texto largo, calibración, y (stretch) LoRA._
