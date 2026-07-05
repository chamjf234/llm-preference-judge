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
| Método (LogReg, C=0.1) | Log loss (validación) |
|---|---|
| Aleatorio (ln 3) | 1.099 |
| TF-IDF texto combinado (ciego a A/B) | 1.083 |
| Solo 6 features numéricas (longitud, A==B, vacías) | 1.072 |
| TF-IDF diferencia tfidf(A)−tfidf(B) | 1.054 |
| TF-IDF diferencia + numéricas | **1.045** |
| DeBERTa-v3 (fine-tuned) | _pendiente_ |

> Dos lecciones del baseline. **(1) La estructura importa:** las features que respetan la
> simetría A/B (diferencia) ganan a las ciegas (combinado). **(2) Los sesgos humanos son señal
> barata:** 6 números sin leer contenido — longitudes (los humanos prefieren la respuesta más
> larga: P(gana A | A más larga) ≈ 0.62), respuestas idénticas (A==B → 90% empate) y vacías —
> casi empatan con 50k features de TF-IDF. Lo que el baseline *no* puede medir es calidad y
> coherencia: ese es el trabajo de DeBERTa en la Fase 2, y ahora tiene un piso honesto que batir.
> Validación con split **agrupado por prompt** (evita fuga entre train y val). Texto **parseado**
> de las listas JSON multi-turno antes de todo (ver EDA sección 0).

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
