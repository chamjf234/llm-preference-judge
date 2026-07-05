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
| Método | Log loss (validación) |
|---|---|
| Aleatorio (ln 3) | 1.099 |
| TF-IDF + LogReg — texto combinado (ciego a A/B) | 1.119 |
| TF-IDF + LogReg — diferencia tfidf(A)−tfidf(B) | **1.092** |
| DeBERTa-v3 (fine-tuned) | _pendiente_ |

> El baseline clásico apenas supera el azar (+0.007), y solo cuando las features respetan la
> estructura A/B del problema. Bag-of-words no "entiende" calidad ni coherencia — que es lo que
> decide la preferencia humana. De ahí el salto a un transformer en la Fase 2.
> Validación con split **agrupado por prompt** (evita fuga entre train y val).

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
