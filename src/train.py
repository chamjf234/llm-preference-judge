"""Fine-tuning de DeBERTa-v3 como clasificador de 3 clases (Fase 2).

Se desarrolla y corre en un notebook de Kaggle (ahí está la GPU); este módulo guarda
la versión limpia y reutilizable del código.

Plan:
    - AutoTokenizer / AutoModelForSequenceClassification("microsoft/deberta-v3-small",
      num_labels=3)
    - Tokenizar el texto de format_pair con estrategia de truncado (decisión clave).
    - HuggingFace Trainer, métrica de validación = log loss.
    - Guardar el mejor modelo y subirlo a HuggingFace Hub (no al repo git).
"""
