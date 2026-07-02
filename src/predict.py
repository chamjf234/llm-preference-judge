"""Inferencia y calibración (Fase 3) + generación de submission.

Plan:
    - Cargar modelo afinado (desde HuggingFace Hub).
    - Predecir probabilidades de las 3 clases.
    - Calibrar (temperature scaling) para bajar el log loss.
    - Escribir submission.csv con columnas: id, winner_model_a, winner_model_b, winner_tie.
"""
