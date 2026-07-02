"""Baseline clásico: TF-IDF + Regresión Logística (Fase 1).

Objetivo: fijar un PISO honesto de log loss contra el cual medir si el transformer
realmente aporta. Se completa en la Fase 1 sobre datos reales.

Plan:
    1. format_pair(prompt, response_a, response_b)  -> texto (ver src/data.py)
    2. TfidfVectorizer  -> matriz numérica
    3. LogisticRegression(multi_class="multinomial")  -> 3 clases
    4. Evaluar con sklearn.metrics.log_loss sobre un hold-out estratificado.
"""
