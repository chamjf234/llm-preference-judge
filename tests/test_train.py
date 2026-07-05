"""Tests de las funciones puras de src/train.py (truncado, ensamblaje, aumento A/B).

Corren en local sin GPU: el objetivo es no descubrir bugs de lógica a mitad de un
entrenamiento de 2 horas en Kaggle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
