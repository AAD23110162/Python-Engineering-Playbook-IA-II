"""
031-E2-naive_bayes.py
--------------------------------
Este script implementa el clasificador Naïve Bayes:
- Asume independencia condicional de las características dado la clase.
- Soporta variantes multinomial/bernoulli/gaussiana a nivel conceptual.
- Calcula probabilidades a posteriori y realiza clasificación.
- Discute calibración y manejo de rareza con suavizado de Laplace.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: clasificación en datasets de juguete con desbalance.
2. INTERACTIVO: permite ingresar datos tabulares y evaluar predicciones.

Autor: Alejandro Aguirre Díaz
"""
