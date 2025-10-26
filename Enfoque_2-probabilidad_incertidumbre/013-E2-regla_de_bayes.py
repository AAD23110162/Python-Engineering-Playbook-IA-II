"""
013-E2-regla_de_bayes.py
--------------------------------
Este script implementa la Regla de Bayes:
- Aplica el Teorema de Bayes: P(H|E) = P(E|H)·P(H) / P(E)
- Calcula probabilidades a posteriori a partir de verosimilitud y probabilidad a priori
- Implementa clasificadores bayesianos ingenuos (Naive Bayes)
- Actualiza creencias mediante observación de nueva evidencia
- Aplica la regla de Bayes en diagnóstico médico, filtrado de spam y clasificación
- Muestra cómo la evidencia modifica las probabilidades previas
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de aplicación del teorema de Bayes
2. INTERACTIVO: permite al usuario ingresar probabilidades y calcular posteriores

Autor: Alejandro Aguirre Díaz
"""
