"""
025-E2-algoritmo_hacia_delante_atras.py
--------------------------------
Este script implementa el Algoritmo Hacia Delante-Atrás (Forward-Backward) para HMM:
- Calcula mensajes hacia delante (α) y hacia atrás (β) para suavizado.
- Obtiene creencias marginales por tiempo y la verosimilitud de la secuencia.
- Discute estabilidad numérica (escalado/normalización).
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo con HMM de juguete y trazas de α/β.
2. INTERACTIVO: permite ingresar matrices de transición/emisión y observaciones.

Autor: Alejandro Aguirre Díaz
"""
