"""
018-E2-eliminacion_de_variables.py
--------------------------------
Este script implementa Eliminación de Variables para inferencia exacta en redes bayesianas:
- Usa factores para representar distribuciones parciales.
- Aplica suma marginal y multiplicación de factores en un orden de eliminación.
- Compara distintas heurísticas de orden (min-degree, min-fill) a nivel conceptual.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo predefinido con trazas de factores por paso.
2. INTERACTIVO: permite cargar factores, escoger orden y ejecutar la eliminación.

Autor: Alejandro Aguirre Díaz
"""
