"""
027-E2-filtros_de_kalman.py
--------------------------------
Este script presenta Filtros de Kalman para modelos lineales-gaussianos:
- Define dinámica lineal y observaciones con ruido gaussiano.
- Implementa el ciclo de predicción-actualización de Kalman a nivel conceptual.
- Discute variantes: Kalman Extendido (EKF) y Unscented (UKF).
- Muestra ejemplos con seguimiento de posición/velocidad en 1D.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: trayectoria sintética con ruido y estimación de estado.
2. INTERACTIVO: permite ajustar matrices A, H, Q, R y estados iniciales.

Autor: Alejandro Aguirre Díaz
"""
