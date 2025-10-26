"""
020-E2-ponderacion_de_verosimilitud.py
--------------------------------
Este script implementa Ponderación de Verosimilitud (Likelihood Weighting) para inferencia aproximada:
- Genera muestras fijando la evidencia y ponderándolas por su verosimilitud.
- Reduce el problema del rechazo cuando la evidencia es poco probable.
- Estima distribuciones posteriori a partir de pesos normalizados.
- Compara varianza frente a muestreo por rechazo en distintos escenarios.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo predefinido para comparación de estimadores con y sin ponderación.
2. INTERACTIVO: permite definir evidencia y número de muestras para estimación.

Autor: Alejandro Aguirre Díaz
"""
