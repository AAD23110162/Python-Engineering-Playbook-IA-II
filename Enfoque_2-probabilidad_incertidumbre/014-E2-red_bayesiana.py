"""
014-E2-red_bayesiana.py
--------------------------------
Este script implementa Redes Bayesianas completas:
- Construye grafos acíclicos dirigidos (DAG) para representar dependencias probabilísticas
- Define tablas de probabilidad condicional (CPT) para cada nodo
- Implementa algoritmos de inferencia exacta: eliminación de variables y propagación de creencias
- Realiza inferencia aproximada mediante muestreo (rechazo, ponderación por verosimilitud)
- Aplica inferencia hacia adelante y hacia atrás en la red
- Identifica independencias condicionales mediante d-separación
- Visualiza la estructura de la red y el flujo de información
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente inferencia en redes bayesianas predefinidas (alarma, diagnóstico)
2. INTERACTIVO: permite construir redes personalizadas y realizar consultas de inferencia

Autor: Alejandro Aguirre Díaz
"""
