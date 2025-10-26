"""
006-E2-tratamiento_probabilistico_lenguaje.py
--------------------------------
Este script implementa Tratamiento Probabilístico del Lenguaje Natural:
- Construye modelos de lenguaje basados en n-gramas (unigramas, bigramas, trigramas)
- Calcula probabilidades de secuencias de palabras usando el modelo de Markov
- Implementa suavizado (Laplace, Good-Turing) para manejar n-gramas no vistos
- Genera texto automáticamente usando probabilidades de transición
- Aplica el modelo para corrección ortográfica y predicción de palabras
- Evalúa perplejidad del modelo sobre corpus de prueba
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente construcción de modelo y generación de texto predefinido
2. INTERACTIVO: permite entrenar modelos con corpus personalizados y generar texto

Autor: Alejandro Aguirre Díaz
"""
