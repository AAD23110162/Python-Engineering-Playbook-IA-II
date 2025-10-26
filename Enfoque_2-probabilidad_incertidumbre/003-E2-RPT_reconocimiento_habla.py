"""
003-E2-RPT_reconocimiento_habla.py
--------------------------------
Este script implementa un sistema simplificado de Reconocimiento Probabilístico del Habla:
- Modela secuencias de fonemas usando Modelos Ocultos de Markov (HMM)
- Calcula la probabilidad de secuencias de observaciones acústicas
- Implementa el algoritmo de Viterbi para encontrar la secuencia más probable de estados
- Reconoce palabras a partir de patrones probabilísticos de señales de audio
- Maneja incertidumbre en las observaciones mediante modelos probabilísticos
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente reconocimiento de palabras predefinidas
2. INTERACTIVO: permite ingresar secuencias de observaciones y decodificar palabras

Autor: Alejandro Aguirre Díaz
"""
