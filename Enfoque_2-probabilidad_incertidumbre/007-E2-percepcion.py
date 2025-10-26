"""
007-E2-percepcion.py
--------------------------------
Este script implementa sistemas de Percepción bajo Incertidumbre:
- Modela sensores con ruido y errores de medición probabilísticos
- Implementa filtrado bayesiano para fusionar múltiples lecturas de sensores
- Aplica el Filtro de Kalman para estimar estados a partir de observaciones ruidosas
- Integra información de múltiples sensores mediante fusión probabilística
- Maneja incertidumbre en localización y mapeo (conceptos básicos de SLAM)
- Visualiza la evolución de las creencias sobre el estado del sistema
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente fusión de sensores en escenarios predefinidos
2. INTERACTIVO: permite configurar modelos de sensores y observar el proceso de filtrado

Autor: Alejandro Aguirre Díaz
"""
