"""
013-E1-temple_simulado.py
--------------------------------
Este script implementa el algoritmo de Temple Simulado (Simulated Annealing):
- Parte de una solución inicial y, a través de una "temperatura" que disminuye gradualmente, acepta mejoras o incluso empeoramientos con cierta probabilidad.
- Muestra por terminal la evolución de la solución, su coste, la temperatura actual y la decisión de aceptación de nuevos estados.
- Incluye dos modos de ejecución:
    1. MODO DEMO: con valores fijos de inicio, temperatura inicial, factor de enfriamiento.
    2. MODO INTERACTIVO: el usuario define solución inicial, temperatura, factor de enfriamiento, número máximo de iteraciones.

Autor: Alejandro Aguirre Díaz
"""

import math
import random

def funcion_objetivo(estado):
    """
    Función que queremos minimizar o maximizar (en este ejemplo minimizamos).
    :paramero estado: valor numérico (float o int)
    :return: coste del estado (float)
    """
    # Ejemplo sencillo: f(x) = (x-5)^2 → mínimo en x=5
    # Esta función parabólica representa el coste; queremos minimizarlo
    return (estado - 5) ** 2

def generar_vecino(estado, paso=1.0):
    """
    Genera un estado vecino a partir del estado actual.
    :parametro estado: valor actual
    :parametro paso: magnitud del cambio
    :return: nuevo estado vecino
    """
    # Variar el estado de forma aleatoria en el rango [-paso, paso]
    return estado + random.uniform(-paso, paso)

def temple_simulado(estado_inicial, temperatura_inicial=100.0, factor_enfriamiento=0.95, paso=1.0, max_iteraciones=100):
    """
    Ejecuta el algoritmo de Temple Simulado.
    :parametro estado_inicial: valor de partida
    :parametro temperatura_inicial: valor de la temperatura inicial
    :parametro factor_enfriamiento: factor (<1) para reducir la temperatura cada iteración
    :parametro paso: magnitud del cambio para generar vecinos
    :parametro max_iteraciones: número máximo de iteraciones
    :return: mejor_estado encontrado y su coste
    """
    # Inicializar estado actual y evaluar su coste
    estado_actual = estado_inicial
    coste_actual = funcion_objetivo(estado_actual)
    # Guardar el mejor estado y coste encontrados hasta el momento
    mejor_estado = estado_actual
    mejor_coste = coste_actual
    # Inicializar la temperatura de control
    temperatura = temperatura_inicial

    # Cabecera de trazas para seguir la evolución del algoritmo
    print("Iter | Temp | Estado actual | Coste actual | Mejor estado | Mej. coste")
    for iteracion in range(1, max_iteraciones + 1):
        # Generar vecino aleatorio y calcular su coste
        vecino = generar_vecino(estado_actual, paso)
        coste_vecino = funcion_objetivo(vecino)
        # Diferencia de coste (negativa si mejora)
        delta = coste_vecino - coste_actual

        # decisión de aceptación
        if delta < 0:
            # Si el vecino mejora (menor coste), se acepta siempre
            acepta = True
        else:
            # Si empeora, se acepta con probabilidad e^{-delta/T}
            prob = math.exp(-delta / temperatura)
            acepta = random.random() < prob

        # mostrar información
        print(f"{iteracion:4d} | {temperatura:7.2f} | {estado_actual:12.4f} | {coste_actual:12.4f} | {mejor_estado:12.4f} | {mejor_coste:12.4f} | {'Acepta' if acepta else 'Rechaza'}")

        # actualizar estado actual si se acepta
        if acepta:
            estado_actual = vecino
            coste_actual = coste_vecino

        # actualizar mejor encontrado
        if coste_actual < mejor_coste:
            mejor_estado = estado_actual
            mejor_coste = coste_actual

        # reducir temperatura
        temperatura *= factor_enfriamiento

        # opción de parada temprana si la temperatura es muy baja
        if temperatura < 1e-3:
            print("Temperatura muy baja, terminando antes del límite.")
            break

    print(f"\nResultado final: mejor estado = {mejor_estado:.4f} | mejor coste = {mejor_coste:.4f}")
    return mejor_estado, mejor_coste

def modo_demo():
    """Modo demostrativo con parámetros fijos."""
    # Informar que se ejecuta el modo demo
    print("\n--- MODO DEMO ---")
    # Definir parámetros fijos para una ejecución reproducible
    estado_inicial = 0.0
    temperatura0 = 100.0
    enfriamiento = 0.90
    paso = 1.0
    max_iter = 50
    print(f"Estado inicial = {estado_inicial}, temp inicial = {temperatura0}, factor enfriamiento = {enfriamiento}, paso = {paso}, max_iter = {max_iter}\n")
    # Llamar al algoritmo principal con estos parámetros
    temple_simulado(estado_inicial, temperatura0, enfriamiento, paso, max_iter)

def modo_interactivo():
    """Modo interactivo donde el usuario define parámetros."""
    # Informar que se ejecuta el modo interactivo
    print("\n--- MODO INTERACTIVO ---")
    try:
        # Leer estado inicial del usuario
        estado_inicial = float(input("Introduce el estado inicial (número): ").strip())
    except ValueError:
        print("Entrada inválida. Se usará 0.0")
        estado_inicial = 0.0
    try:
        # Leer temperatura inicial
        temperatura0 = float(input("Introduce la temperatura inicial: ").strip())
    except ValueError:
        print("Entrada inválida. Se usará 100.0")
        temperatura0 = 100.0
    try:
        # Leer el factor de enfriamiento y validar que esté en (0,1)
        enfriamiento = float(input("Introduce el factor de enfriamiento (ejemplo 0.9): ").strip())
        if not (0 < enfriamiento < 1):
            raise ValueError
    except ValueError:
        print("Entrada inválida. Se usará 0.90")
        enfriamiento = 0.90
    try:
        # Leer el tamaño del paso para generar vecinos
        paso = float(input("Introduce el tamaño del paso para generar vecinos: ").strip())
    except ValueError:
        print("Entrada inválida. Se usará 1.0")
        paso = 1.0
    try:
        # Leer el máximo de iteraciones
        max_iter = int(input("Introduce número máximo de iteraciones: ").strip())
    except ValueError:
        print("Entrada inválida. Se usará 100")
        max_iter = 100

    print(f"\nEstado inicial = {estado_inicial}, temp inicial = {temperatura0}, factor enfriamiento = {enfriamiento}, paso = {paso}, max_iter = {max_iter}\n")
    # Ejecutar el algoritmo con los parámetros elegidos por el usuario
    temple_simulado(estado_inicial, temperatura0, enfriamiento, paso, max_iter)

def main():
    # Menú sencillo para seleccionar el modo de ejecución
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()

    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        # Manejo de opción incorrecta
        print("Opción no válida.")

if __name__ == "__main__":
    main()
