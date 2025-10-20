"""
011-E1-asscension_colinas.py
--------------------------------
Este script implementa el algoritmo de Ascenso de Colinas (Hill Climbing):
- Parte de un estado inicial y en cada iteración selecciona un “vecino” que mejore el valor de la función objetivo.
- Termina cuando no encuentra un vecino mejor (óptimo local) o alcanza un número máximo de iteraciones.
- Incluye dos modos de ejecución:
    1. MODO DEMO: usa estado inicial predefinido y muestra paso a paso por terminal.
    2. MODO INTERACTIVO: el usuario define el estado inicial y parámetros básicos.

Autor: Alejandro Aguirre Díaz
"""

import random

def funcion_objetivo(estado):
    """
    Define la función que queremos maximizar o minimizar.
    En este ejemplo, es una función simple sobre un valor numérico.
    :parametro estado: número (int o float)
    :return: valor de la función objetivo (float)
    """
    # Ejemplo: queremos maximizar f(x) = -(x-5)^2 + 10  (máximo en x=5)
    # Esta función es una parábola invertida centrada en 5
    return - (estado - 5) ** 2 + 10

def generar_vecinos(estado, paso=1.0):
    """
    Genera los estados vecinos del estado actual.
    :parametro estado: número actual
    :parametro paso: magnitud del cambio para vecinos
    :return: lista de estados vecinos
    """
    # Vecinos adyacentes: mover a la derecha (+paso) y a la izquierda (-paso)
    return [estado + paso, estado - paso]

def ascenso_colinas(estado_inicial, max_iteraciones=100, paso=1.0):
    """
    Ejecuta el algoritmo de ascenso de colinas.
    :parametro estado_inicial: valor de partida
    :parametro max_iteraciones: número máximo de iteraciones permitidas
    :parametro paso: magnitud de cambio entre vecinos
    :return: (mejor_estado, valor_objetivo)
    """
    # Inicializar estado y evaluar función objetivo en el estado actual
    estado_actual = estado_inicial
    valor_actual = funcion_objetivo(estado_actual)

    # Iterar hasta alcanzar el máximo de iteraciones
    for iteracion in range(1, max_iteraciones + 1):
        # Generar candidatos vecinos alrededor del estado actual
        vecinos = generar_vecinos(estado_actual, paso)

        # Inicializar el mejor vecino como inexistente y el mejor valor como el actual
        mejor_vecino = None
        mejor_valor = valor_actual

        # Evaluar cada vecino y quedarse con el que maximiza la función objetivo
        for v in vecinos:
            valor_v = funcion_objetivo(v)
            if valor_v > mejor_valor:
                mejor_valor = valor_v
                mejor_vecino = v

        # Reportar el estado de la iteración actual
        print(f"Iteración {iteracion}: estado = {estado_actual:.2f}, valor = {valor_actual:.2f}")

        # Si no hay mejora (ningún vecino supera el valor actual), parar: óptimo local o meseta
        if mejor_vecino is None:
            print("No se encontró vecino mejor. Se alcanza óptimo local o meseta.")
            break

        # Actualizar estado actual al mejor vecino encontrado
        estado_actual = mejor_vecino
        valor_actual = mejor_valor

    # Mostrar resultado final tras terminar iteraciones o alcanzar óptimo local
    print(f"Resultado final: estado = {estado_actual:.2f}, valor = {valor_actual:.2f}")
    return estado_actual, valor_actual

def modo_demo():
    """Modo demostrativo con valores fijos."""
    print("\n--- MODO DEMO ---")
    estado_inicial = 0.0
    max_iter = 10
    paso = 1.0
    # Mostrar configuración inicial del modo demo
    print(f"Estado inicial = {estado_inicial}, max_iter = {max_iter}, paso = {paso}\n")
    ascenso_colinas(estado_inicial, max_iter, paso)

def modo_interactivo():
    """Modo interactivo donde el usuario define parámetros."""
    print("\n--- MODO INTERACTIVO ---")
    try:
        # Solicitar estado inicial al usuario
        estado_inicial = float(input("Introduce el estado inicial (número): ").strip())
    except ValueError:
        print("Entrada inválida. Se usará 0.0")
        estado_inicial = 0.0
    try:
        # Solicitar número máximo de iteraciones
        max_iter = int(input("Introduce el número máximo de iteraciones: ").strip())
        if max_iter < 1:
            raise ValueError
    except ValueError:
        print("Entrada inválida. Se usará 100")
        max_iter = 100
    try:
        # Solicitar tamaño del paso para generar vecinos
        paso = float(input("Introduce el tamaño del paso para vecinos: ").strip())
        if paso <= 0:
            raise ValueError
    except ValueError:
        print("Entrada inválida. Se usará 1.0")
        paso = 1.0

    # Resumen de configuración elegida por el usuario
    print(f"\nEstado inicial = {estado_inicial}, max_iter = {max_iter}, paso = {paso}\n")
    ascenso_colinas(estado_inicial, max_iter, paso)

def main():
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()

    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
