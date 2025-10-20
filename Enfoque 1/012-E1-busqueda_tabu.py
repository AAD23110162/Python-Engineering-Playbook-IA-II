"""
012-E1-busqueda_tabu.py
--------------------------------
Este script implementa el algoritmo de Búsqueda Tabú (Tabu Search):
- Parte de una solución inicial y explora su vecindario iterativamente.
- Utiliza una lista tabú para evitar volver a soluciones recientes y favorecer exploración.
- Incorpora mecanismos de diversificación y memoria adaptativa (versión simplificada).
- Incluye dos modos de ejecución:
    1. MODO DEMO: solución inicial y parámetros fijos, muestra progreso por terminal.
    2. MODO INTERACTIVO: usuario define solución inicial, tamaño vecindario, número de iteraciones, tamaño lista tabú.

Autor: Alejandro Aguirre Díaz
"""

import random

def generar_vecindario(solucion_actual, tamaño_vecindario=10, rango_variacion=1):
    """
    Genera un conjunto de soluciones vecinas de la solución actual.
    :parametro solucion_actual: lista o estructura que representa la solución actual
    :parametro tamaño_vecindario: número de vecinos a generar
    :parametro rango_variacion: magnitud de variación para generar vecinos
    :return: lista de soluciones vecinas
    """
    # Lista para acumular los vecinos generados
    vecinos = []
    for _ in range(tamaño_vecindario):
        # Copiar la solución actual para modificarla y crear un vecino
        vecino = solucion_actual.copy()
        # Para este ejemplo, generamos un vecino permutando dos posiciones al azar
        i, j = random.sample(range(len(solucion_actual)), 2)
        vecino[i], vecino[j] = vecino[j], vecino[i]
        # Añadir el vecino generado a la lista de vecindario
        vecinos.append(vecino)
    return vecinos

def evaluar_solucion(solucion):
    """
    Evalúa la función objetivo de la solución dada.
    En este ejemplo: minimizamos la suma de los valores absolutos de diferencias entre elementos.
    :parametro solucion: lista de enteros
    :return: coste (float)
    """
    # Ejemplo sencillo: penaliza dispersión de valores
    # Cuanto más diferentes sean elementos consecutivos, mayor será el coste
    return sum(abs(solucion[i] - solucion[i+1]) for i in range(len(solucion)-1))

def busqueda_tabu(solucion_inicial, max_iteraciones=100, tamaño_vecindario=10, tamaño_tabu=5):
    """
    Ejecuta el algoritmo de búsqueda tabú.
    :parametro solucion_inicial: lista que representa la solución de partida
    :parametro max_iteraciones: número máximo de iteraciones
    :parametro tamaño_vecindario: número de vecinos que se evaluarán por iteración
    :parametro tamaño_tabu: longitud de la lista tabú
    :return: mejor_solucion encontrada y coste
    """
    # Estado actual y mejor solución inicializados con la solución de partida
    solucion_actual = solucion_inicial.copy()
    mejor_solucion = solucion_actual.copy()
    # Evaluar la solución inicial para establecer el mejor coste conocido
    mejor_coste = evaluar_solucion(mejor_solucion)
    # Lista tabú: memoria de soluciones recientes para evitar ciclos y fomentar exploración
    lista_tabu = []

    # Cabecera de trazas para observar progreso por iteración
    print("Iter | coste_actual | mejor_coste | solución_actual")
    for iter in range(1, max_iteraciones + 1):
        # Generar vecindario de la solución actual
        vecinos = generar_vecindario(solucion_actual, tamaño_vecindario)
        # Escoger el mejor vecino que no esté en lista tabú
        mejor_vecino = None
        mejor_vecino_coste = float('inf')
        for v in vecinos:
            # Si la solución vecina está marcada como tabú, omitirla
            if v in lista_tabu:
                continue
            # Evaluar coste de la solución vecina
            coste_v = evaluar_solucion(v)
            if coste_v < mejor_vecino_coste:
                mejor_vecino = v
                mejor_vecino_coste = coste_v

        if mejor_vecino is None:
            print(f"Iteración {iter}: no hay vecino viable fuera de tabú, terminando.")
            break

        # Moverse al mejor vecino encontrado
        solucion_actual = mejor_vecino
        coste_actual = mejor_vecino_coste

        # Actualizar lista tabú (FIFO): añadir solución actual y mantener tamaño máximo
        lista_tabu.append(solucion_actual)
        if len(lista_tabu) > tamaño_tabu:
            lista_tabu.pop(0)

        # Actualizar mejor solución si la actual mejora el mejor coste conocido
        if coste_actual < mejor_coste:
            mejor_solucion = solucion_actual.copy()
            mejor_coste = coste_actual

        print(f"{iter} | {coste_actual:.2f} | {mejor_coste:.2f} | {solucion_actual}")

    print(f"\nMejor solución final: {mejor_solucion} | coste = {mejor_coste:.2f}")
    return mejor_solucion, mejor_coste

def modo_demo():
    """Modo demostrativo con parámetros fijos."""
    print("\n--- MODO DEMO ---")
    solucion_inicial = [5, 1, 4, 2, 3]
    max_iter = 50
    tamaño_vec = 8
    tamaño_tabu = 3
    # Mostrar configuración del modo demo
    print(f"Solución inicial = {solucion_inicial}, max_iter = {max_iter}, tamaño_vecindario = {tamaño_vec}, tamaño_tabu = {tamaño_tabu}\n")
    busqueda_tabu(solucion_inicial, max_iter, tamaño_vec, tamaño_tabu)

def modo_interactivo():
    """Modo interactivo donde el usuario define parámetros."""
    print("\n--- MODO INTERACTIVO ---")
    # Solicitar solución inicial como lista separada por comas
    entrada = input("Introduce la solución inicial como lista de números separados por comas (ej: 5,1,4,2,3): ").strip()
    try:
        solucion_inicial = [int(x) for x in entrada.split(",")]
    except ValueError:
        print("Entrada inválida. Se usará [5,1,4,2,3]")
        solucion_inicial = [5,1,4,2,3]
    try:
        # Solicitar parámetros de ejecución: iteraciones, vecindario y tamaño de lista tabú
        max_iter = int(input("Introduce número máximo de iteraciones: ").strip())
        tamaño_vec = int(input("Introduce tamaño del vecindario: ").strip())
        tamaño_tabu = int(input("Introduce tamaño de la lista tabú: ").strip())
    except ValueError:
        print("Parámetros inválidos, se usarán valores por defecto.")
        max_iter = 50
        tamaño_vec = 8
        tamaño_tabu = 3

    # Resumen de la configuración establecida por el usuario
    print(f"\nSolución inicial = {solucion_inicial}, max_iter = {max_iter}, tamaño_vecindario = {tamaño_vec}, tamaño_tabu = {tamaño_tabu}\n")
    busqueda_tabu(solucion_inicial, max_iter, tamaño_vec, tamaño_tabu)

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
