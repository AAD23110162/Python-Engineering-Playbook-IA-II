"""
010-E1-busquedasA_AO.py
--------------------------------
Este script implementa los algoritmos A* y AO* de búsqueda informada en grafos:
- Para A* : usa f(n) = g(n) + h(n) donde g(n) es el coste acumulado desde el origen y h(n) la heurística estimada al objetivo. :contentReference[oaicite:2]{index=2}
- Para AO* : una variante para grafos AND-OR (ramas de decisiones múltiples).
- El programa muestra por terminal el progreso del algoritmo: nodos abiertos, cerrados, f(n), g(n), h(n), selección del siguiente nodo a expandir.
- Incluye modo DEMO (origen y destino predefinidos) y modo INTERACTIVO (usuario ingresa origen, destino, selección de algoritmo).

Autor: Alejandro Aguirre Díaz
"""

import heapq

def Bus_A (grafo, heuristica, origen, destino):
    """
    Implementación del algoritmo A* (A-estrella).
    :parametro grafo: dict, claves = nodos, valores = lista de tuplas (vecino, coste)
    :parametro heuristica: dict, claves = (nodo, destino) → valor heurístico h(nodo)
    :parametro origen: nodo de inicio
    :parametro destino: nodo objetivo
    :return: lista con camino óptimo y coste total, o (None, None) si no hay camino
    """
    # Estructura de frontera (cola de prioridad) ordenada por f(n) = g(n) + h(n)
    abiertos = []
    # cada entrada en 'abiertos': (f, nodo, camino, g)
    # Inicializamos con el nodo origen: g=0, f = h(origen)
    heapq.heappush(abiertos, (heuristica.get((origen,destino), float('inf')), origen, [origen], 0))  
    # Conjunto de cerrados para no re-expandir nodos ya procesados
    cerrados = set()
    # Diccionario con el mejor coste g conocido hasta cada nodo
    costo_g = {origen: 0}

    # Cabecera de trazas para entender el progreso de A*
    print(f"Iniciando A* desde {origen} hasta {destino}")
    print("Nodo | g(n) | h(n) | f(n) | Camino")

    while abiertos:
        # Seleccionar el nodo con menor f(n) de la frontera
        f_actual, nodo_actual, camino_actual, g_actual = heapq.heappop(abiertos)
        # Si ya fue cerrado (procesado con mejor g), saltar
        if nodo_actual in cerrados:
            continue

        # Calcular heurística y mostrar la fila de estado
        h_actual = heuristica.get((nodo_actual,destino), float('inf'))
        print(f"{nodo_actual} | {g_actual:.2f} | {h_actual:.2f} | {f_actual:.2f} | {camino_actual}")

        # Si alcanzamos el objetivo, retornar el camino y el coste acumulado
        if nodo_actual == destino:
            print("Objetivo alcanzado.")
            return camino_actual, g_actual

        # Mover a cerrados para evitar re-expansión
        cerrados.add(nodo_actual)

        # Explorar aristas salientes del nodo actual
        for (vecino, coste_arista) in grafo.get(nodo_actual, []):
            # Tentativo: coste g hasta el vecino a través del nodo actual
            g_vecino = g_actual + coste_arista
            # Heurística del vecino hacia el destino
            h_vecino = heuristica.get((vecino,destino), float('inf'))
            # Evaluación total f = g + h
            f_vecino = g_vecino + h_vecino

            # Si el vecino ya está en cerrados con mejor o igual g, ignorar
            if vecino in cerrados and g_vecino >= costo_g.get(vecino, float('inf')):
                continue

            # Si encontramos un mejor g para el vecino, actualizar y añadir a frontera
            if g_vecino < costo_g.get(vecino, float('inf')):
                costo_g[vecino] = g_vecino
                heapq.heappush(abiertos, (f_vecino, vecino, camino_actual + [vecino], g_vecino))

    print("No se encontró camino al objetivo.")
    return None, None

def Bus_AO (grafo_or, heuristica, origen, destino):
    """
    Implementación simplificada del algoritmo AO* para grafos AND-OR.
    En este ejemplo, el grafo está tratado como OR-grafo (asumimos sólo elecciones simples) — solo para demostración.
    :parametro grafo_or: dict, claves = nodos, valores = lista de tuplas (vecino, coste)
    :parametro heuristica: dict, igual que antes
    :parametro origen: nodo de inicio
    :parametro destino: nodo objetivo
    :return: lista con camino y coste aproximado
    """
    # Para simplificar, se reutiliza A* (OR) y se indica que podría extenderse a AND-OR
    print("Se ejecuta AO* (modo simplificado equivalente a A*)")
    return Bus_A (grafo_or, heuristica, origen, destino)

def modo_demo(grafo, heuristica):
    """Modo demostrativo predefinido."""
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    algoritmo = 'A*'
    # Mostrar algoritmo y parámetros seleccionados
    print(f"Seleccionado algoritmo {algoritmo}")
    # Ejecutar A* con los parámetros de demo
    camino, coste = Bus_A (grafo, heuristica, origen, destino)
    # Presentar resultado final
    if camino:
        print(f"Camino encontrado: {camino} | Coste total: {coste}")
    else:
        print("No se encontró camino.")

def modo_interactivo(grafo, heuristica):
    """Modo interactivo: usuario elige algoritmo, origen y destino."""
    print("\n--- MODO INTERACTIVO ---")
    print("Algoritmos disponibles: 1) A*   2) AO*")
    # Selección de algoritmo
    op_alg = input("Introduce el número del algoritmo: ").strip()
    if op_alg == '1':
        algoritmo = 'A*'
    elif op_alg == '2':
        algoritmo = 'AO*'
    else:
        print("Opción no válida. Se usará A* por defecto.")
        algoritmo = 'A*'

    # Mostrar nodos disponibles y solicitar parámetros
    print("Nodos disponibles:", list(grafo.keys()))
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validación de existencia de nodos en el grafo
    if origen not in grafo or destino not in grafo:
        print("Error: Uno o ambos nodos no existen.")
        return

    # Despachar ejecución según algoritmo elegido
    if algoritmo == 'A*':
        camino, coste = Bus_A (grafo, heuristica, origen, destino)
    else:
        camino, coste = Bus_AO (grafo, heuristica, origen, destino)

    # Presentar resultado final
    if camino:
        print(f"Camino encontrado: {camino} | Coste total: {coste}")
    else:
        print("No se encontró camino.")

def main():
    # Grafo ponderado de ejemplo (nodos A-F)
    grafo_ejemplo = {
        'A': [('B',2), ('C',5), ('E',1)],
        'B': [('C',2), ('D',4), ('F',10)],
        'C': [('D',1), ('F',3)],
        'D': [('E',2)],
        'E': [('F',2), ('B',3)],
        'F': [('A',2)]
    }
    # Heurística de ejemplo: estimación de coste restante desde cada nodo al objetivo F
    heuristica_ejemplo = {
        ('A','F'):6, ('B','F'):4, ('C','F'):3, ('D','F'):2, ('E','F'):1, ('F','F'):0
    }

    # Menú de selección de modo de ejecución
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()

    # Ejecutar el modo elegido
    if opcion == '1':
        modo_demo(grafo_ejemplo, heuristica_ejemplo)
    elif opcion == '2':
        modo_interactivo(grafo_ejemplo, heuristica_ejemplo)
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()