"""
002-E1-busqueda_costo_uniforme.py
--------------------------------
Este script implementa el algoritmo de Búsqueda de Costo Uniforme (Uniform Cost Search, UCS)
sobre un grafo ponderado:
 - Encuentra el camino de menor coste entre un nodo origen y un nodo destino.
 - El grafo está representado como diccionario con listas de tuplas (vecino, coste).
 - Incluye dos modos de ejecución:
     1. MODO DEMO: ejemplo automático.
     2. MODO INTERACTIVO: usuario ingresa origen y destino.

Autor: Alejandro Aguirre Díaz
"""

import heapq

def uniform_cost_search(grafo, origen, destino):
    """
    Realiza la búsqueda de costo uniforme (UCS) desde 'origen' hasta 'destino'.
    :parametro grafo: dict, claves = nodos, valores = lista de (vecino, coste)
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :return: (camino, coste_total) o (None, None) si no existe camino
    """
    # Inicializamos la frontera como una cola de prioridad (heap)
    # Cada entrada es una tupla: (coste_acumulado, nodo_actual, camino_hasta_ahora)
    frontera = [(0, origen, [origen])]
    # 'mejor_coste' guarda el coste mínimo conocido para llegar a cada nodo
    mejor_coste = {origen: 0}

    while frontera:
        coste, nodo, camino = heapq.heappop(frontera)
        # Si este elemento extraído del heap tiene un coste mayor que el mejor
        # coste conocido para 'nodo', lo ignoramos (entrada obsoleta).
        if coste > mejor_coste.get(nodo, float('inf')):
            continue

        # Si hemos llegado al destino devolvemos el camino y el coste.
        # En UCS, la primera vez que extraemos el destino desde la frontera
        # con el menor coste, ese resultado es óptimo.
        if nodo == destino:
            return camino, coste

        # Expandir los vecinos del nodo actual
        for (vecino, peso) in grafo.get(nodo, []):
            nuevo_coste = coste + peso
            # Si encontramos un coste más barato para llegar al vecino,
            # actualizamos el registro y añadimos la nueva entrada a la frontera.
            if nuevo_coste < mejor_coste.get(vecino, float('inf')):
                mejor_coste[vecino] = nuevo_coste
                nuevo_camino = camino + [vecino]
                heapq.heappush(frontera, (nuevo_coste, vecino, nuevo_camino))
    # Si agotamos la frontera y no encontramos destino
    return None, None

def demo_mode(grafo):
    """Ejecuta el modo demostrativo con origen y destino predefinidos."""
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    print(f"Buscando camino de menor coste desde {origen} hasta {destino}...\n")
    camino, coste = uniform_cost_search(grafo, origen, destino)
    if camino:
        print(f"Camino encontrado: {camino}  |  Coste total: {coste}")
    else:
        print(f"No existe camino desde {origen} hasta {destino}.")

def interactive_mode(grafo):
    """Ejecuta el modo interactivo donde el usuario elige origen y destino."""
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles:", list(grafo.keys()))
    print("Ejemplo: A, B, C, D, E, F\n")

    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validación básica: comprueba que ambos nodos están en el grafo
    if origen not in grafo or destino not in grafo:
        print("\n Error: Uno o ambos nodos no existen en el grafo.")
        return

    print(f"\nBuscando camino de menor coste desde {origen} hasta {destino}...\n")
    camino, coste = uniform_cost_search(grafo, origen, destino)
    if camino:
        print(f"Camino encontrado: {camino}  |  Coste total: {coste}")
    else:
        print(f"No existe camino desde {origen} hasta {destino}.")

def main():
    # Grafo ponderado de ejemplo (usando nodos A-F, varias conexiones y distintos costos)
    grafo_ejemplo = {
        'A': [('B', 2), ('C', 5), ('E', 1)],
        'B': [('C', 2), ('D', 4), ('F', 10)],
        'C': [('D', 1), ('F', 3)],
        'D': [('E', 2)],
        'E': [('F', 2), ('B', 3)],
        'F': []
    }

    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (ingresar origen y destino)\n")

    opcion = input("Ingrese el número de opción: ").strip()

    # Ejecuta el modo elegido por el usuario
    if opcion == '1':
        demo_mode(grafo_ejemplo)
    elif opcion == '2':
        interactive_mode(grafo_ejemplo)
    else:
        print("\n Opción no válida. Por favor, intente nuevamente.")

if __name__ == "__main__":
    main()
