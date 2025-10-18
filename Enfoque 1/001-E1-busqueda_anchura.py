"""
001-E1-busqueda_anchura.py
--------------------------------
Este script implementa el algoritmo de Búsqueda en Anchura (Breadth-First Search, BFS)
sobre un grafo representado como diccionario (lista de adyacencia).

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente un ejemplo predefinido.
2. INTERACTIVO: solicita al usuario que elija un nodo de origen y destino y muestra
    el camino encontrado (si existe).

Autor: Alejandro Aguirre Díaz
"""

from collections import deque

def bfs(grafo, origen, destino):
    """
    Realiza la búsqueda en anchura (BFS) desde 'origen' hasta 'destino' en el grafo.
    :parametro grafo: dict -> llaves = nodos, valores = lista de adyacentes
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :return: lista con el camino encontrado, o None si no existe
    """
    # Caso trivial: si el nodo origen es el mismo que el destino, devolvemos la lista con él
    if origen == destino:
        return [origen]

    # Conjuntos/colas que usa BFS:
    visitados = set()  # evita volver a procesar nodos ya vistos
    # cola: contiene tuplas (nodo_actual, camino_desde_origen)
    cola = deque([(origen, [origen])])  # iniciamos la cola con el origen y el camino que lo contiene
    visitados.add(origen)

    while cola:
        nodo_actual, camino = cola.popleft()
        # Expandimos el siguiente nodo en la cola (orden por capas)
        for vecino in grafo.get(nodo_actual, []):
            # Si no se ha visitado, lo marcamos y añadimos a la cola con el camino actualizado
            if vecino not in visitados:
                nuevo_camino = camino + [vecino]
                # Si el vecino es el destino, devolvemos inmediatamente el camino encontrado
                if vecino == destino:
                    return nuevo_camino
                visitados.add(vecino)
                cola.append((vecino, nuevo_camino))
    return None


def demo_mode(grafo):
    """Ejecuta el modo demostrativo con origen y destino fijos."""
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    print(f"Buscando camino desde {origen} hasta {destino}...\n")
    camino = bfs(grafo, origen, destino)

    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No existe camino desde {origen} hasta {destino}.")


def interactive_mode(grafo):
    """Ejecuta el modo interactivo donde el usuario elige origen y destino."""
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles en el grafo:", list(grafo.keys()))
    print("Ejemplo: A, B, C, D, E, F\n")

    origen = input("Introduce el nodo de ORIGEN (entre las opciones mostradas): ").strip().upper()
    destino = input("Introduce el nodo de DESTINO (entre las opciones mostradas): ").strip().upper()

    # Validación básica: los nodos deben existir en las claves del diccionario
    if origen not in grafo or destino not in grafo:
        print("\n Error: Uno o ambos nodos no existen en el grafo.")
        return

    print(f"\nBuscando camino desde {origen} hasta {destino}...\n")
    camino = bfs(grafo, origen, destino)

    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No existe camino desde {origen} hasta {destino}.")


def main():
    # Grafo de ejemplo
    grafo = {
        'A': ['B', 'C', 'E'],
        'B': ['C', 'D', 'F'],
        'C': ['A', 'D'],
        'D': ['E'],
        'E': ['B', 'F'],
        'F': ['C', 'A']
    }

    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (elegir origen y destino)\n")

    opcion = input("Ingrese el número de opción: ").strip()

    # En función de la opción elegida, arrancamos el modo demo o interactivo
    if opcion == '1':
        demo_mode(grafo)
    elif opcion == '2':
        interactive_mode(grafo)
    else:
        print("\n Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    main()
