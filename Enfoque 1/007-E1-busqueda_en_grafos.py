"""
007-E1-busqueda_en_grafos.py
--------------------------------
Este script implementa una base genérica para la búsqueda en grafos (Graph Search) en un grafo representado como lista de adyacencia.
- Permite explorar todos los nodos alcanzables desde un origen.
- Puede ser adaptado para búsquedas no informadas (BFS, DFS) o informadas.
- Incluye modo DEMO e modo INTERACTIVO.

Autor: Alejandro Aguirre Díaz
"""

from collections import deque

def busqueda_en_grafos(grafo, origen):
    """
    Realiza una búsqueda en el grafo desde el nodo origen.
    Devuelve el conjunto de nodos alcanzables y el árbol de expansión.
    :parametro grafo: dict, claves = nodos, valores = lista de adyacentes
    :parametro origen: nodo inicial
    :return: dict de padres (cada nodo -> nodo padre) para reconstruir caminos
    """
    visitados = set()
    padres = {origen: None}
    cola = deque([origen])
    visitados.add(origen)

    while cola:
        nodo_actual = cola.popleft()
        for vecino in grafo.get(nodo_actual, []):
            if vecino not in visitados:
                visitados.add(vecino)
                padres[vecino] = nodo_actual
                cola.append(vecino)
    return padres

def reconstruir_camino(padres, origen, destino):
    """
    Reconstruye el camino desde origen hasta destino usando el diccionario padres.
    :parametro padres: dict, cada nodo mapea a su padre
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :return: lista con el camino o None si destino no es alcanzable
    """
    if destino not in padres:
        return None
    camino = []
    nodo = destino
    while nodo is not None:
        camino.append(nodo)
        nodo = padres[nodo]
    camino.reverse()
    if camino[0] == origen:
        return camino
    return None

def modo_demo(grafo):
    """Ejecuta el modo demostrativo con valores predefinidos."""
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    print(f"Explorando grafo desde {origen} y reconstruyendo camino hasta {destino}...\n")
    padres = busqueda_en_grafos(grafo, origen)
    camino = reconstruir_camino(padres, origen, destino)
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No se encontró camino desde {origen} hasta {destino}.")

def modo_interactivo(grafo):
    """Ejecuta el modo interactivo donde el usuario elige origen y destino."""
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles:", list(grafo.keys()))
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    if origen not in grafo:
        print(f" Nodo de origen {origen} no existe en el grafo.")
        return
    if destino not in grafo:
        print(f" Nodo de destino {destino} no existe en el grafo.")
        return

    print(f"\nExplorando grafo desde {origen} y reconstruyendo camino hasta {destino}...\n")
    padres = busqueda_en_grafos(grafo, origen)
    camino = reconstruir_camino(padres, origen, destino)
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No se encontró camino desde {origen} hasta {destino}.")

def main():
    # Grafo de ejemplo (nodos A-F)
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['D', 'E'],
        'C': ['A', 'F'],
        'D': ['E', 'F'],
        'E': ['B', 'F'],
        'F': ['C']
    }

    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (elegir origen y destino)\n")

    opcion = input("Ingrese el número de opción: ").strip()
    if opcion == '1':
        modo_demo(grafo_ejemplo)
    elif opcion == '2':
        modo_interactivo(grafo_ejemplo)
    else:
        print("\n Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()

