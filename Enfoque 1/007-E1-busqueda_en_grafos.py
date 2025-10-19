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
    # Inicializar conjunto de nodos visitados
    visitados = set()
    # Inicializar diccionario de padres para reconstruir caminos (origen no tiene padre)
    padres = {origen: None}
    # Inicializar cola para búsqueda en anchura (BFS)
    cola = deque([origen])
    # Marcar el origen como visitado
    visitados.add(origen)

    # Continuar mientras haya nodos en la cola por explorar
    while cola:
        # Extraer el siguiente nodo de la cola (FIFO)
        nodo_actual = cola.popleft()
        # Explorar todos los vecinos del nodo actual
        for vecino in grafo.get(nodo_actual, []):
            # Si el vecino no ha sido visitado aún
            if vecino not in visitados:
                # Marcar el vecino como visitado
                visitados.add(vecino)
                # Registrar que llegamos a este vecino desde nodo_actual
                padres[vecino] = nodo_actual
                # Agregar el vecino a la cola para explorar sus vecinos posteriormente
                cola.append(vecino)
    # Retornar el diccionario de padres para reconstruir caminos
    return padres

def reconstruir_camino(padres, origen, destino):
    """
    Reconstruye el camino desde origen hasta destino usando el diccionario padres.
    :parametro padres: dict, cada nodo mapea a su padre
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :return: lista con el camino o None si destino no es alcanzable
    """
    # Verificar si el destino fue alcanzado durante la búsqueda
    if destino not in padres:
        return None
    
    # Inicializar lista para construir el camino
    camino = []
    # Comenzar desde el destino y retroceder usando los padres
    nodo = destino
    # Retroceder hasta llegar al origen (que tiene padre None)
    while nodo is not None:
        # Agregar el nodo actual al camino
        camino.append(nodo)
        # Moverse al nodo padre
        nodo = padres[nodo]
    
    # Invertir el camino para que vaya de origen a destino
    camino.reverse()
    
    # Verificar que el camino realmente comienza en el origen
    if camino[0] == origen:
        return camino
    # Si no comienza en origen, algo salió mal
    return None

def modo_demo(grafo):
    # Ejecuta el modo demostrativo con valores predefinidos.
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    # Informar parámetros de la búsqueda
    print(f"Explorando grafo desde {origen} y reconstruyendo camino hasta {destino}...\n")
    # Ejecutar búsqueda en el grafo para obtener árbol de expansión
    padres = busqueda_en_grafos(grafo, origen)
    # Reconstruir el camino específico desde origen hasta destino
    camino = reconstruir_camino(padres, origen, destino)
    # Mostrar resultado según se encuentre camino o no
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No se encontró camino desde {origen} hasta {destino}.")

def modo_interactivo(grafo):
    # Ejecuta el modo interactivo donde el usuario elige origen y destino.
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles:", list(grafo.keys()))
    # Solicitar nodos de origen y destino al usuario
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validar que el nodo de origen exista en el grafo
    if origen not in grafo:
        print(f" Nodo de origen {origen} no existe en el grafo.")
        return
    # Validar que el nodo de destino exista en el grafo
    if destino not in grafo:
        print(f" Nodo de destino {destino} no existe en el grafo.")
        return

    # Informar parámetros y ejecutar búsqueda
    print(f"\nExplorando grafo desde {origen} y reconstruyendo camino hasta {destino}...\n")
    # Ejecutar búsqueda en el grafo para obtener árbol de expansión
    padres = busqueda_en_grafos(grafo, origen)
    # Reconstruir el camino específico desde origen hasta destino
    camino = reconstruir_camino(padres, origen, destino)
    # Mostrar resultado según se encuentre camino o no
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No se encontró camino desde {origen} hasta {destino}.")

def main():
    # Definir grafo de ejemplo (nodos A-F)
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['D', 'E'],
        'C': ['A', 'F'],
        'D': ['E', 'F'],
        'E': ['B', 'F'],
        'F': ['C']
    }

    # Solicitar modo de ejecución al usuario
    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (elegir origen y destino)\n")

    opcion = input("Ingrese el número de opción: ").strip()
    # Ejecutar el modo correspondiente según la opción elegida
    if opcion == '1':
        # Ejecutar modo demostrativo con valores predefinidos
        modo_demo(grafo_ejemplo)
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario elige los nodos
        modo_interactivo(grafo_ejemplo)
    else:
        # Manejo de opción inválida
        print("\n Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()

