"""
003-E1-busqueda_profundidad.py
--------------------------------
Este script implementa el algoritmo de Búsqueda en Profundidad (Depth-First Search, DFS) 
sobre un grafo representado como lista de adyacencia.

El programa puede ejecutarse en dos modos:
1. MODO DEMO: ejecuta automáticamente un ejemplo predefinido.
2. MODO INTERACTIVO: solicita al usuario que elija un nodo de origen y destino.

Autor: Alejandro Aguirre Díaz
"""

def dfs(grafo, origen, destino, visitados=None, camino=None):
    """
    Realiza la búsqueda en profundidad (DFS) recursiva desde 'origen' hasta 'destino'.
    :parametro grafo: dict -> llaves = nodos, valores = lista de adyacentes
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :parametro visitados: conjunto de nodos ya visitados
    :parametro camino: lista con el camino actual
    :return: lista con el camino encontrado o None si no existe
    """
    # Inicialización de estructuras auxiliares en la primera llamada
    if visitados is None:
        visitados = set()
    if camino is None:
        camino = []

    # Marcamos el nodo actual como visitado y lo añadimos al camino actual
    visitados.add(origen)
    camino.append(origen)

    # Si alcanzamos el destino devolvemos el camino construido
    if origen == destino:
        return camino

    # Recorremos recursivamente cada vecino no visitado
    for vecino in grafo.get(origen, []):
        if vecino not in visitados:
            # Pasamos una copia del camino para que cada rama recursiva
            # tenga su propio historial (evita efectos colaterales)
            resultado = dfs(grafo, vecino, destino, visitados, camino.copy())
            # Si la recursión encontró el destino, propagamos el resultado
            if resultado:
                return resultado

    # Si agotamos las ramas sin encontrar el destino devolvemos None
    return None


def demo_mode(grafo):
    #Ejecuta el modo demostrativo con origen y destino fijos.
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    # Mostrar información sobre la búsqueda que se va a ejecutar
    print(f"Buscando camino en profundidad desde {origen} hasta {destino}...\n")

    # Llamada a la función DFS para intentar encontrar un camino
    camino = dfs(grafo, origen, destino)

    # Comprobar si DFS devolvió un camino o None y mostrar resultado
    if camino:
        # Si se encontró un camino, imprimir la secuencia de nodos
        print(f"Camino encontrado: {camino}")
    else:
        # Si no se encontró, informar al usuario
        print(f"No existe camino desde {origen} hasta {destino}.")


def interactive_mode(grafo):
    #Ejecuta el modo interactivo donde el usuario elige origen y destino.
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles en el grafo:", list(grafo.keys()))
    print("Ejemplo: A, B, C, D, E, F\n")

    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validar que los nodos ingresados existan en el grafo
    if origen not in grafo or destino not in grafo:
        # Si alguno no existe, mostrar error y salir del modo interactivo
        print("\n Error: Uno o ambos nodos no existen en el grafo.")
        return

    # Informar al usuario que se iniciará la búsqueda
    print(f"\nBuscando camino en profundidad desde {origen} hasta {destino}...\n")

    # Ejecutar DFS con los nodos proporcionados
    camino = dfs(grafo, origen, destino)

    # Mostrar el resultado de la búsqueda (camino o no encontrado)
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No existe camino desde {origen} hasta {destino}.")


def main():
    # Grafo de ejemplo
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['D', 'E'],
        'C': ['A', 'F'],
        'D': ['C', 'F'],
        'E': ['B', 'F'],
        'F': ['A']
    }

    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (elegir origen y destino)\n")

    opcion = input("Ingrese el número de opción: ").strip()

    # Decidir qué modo ejecutar según la opción ingresada
    if opcion == '1':
        # Ejecutar modo demostrativo con valores por defecto
        demo_mode(grafo_ejemplo)
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario introduce los nodos
        interactive_mode(grafo_ejemplo)
    else:
        # Opción no válida: informar al usuario
        print("\n Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    main()
