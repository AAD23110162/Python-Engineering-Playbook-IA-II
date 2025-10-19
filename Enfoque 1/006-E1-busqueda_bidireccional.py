"""
006-E1-busqueda_bidireccional.py
--------------------------------
Este script implementa el algoritmo de Búsqueda Bidireccional (Bidirectional Search):
- Realiza dos búsquedas simultáneas: una desde el nodo de origen hacia el destino y otra desde el destino hacia el origen.
- Cuando ambas búsquedas se encuentran en un mismo nodo se combina el camino y se devuelve la solución.
- El grafo está representado como lista de adyacencia para la búsqueda hacia adelante, y se asume que también se puede invertir para la búsqueda hacia atrás.

Modos de ejecución:
1. MODO DEMO: ejemplo automático con origen y destino predefinidos.
2. MODO INTERACTIVO: el usuario ingresa origen y destino, se muestran las opciones disponibles.

Autor: Alejandro Aguirre Díaz
"""

from collections import deque

def invert_graph(grafo):
    """
    Crea un grafo inverso (aristas invertidas) para permitir la búsqueda backward.
    :parametro grafo: dict, llaves = nodo, valores = lista de nodos adyacentes
    :return: dict, grafo invertido
    """
    # Inicializar grafo invertido con todos los nodos
    invertido = {nodo: [] for nodo in grafo}
    # Invertir cada arista del grafo original
    for nodo, vecinos in grafo.items():
        for vecino in vecinos:
            # Asegurar que el vecino existe en el grafo invertido
            if vecino not in invertido:
                invertido[vecino] = []
            # Agregar arista invertida
            invertido[vecino].append(nodo)
    return invertido

def bidirectional_search(grafo, origen, destino):
    """
    Realiza búsqueda bidireccional en el grafo desde 'origen' hasta 'destino'.
    :parametro grafo: dict, llaves = nodos, valores = lista de vecinos (aristas)
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :return: lista representando el camino encontrado, o None si no hay camino
    """
    # Caso trivial: origen y destino son el mismo nodo
    if origen == destino:
        return [origen]

    # Crear grafo invertido para la búsqueda desde destino hacia origen
    grafo_invertido = invert_graph(grafo)

    # Inicializar fronteras (colas) para cada dirección de búsqueda
    frontera_adelante = deque([origen])
    frontera_atras = deque([destino])

    # Inicializar conjuntos de visitados con caminos parciales para cada dirección
    visitados_adelante = {origen: [origen]}
    visitados_atras = {destino: [destino]}

    # Continuar mientras ambas fronteras tengan nodos por explorar
    while frontera_adelante and frontera_atras:
        # Expandir un nivel en la búsqueda hacia adelante (desde origen)
        if frontera_adelante:
            nodo_actual = frontera_adelante.popleft()
            # Explorar vecinos del nodo actual
            for vecino in grafo.get(nodo_actual, []):
                if vecino not in visitados_adelante:
                    # Registrar camino hasta este vecino
                    visitados_adelante[vecino] = visitados_adelante[nodo_actual] + [vecino]
                    frontera_adelante.append(vecino)
                    # Comprobar intersección: si el vecino ya fue visitado desde destino
                    if vecino in visitados_atras:
                        # Se encontró punto de encuentro, combinar caminos
                        camino_adelante = visitados_adelante[vecino]
                        camino_atras = visitados_atras[vecino]
                        # Invertir camino desde destino (excepto nodo de encuentro) y concatenar
                        return camino_adelante + visitados_atras[vecino][::-1][1:]
        
        # Expandir un nivel en la búsqueda hacia atrás (desde destino)
        if frontera_atras:
            nodo_actual_atras = frontera_atras.popleft()
            # Explorar vecinos en el grafo invertido
            for vecino_atras in grafo_invertido.get(nodo_actual_atras, []):
                if vecino_atras not in visitados_atras:
                    # Registrar camino hasta este vecino
                    visitados_atras[vecino_atras] = visitados_atras[nodo_actual_atras] + [vecino_atras]
                    frontera_atras.append(vecino_atras)
                    # Comprobar intersección: si el vecino ya fue visitado desde origen
                    if vecino_atras in visitados_adelante:
                        # Se encontró punto de encuentro, combinar caminos
                        camino_adelante = visitados_adelante[vecino_atras]
                        camino_atras = visitados_atras[vecino_atras]
                        # Invertir camino desde destino (excepto nodo de encuentro) y concatenar
                        return camino_adelante + visitados_atras[vecino_atras][::-1][1:]

    # Si se agotan ambas fronteras sin encontrar intersección, no hay camino
    return None

def modo_demo(grafo):
    """Modo demostrativo con valores predefinidos."""
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    # Informar parámetros de la búsqueda bidireccional
    print(f"Buscando camino bidireccional desde {origen} hasta {destino}...\n")
    # Ejecutar búsqueda bidireccional
    camino = bidirectional_search(grafo, origen, destino)
    # Mostrar resultado según se encuentre camino o no
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No existe camino entre {origen} y {destino} mediante búsqueda bidireccional.")

def modo_interactivo(grafo):
    """Modo interactivo donde el usuario elige origen y destino."""
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles:", list(grafo.keys()))
    # Solicitar nodos de origen y destino
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validar que los nodos existan en el grafo
    if origen not in grafo or destino not in grafo:
        print("\n Uno o ambos nodos no existen en el grafo.")
        return

    # Informar parámetros y ejecutar búsqueda bidireccional
    print(f"\nBuscando camino bidireccional desde {origen} hasta {destino}...\n")
    camino = bidirectional_search(grafo, origen, destino)
    # Mostrar resultado según se encuentre camino o no
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No existe camino entre {origen} y {destino} mediante búsqueda bidireccional.")

def main():
    # Definir grafo de ejemplo (nodos A–F, con múltiples conexiones, dirigido)
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['D', 'E', 'F'],
        'C': ['A', 'F'],
        'D': ['E'],
        'E': ['B', 'F'],
        'F': ['C']
    }

    # Solicitar modo de ejecución al usuario
    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (elegir origen y destino)\n")

    opcion = input("Ingrese el número de opción: ").strip()
    # Ejecutar el modo correspondiente
    if opcion == '1':
        modo_demo(grafo_ejemplo)
    elif opcion == '2':
        modo_interactivo(grafo_ejemplo)
    else:
        # Manejo de opción inválida
        print("\n Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()
