"""
005-E1-busqueda_profundidad_iterativa.py
----------------------------------------
Este script implementa el algoritmo de Búsqueda en Profundidad Iterativa (Iterative Deepening Depth-First Search, IDDFS).
El algoritmo realiza múltiples iteraciones de búsqueda en profundidad limitada, incrementando el límite de profundidad,
hasta hallar el nodo destino o alcanzar un límite máximo definido.

Modos de ejecución:
1. MODO DEMO: usa un grafo de ejemplo, origen y destino predefinidos, y un límite máximo predeterminado.
2. MODO INTERACTIVO: el usuario ingresa el origen, destino y opcionalmente el límite máximo de profundidad.

Autor: Alejandro Aguirre Díaz
"""

def depth_limited_search(grafo, origen, destino, limite, camino=None, visitados=None):
    """
    Realiza búsqueda en profundidad con un límite de profundidad (DLS).
    :parametro grafo: dict, claves = nodos, valores = lista de adyacentes
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :parametro limite: entero ≥ 0, profundidad máxima permitida
    :parametro camino: lista con el camino recorrido hasta ahora
    :parametro visitados: conjunto de los nodos visitados en esta rama
    :return: lista del camino si se encuentra destino dentro del límite, o None si no se encuentra
    """
    # Inicializar estructuras auxiliares en la primera llamada
    if camino is None:
        camino = []
    if visitados is None:
        visitados = set()

    # Añadir el nodo actual al camino y marcarlo como visitado
    camino.append(origen)
    visitados.add(origen)

    # Caso base: si el nodo actual es el destino, retornar el camino construido
    if origen == destino:
        return camino

    # Caso base: si el límite de profundidad se agotó, no continuar
    if limite == 0:
        return None

    # Explorar recursivamente los vecinos no visitados
    for vecino in grafo.get(origen, []):
        if vecino not in visitados:
            # Llamada recursiva con límite reducido y copias de camino/visitados
            resultado = depth_limited_search(grafo, vecino, destino, limite-1,
                                             camino.copy(), visitados.copy())
            # Si se encuentra un camino válido, retornarlo inmediatamente
            if resultado:
                return resultado

    # Si ningún vecino lleva al destino dentro del límite, retornar None
    return None


def iterative_deepening_search(grafo, origen, destino, max_profundidad=10):
    """
    Realiza iterativamente DLS con límite creciente desde 0 hasta max_profundidad.
    :param grafo: dict
    :param origen: nodo inicial
    :param destino: nodo objetivo
    :param max_profundidad: entero ≥ 0, profundidad máxima a probar
    :return: lista del camino si lo encuentra antes de superar max_profundidad, o None
    """
    # Probar límites de profundidad desde 0 hasta el máximo especificado
    for limite in range(max_profundidad + 1):
        print(f" Probando con límite de profundidad = {limite}")
        # Ejecutar búsqueda limitada para el límite actual
        camino = depth_limited_search(grafo, origen, destino, limite)
        # Si se encuentra un camino, retornarlo inmediatamente
        if camino:
            return camino
    # Si ningún límite produce resultado, retornar None
    return None


def modo_demo(grafo):
    # Modo demostrativo con parámetros fijos
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    limite_max = 5
    # Informar parámetros de la búsqueda
    print(f"Buscando camino desde {origen} hasta {destino} con límite máximo = {limite_max}\n")
    # Ejecutar búsqueda iterativa
    resultado = iterative_deepening_search(grafo, origen, destino, limite_max)
    # Mostrar resultado según se encuentre camino o no
    if resultado:
        print(f" Camino encontrado: {resultado}")
    else:
        print(f"No se encontró camino con límite ≤ {limite_max}.")


def modo_interactivo(grafo):
    # Modo interactivo: el usuario ingresa parámetros
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles:", list(grafo.keys()))
    # Solicitar nodos de origen y destino
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()
    # Validar que los nodos existan en el grafo
    if origen not in grafo or destino not in grafo:
        print(" Uno o ambos nodos no están en el grafo.")
        return
    try:
        # Solicitar límite máximo de profundidad
        limite_max = int(input("Introduce el LÍMITE máximo de profundidad (entero ≥ 0): ").strip())
        if limite_max < 0:
            raise ValueError
    except ValueError:
        # Si el valor es inválido, usar valor por defecto
        print(" Límite inválido, se usará valor por defecto 5.")
        limite_max = 5

    # Informar parámetros y ejecutar búsqueda
    print(f"\nBuscando desde {origen} hasta {destino} con límite máximo = {limite_max}\n")
    resultado = iterative_deepening_search(grafo, origen, destino, limite_max)
    # Mostrar resultado según se encuentre camino o no
    if resultado:
        print(f" Camino encontrado: {resultado}")
    else:
        print(f"No se encontró camino con límite ≤ {limite_max}.")


def main():
    # Definir grafo de ejemplo con ciclos intencionados
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['D', 'E'],
        'C': ['F', 'A'],   # Ciclo intencionado
        'D': ['E', 'F'],
        'E': ['B', 'F'],
        'F': []
    }

    # Solicitar modo de ejecución al usuario
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese la opción (1 o 2): ").strip()
    # Ejecutar el modo correspondiente
    if opcion == '1':
        modo_demo(grafo_ejemplo)
    elif opcion == '2':
        modo_interactivo(grafo_ejemplo)
    else:
        # Manejo de opción inválida
        print(" Opción no válida.")

if __name__ == "__main__":
    main()
