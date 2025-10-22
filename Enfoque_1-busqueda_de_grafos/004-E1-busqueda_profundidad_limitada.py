"""
004-E1-busqueda_profundidad_limitada.py
--------------------------------
Este script implementa el algoritmo de Búsqueda en Profundidad Limitada (Depth-Limited Search, DLS),
una variante del algoritmo DFS que impone un límite máximo de profundidad en la exploración del grafo.

El programa puede ejecutarse en dos modos:
1. MODO DEMO: ejemplo automático con un límite predefinido.
2. MODO INTERACTIVO: el usuario elige origen, destino y el límite de profundidad.

Autor: Alejandro Aguirre Díaz
"""

def dls(grafo, origen, destino, limite, profundidad=0, camino=None):
    """
    Realiza búsqueda en profundidad con límite de profundidad (DLS).
    :parametro grafo: dict -> llaves = nodos, valores = lista de adyacentes
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :parametro limite: profundidad máxima permitida
    :parametro profundidad: nivel actual (interno)
    :parametro camino: lista con el camino recorrido hasta ahora
    :return: lista con el camino si se encuentra dentro del límite, None si no se encuentra
    """
    # Inicializar el camino si es la primera llamada
    if camino is None:
        camino = []

    # Agregar el nodo actual al camino en construcción
    camino.append(origen)

    # Si encontramos el destino
    if origen == destino:
        return camino

    # Si alcanzamos el límite sin encontrar el destino
    if profundidad >= limite:
        return None

    # Continuamos expandiendo vecinos
    for vecino in grafo.get(origen, []):
        # Evitar ciclos: no volver a nodos ya presentes en el camino actual
        if vecino not in camino:
            # Llamada recursiva incrementando profundidad y usando copia del camino
            resultado = dls(grafo, vecino, destino, limite, profundidad + 1, camino.copy())
            # Si se encontró un resultado en esta rama, retornarlo inmediatamente
            if resultado:
                return resultado

    return None


def demo_mode(grafo):
    # Ejecuta el modo demostrativo con un límite predefinido.
    print("\n--- MODO DEMO ---")
    origen = 'A'
    destino = 'F'
    limite = 3
    # Mostrar los parámetros de la búsqueda a realizar
    print(f"Buscando camino desde {origen} hasta {destino} con límite de profundidad = {limite}\n")

    # Ejecutar DLS con valores por defecto del modo demo
    camino = dls(grafo, origen, destino, limite)
    # Informar del resultado según se encuentre camino o no
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No se encontró camino dentro del límite de profundidad ({limite}).")


def interactive_mode(grafo):
    # Ejecuta el modo interactivo donde el usuario elige origen, destino y límite.
    print("\n--- MODO INTERACTIVO ---")
    print("Nodos disponibles:", list(grafo.keys()))
    print("Ejemplo: A, B, C, D, E, F\n")

    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validación temprana de que los nodos pertenecen al grafo
    if origen not in grafo or destino not in grafo:
        print("\n Error: Uno o ambos nodos no existen en el grafo.")
        return

    try:
        # Solicitar y convertir el límite de profundidad a entero
        limite = int(input("Introduce el LÍMITE de profundidad (entero positivo): ").strip())
        if limite < 0:
            raise ValueError
    except ValueError:
        print("\n Error: El límite debe ser un número entero positivo.")
        return

    # Informar el inicio de la búsqueda con los parámetros elegidos
    print(f"\nBuscando camino desde {origen} hasta {destino} con límite = {limite}\n")
    # Ejecutar la búsqueda DLS y obtener el posible camino
    camino = dls(grafo, origen, destino, limite)
    # Mostrar el resultado de la ejecución
    if camino:
        print(f"Camino encontrado: {camino}")
    else:
        print(f"No se encontró camino dentro del límite de profundidad ({limite}).")


def main():
    # Grafo de ejemplo
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['C', 'D'],
        'C': ['E', 'F'],
        'D': ['A', 'F'],
        'E': ['B', 'F'],
        'F': ['C']
    }

    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (ejemplo automático)")
    print("2. Modo INTERACTIVO (elegir origen, destino y límite)\n")

    opcion = input("Ingrese el número de opción: ").strip()

    # Ejecutar el modo seleccionado por el usuario
    if opcion == '1':
        # Modo demostrativo con parámetros predefinidos
        demo_mode(grafo_ejemplo)
    elif opcion == '2':
        # Modo interactivo: el usuario ingresa origen, destino y límite
        interactive_mode(grafo_ejemplo)
    else:
        # Manejo de opción inválida
        print("\n Opción no válida. Intente nuevamente.")


if __name__ == "__main__":
    main()
