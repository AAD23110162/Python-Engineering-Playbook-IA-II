"""
008-E1-heuristicas.py
--------------------------------
Este script define y demuestra el uso de una **función heurística** en el contexto de búsqueda informada en grafos.
- Incluye una función heurística de ejemplo (estimación “a priori” del coste restante hasta el objetivo).
- Integra dos modos de ejecución:
    1. MODO DEMO: utiliza un grafo pequeño con valores heurísticos predefinidos.
    2. MODO INTERACTIVO: permite al usuario elegir origen y destino, y presenta la estimación heurística para cada nodo.

Autor: Alejandro Aguirre Díaz
"""

def funcion_heuristica(nodo, objetivo, heuristica):
    """
    Obtiene el valor heurístico h(nodo) de estimación desde el nodo actual hasta el nodo objetivo.
    :parametro nodo: nodo actual
    :parametro objetivo: nodo objetivo
    :parametro heuristica: dict que mapea (nodo, objetivo) -> valor heurístico
    :return: valor heurístico (float o int), o None si no está definido
    """
    # Buscar en el diccionario heurística el valor para (nodo, objetivo)
    # Si no existe, retorna None
    return heuristica.get((nodo, objetivo), None)

def demo_mode(grafo, heuristica, objetivo):
    # Modo demostrativo: muestra los valores heurísticos para todos los nodos respecto al objetivo.
    print("\n--- MODO DEMO — Función heurística ---")
    # Mostrar el nodo objetivo elegido para esta demostración
    print(f"Objetivo elegido: {objetivo}")
    print("Valores heurísticos h(nodo → objetivo):")
    # Iterar sobre todos los nodos del grafo
    for nodo in grafo.keys():
        # Obtener el valor heurístico para cada nodo respecto al objetivo
        h_val = funcion_heuristica(nodo, objetivo, heuristica)
        # Mostrar el valor heurístico estimado
        print(f"  {nodo} → {objetivo} : {h_val}")
    # Nota informativa sobre el uso de heurísticas
    print("\nNota: Estos valores guían la búsqueda informada, pero no garantizan optimalidad si no son admisibles.")

def interactive_mode(grafo, heuristica):
    # Modo interactivo: el usuario ingresa el nodo objetivo y se muestran los valores heurísticos disponibles.
    print("\n--- MODO INTERACTIVO — Función heurística ---")
    print("Nodos disponibles:", list(grafo.keys()))
    # Solicitar al usuario el nodo de origen
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    # Solicitar al usuario el nodo de destino (objetivo)
    destino = input("Introduce el nodo de DESTINO (objetivo): ").strip().upper()

    # Validar que ambos nodos existan en el grafo
    if origen not in grafo or destino not in grafo:
        print("⚠️ Uno o ambos nodos no existen en el grafo.")
        return

    # Mostrar todas las estimaciones heurísticas para el objetivo elegido
    print(f"\nEstimaciones heurísticas para objetivo {destino}:")
    # Iterar sobre todos los nodos del grafo
    for nodo in grafo.keys():
        # Obtener el valor heurístico para cada nodo
        h_val = funcion_heuristica(nodo, destino, heuristica)
        # Mostrar el valor heurístico
        print(f"  {nodo} → {destino} : {h_val}")
    
    # Mostrar específicamente el valor heurístico del nodo de origen
    h_origen = funcion_heuristica(origen, destino, heuristica)
    print(f"\nValor heurístico para inicio {origen} → {destino}: {h_origen}")
    # Explicar la utilidad de estos valores
    print("Este valor guiaría una búsqueda informada como A* o búsqueda voraz.")

def main():
    # Definir grafo de ejemplo (nodos A-F, conexiones irrelevantes para esta demo)
    grafo_ejemplo = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['D', 'F'],
        'D': ['E'],
        'E': ['F'],
        'F': []
    }

    # Definir valores heurísticos de ejemplo: (nodo, objetivo) → estimación de coste restante
    # Estos valores representan la distancia estimada desde cada nodo hasta el objetivo F
    heuristica_ejemplo = {
        ('A','F'): 6,  # Estimación desde A hasta F
        ('B','F'): 4,  # Estimación desde B hasta F
        ('C','F'): 3,  # Estimación desde C hasta F
        ('D','F'): 2,  # Estimación desde D hasta F
        ('E','F'): 1,  # Estimación desde E hasta F
        ('F','F'): 0,  # El objetivo a sí mismo siempre es 0
        # También podríamos definir otros verdaderos objetivos
    }

    # Solicitar modo de ejecución al usuario
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()

    # Ejecutar el modo correspondiente según la opción elegida
    if opcion == '1':
        # Ejecutar modo demostrativo con objetivo F predefinido
        demo_mode(grafo_ejemplo, heuristica_ejemplo, objetivo='F')
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario elige origen y destino
        interactive_mode(grafo_ejemplo, heuristica_ejemplo)
    else:
        # Manejo de opción inválida
        print(" Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()
