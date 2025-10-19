"""
009-E1-busqueda_voraz_primero_mejor.py
----------------------------------------
Este script implementa el algoritmo de Búsqueda Voraz Primero el Mejor (Greedy Best-First Search):
- Utiliza únicamente una función heurística h(n) que estima el coste desde el nodo actual hasta el objetivo.
- Expande el nodo con el valor más bajo de h(n) en cada iteración.
- Incluye modo DEMO (origen y destino fijos) y modo INTERACTIVO (usuario elige origen y destino).
- Nota: la solución encontrada puede **no ser la óptima**, pues no considera el coste real g(n).

Autor: Alejandro Aguirre Díaz
"""

import heapq

def busqueda_voraz(grafo, heuristica, origen, destino):
    """
    Realiza la búsqueda voraz en el grafo desde 'origen' hasta 'destino'.
    :parametro grafo: dict, claves = nodos, valores = lista de adyacentes
    :parametro heuristica: dict, claves = (nodo, destino) -> valor heurístico h(n)
    :parametro origen: nodo de inicio
    :parametro destino: nodo objetivo
    :return: lista con el camino encontrado, o None si no hay camino
    """
    print("=" * 70)
    print("INICIO DE BÚSQUEDA VORAZ PRIMERO EL MEJOR")
    print("=" * 70)
    print(f"Origen: {origen} | Destino: {destino}\n")
    
    # Inicializar lista de nodos abiertos (frontera) como cola de prioridad
    abiertos = []
    # cada entrada en abiertos: (h(nodo), nodo, camino_hasta_nodo)
    h_inicial = heuristica.get((origen, destino), float('inf'))
    heapq.heappush(abiertos, (h_inicial, origen, [origen]))
    print(f"[OK] Inicializando con nodo origen: {origen}")
    print(f"  h({origen}) = {h_inicial}\n")
    
    # Conjunto de nodos visitados para evitar ciclos
    visitados = set()
    iteracion = 0

    # Bucle principal: explorar mientras haya nodos en la frontera
    while abiertos:
        iteracion += 1
        print(f"--- ITERACIÓN {iteracion} ---")
        
        # Extraer el nodo con menor valor heurístico (más prometedor)
        h_val, nodo_actual, camino = heapq.heappop(abiertos)
        print(f"→ Expandiendo nodo: {nodo_actual} (h = {h_val})")
        print(f"  Camino hasta aquí: {' → '.join(camino)}")

        # Verificar si alcanzamos el destino
        if nodo_actual == destino:
            print(f"\n{'=' * 70}")
            print(f"META ALCANZADA! Nodo {destino} encontrado")
            print(f"{'=' * 70}")
            print(f"Camino final: {' → '.join(camino)}")
            print(f"Longitud del camino: {len(camino)} nodos")
            print(f"Iteraciones totales: {iteracion}\n")
            return camino

        # Si el nodo ya fue visitado, saltarlo
        if nodo_actual in visitados:
            print(f"  [AVISO] Nodo {nodo_actual} ya visitado, se omite\n")
            continue
        
        # Marcar el nodo como visitado
        visitados.add(nodo_actual)
        print(f"  [OK] Marcando {nodo_actual} como visitado")

        # Explorar los vecinos del nodo actual
        vecinos = grafo.get(nodo_actual, [])
        print(f"  Vecinos de {nodo_actual}: {vecinos}")
        
        vecinos_agregados = []
        for vecino in vecinos:
            # Solo considerar vecinos no visitados
            if vecino not in visitados:
                # Obtener el valor heurístico del vecino
                h_vecino = heuristica.get((vecino, destino), float('inf'))
                # Construir el nuevo camino incluyendo el vecino
                nuevo_camino = camino + [vecino]
                # Agregar el vecino a la frontera con su valor heurístico
                heapq.heappush(abiertos, (h_vecino, vecino, nuevo_camino))
                vecinos_agregados.append(f"{vecino}(h={h_vecino})")
        
        if vecinos_agregados:
            print(f"  [+] Agregando a frontera: {', '.join(vecinos_agregados)}")
        else:
            print(f"  [AVISO] No hay vecinos no visitados para agregar")
        
        # Mostrar estado actual de la frontera
        if abiertos:
            frontera_info = [f"{n}(h={h})" for h, n, _ in sorted(abiertos)[:5]]
            print(f"  [Frontera] actual (primeros 5): {', '.join(frontera_info)}")
        print(f"  Visitados: {sorted(visitados)}\n")

    # Si se agota la frontera sin encontrar el destino
    print(f"{'=' * 70}")
    print(f"NO SE ENCONTRÓ CAMINO")
    print(f"{'=' * 70}")
    print(f"Se exploraron {iteracion} iteraciones sin alcanzar el destino {destino}\n")
    return None

def modo_demo(grafo, heuristica):
    # Modo demostrativo con origen y destino predefinidos.
    print("\n" + "=" * 70)
    print("MODO DEMO - BÚSQUEDA VORAZ PRIMERO EL MEJOR")
    print("=" * 70)
    origen = 'A'
    destino = 'F'
    print(f"\n[Configuración predefinida]")
    print(f"   - Origen: {origen}")
    print(f"   - Destino: {destino}")
    print(f"   - Algoritmo: Búsqueda Voraz (Greedy Best-First)")
    print(f"\nPresiona Enter para iniciar la búsqueda...")
    input()
    
    # Ejecutar búsqueda voraz con trazas detalladas
    camino = busqueda_voraz(grafo, heuristica, origen, destino)
    
    # Mostrar resumen final
    if camino:
        print(f"[EXITO] RESULTADO: Camino encontrado exitosamente")
        print(f"   Camino: {' → '.join(camino)}")
    else:
        print(f"[ERROR] RESULTADO: No existe camino desde {origen} hasta {destino}.")

def modo_interactivo(grafo, heuristica):
    # Modo interactivo donde el usuario ingresa origen y destino.
    print("\n" + "=" * 70)
    print("MODO INTERACTIVO - BÚSQUEDA VORAZ PRIMERO EL MEJOR")
    print("=" * 70)
    print("\n[Nodos disponibles] en el grafo:", list(grafo.keys()))
    
    # Mostrar valores heurísticos disponibles
    print("\n[Valores heurísticos disponibles]")
    for (nodo, destino_h), valor in sorted(heuristica.items()):
        print(f"   h({nodo} → {destino_h}) = {valor}")
    
    print("\n" + "-" * 70)
    # Solicitar nodos de origen y destino al usuario
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()

    # Validar que los nodos existan en el grafo
    if origen not in grafo or destino not in grafo:
        print("\n[ERROR] Error: Uno o ambos nodos no existen en el grafo.")
        return

    print(f"\n[Configuración elegida]")
    print(f"   - Origen: {origen}")
    print(f"   - Destino: {destino}")
    print(f"   - Algoritmo: Búsqueda Voraz (Greedy Best-First)")
    print(f"\nPresiona Enter para iniciar la búsqueda...")
    input()
    
    # Ejecutar búsqueda voraz con trazas detalladas
    camino = busqueda_voraz(grafo, heuristica, origen, destino)
    
    # Mostrar resumen final
    if camino:
        print(f"[EXITO] RESULTADO: Camino encontrado exitosamente")
        print(f"   Camino: {' → '.join(camino)}")
    else:
        print(f"[ERROR] RESULTADO: No existe camino desde {origen} hasta {destino}.")

def main():
    # Definir grafo de ejemplo (nodos A-F con sus conexiones)
    grafo_ejemplo = {
        'A': ['B', 'C', 'E'],
        'B': ['D', 'E', 'F'],
        'C': ['F'],
        'D': ['F'],
        'E': ['B', 'F'],
        'F': []
    }

    # Definir valores heurísticos de ejemplo: (nodo, destino) -> estimación h(n)
    # Estos valores representan la distancia estimada desde cada nodo hasta el objetivo F
    heuristica_ejemplo = {
        ('A','F'): 5,  # Estimación desde A hasta F
        ('B','F'): 3,  # Estimación desde B hasta F
        ('C','F'): 2,  # Estimación desde C hasta F
        ('D','F'): 1,  # Estimación desde D hasta F
        ('E','F'): 2,  # Estimación desde E hasta F
        ('F','F'): 0   # El objetivo a sí mismo siempre es 0
    }

    # Mostrar menú principal
    print("\n" + "=" * 70)
    print("BÚSQUEDA VORAZ PRIMERO EL MEJOR (Greedy Best-First Search)")
    print("=" * 70)
    print("\n[Información del algoritmo]")
    print("   - Expande siempre el nodo con menor h(n) (más prometedor)")
    print("   - No garantiza solución óptima")
    print("   - Solo usa heurística, no considera coste acumulado\n")
    
    print("Seleccione el modo de ejecución:")
    print("1. Modo DEMO (origen y destino predefinidos)")
    print("2. Modo INTERACTIVO (elegir origen y destino)\n")
    opcion = input("Ingrese el número de opción: ").strip()

    # Ejecutar el modo correspondiente según la opción elegida
    if opcion == '1':
        # Ejecutar modo demostrativo con valores predefinidos
        modo_demo(grafo_ejemplo, heuristica_ejemplo)
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario elige los nodos
        modo_interactivo(grafo_ejemplo, heuristica_ejemplo)
    else:
        # Manejo de opción inválida
        print("\n[ERROR] Opción no válida. Intente nuevamente.")

if __name__ == "__main__":
    main()
