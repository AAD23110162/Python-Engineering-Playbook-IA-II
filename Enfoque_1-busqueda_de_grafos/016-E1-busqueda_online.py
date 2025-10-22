"""
016-E1-busqueda_online.py
--------------------------------
Este script implementa una versión simplificada del algoritmo de Búsqueda Online:
- El agente explora un grafo desconocido inicialmente y va descubriendo los nodos adyacentes conforme avanza.
- En cada paso, el agente decide "hacia dónde moverse" basándose en la información que ha adquirido hasta ahora.
- Incluye modo DEMO (grafo oculto parcialmente, el agente explora) y modo INTERACTIVO (usuario puede intervenir).
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

import random
from collections import deque

def explorar_vecinos(grafo, nodo_actual, descubierto):
    """
    Retorna los vecinos del nodo_actual que aún no han sido completamente explorados.
    :parametro grafo: dict, claves = nodos, valores = lista de vecinos
    :parametro nodo_actual: nodo donde está el agente
    :parametro descubierto: set de nodos ya descubiertos
    :return: lista de vecinos nuevos para explorar
    """
    # Obtener todos los vecinos del nodo actual desde el grafo
    # Si el nodo no existe en el grafo, retornar lista vacía
    vecinos = grafo.get(nodo_actual, [])
    
    # Filtrar solo los vecinos que NO han sido descubiertos aún
    # Esto permite explorar nodos nuevos y evitar ciclos
    nuevos = [v for v in vecinos if v not in descubierto]
    
    # Retornar la lista de vecinos nuevos por explorar
    return nuevos

def busqueda_online(grafo, origen, destino, max_pasos=100):
    """
    Simula la búsqueda online: el agente parte de origen, explora vecinos poco a poco y decide un camino hasta destino.
    :parametro grafo: dict, claves = nodos, valores = lista de vecinos
    :parametro origen: nodo inicial
    :parametro destino: nodo objetivo
    :parametro max_pasos: máximo de movimientos permitidos
    :return: lista con el camino realizado o None si no se alcanza el destino
    """
    # Inicializar conjunto de nodos descubiertos con el origen
    # Este conjunto registra todos los nodos que el agente ha visitado o conoce
    descubrimiento = {origen}
    
    # Inicializar el camino con el nodo origen
    # Esta lista mantiene la secuencia de nodos visitados
    camino = [origen]
    
    # El nodo actual es el origen al comenzar
    # Variable que indica dónde está posicionado el agente
    nodo_actual = origen

    # Iterar hasta el número máximo de pasos permitidos
    # Cada iteración representa un movimiento o decisión del agente
    for paso in range(1, max_pasos + 1):
        # Verificar si hemos alcanzado el destino
        # Si el nodo actual es el objetivo, la búsqueda termina con éxito
        if nodo_actual == destino:
            print(f"[EXITO] Destino {destino} alcanzado en el paso {paso - 1}!")
            return camino

        # Explorar vecinos del nodo actual que aún no han sido descubiertos
        # Llamar a la función que filtra vecinos nuevos
        vecinos = explorar_vecinos(grafo, nodo_actual, descubrimiento)
        
        # Si no quedan vecinos nuevos por explorar, realizar retroceso (backtracking)
        if not vecinos:
            # Verificar si podemos retroceder (hay más de un nodo en el camino)
            # Si solo queda el origen, no hay dónde retroceder
            if len(camino) > 1:
                # Remover el nodo actual del camino (retroceso)
                # Esto simula que el agente regresa al nodo anterior
                camino.pop()
                
                # Actualizar el nodo actual al nodo anterior en el camino
                # El agente ahora está en el último nodo del camino actualizado
                nodo_actual = camino[-1]
                
                # Imprimir el retroceso realizado para seguimiento visual
                print(f"Paso {paso}: [RETROCESO] regresando a {nodo_actual}. Camino actual: {camino}")
                
                # Continuar con la siguiente iteración sin incrementar el camino
                continue
            else:
                # No hay más nodos a los que retroceder, búsqueda fallida
                # El agente está atrapado y no puede encontrar el destino
                print(f"Paso {paso}: [ERROR] No hay más nodos para explorar ni retroceder.")
                break

        # Decidir hacia dónde moverse: selección aleatoria del vecino nuevo
        # En búsqueda online real, esta decisión podría basarse en heurísticas
        siguiente = random.choice(vecinos)
        
        # Marcar el nuevo nodo como descubierto
        # Agregar al conjunto de nodos conocidos
        descubrimiento.add(siguiente)
        
        # Agregar el nuevo nodo al camino
        # Extender la ruta que el agente ha seguido
        camino.append(siguiente)
        
        # Actualizar el nodo actual al nuevo nodo
        # El agente ahora está posicionado en este nodo
        nodo_actual = siguiente
        
        # Imprimir el movimiento realizado para seguimiento visual
        # Mostrar avance y estado actual del camino
        print(f"Paso {paso}: [AVANCE] moviéndose a {nodo_actual}. Camino actual: {camino}")

    # Si se alcanzó el máximo de pasos sin llegar al destino
    # La búsqueda falló por límite de iteraciones
    print(f"\n[AVISO] Se alcanzó el máximo de pasos ({max_pasos}) sin hallar el destino.")
    return None

def modo_demo(grafo):
    """Modo demostrativo con parámetros fijos."""
    # Encabezado del modo demo
    print("\n--- MODO DEMO ---")
    
    # Definir origen y destino predefinidos para la demostración
    origen = 'A'
    destino = 'F'
    
    # Informar al usuario sobre los parámetros de la exploración
    print(f"Exploración online comenzando en {origen}, objetivo {destino}.\n")
    
    # Ejecutar la búsqueda online con máximo de 20 pasos
    camino = busqueda_online(grafo, origen, destino, max_pasos=20)
    
    # Verificar si se encontró un camino al destino
    if camino:
        # Si se encontró, mostrar el camino completo
        print(f"Camino alcanzado: {camino}")
    else:
        # Si no se encontró, informar al usuario
        print(f"No se alcanzó el destino {destino} mediante exploración online.")

def modo_interactivo(grafo):
    """Modo interactivo donde usuario define origen, destino y pasos máximos."""
    # Encabezado del modo interactivo
    print("\n--- MODO INTERACTIVO ---")
    
    # Mostrar los nodos disponibles en el grafo
    print("Nodos disponibles:", list(grafo.keys()))
    
    # Solicitar al usuario que ingrese el nodo de origen
    origen = input("Introduce el nodo de ORIGEN: ").strip().upper()
    
    # Solicitar al usuario que ingrese el nodo de destino
    destino = input("Introduce el nodo de DESTINO: ").strip().upper()
    
    # Solicitar el número máximo de pasos y validar la entrada
    try:
        # Intentar convertir la entrada a entero
        max_pasos = int(input("Introduce número máximo de pasos permitidos: ").strip())
        
        # Verificar que sea un número positivo
        if max_pasos < 1:
            raise ValueError
    except ValueError:
        # Si la entrada es inválida, usar valor predeterminado
        print("Entrada inválida. Se usará 50 pasos.")
        max_pasos = 50

    # Validar que ambos nodos existan en el grafo
    if origen not in grafo or destino not in grafo:
        # Si alguno no existe, mostrar error y terminar
        print("Error: Uno o ambos nodos no existen en el grafo.")
        return

    # Informar al usuario sobre los parámetros configurados
    print(f"\nExploración online iniciada en {origen}, objetivo {destino}, max pasos = {max_pasos}.\n")
    
    # Ejecutar la búsqueda online con los parámetros del usuario
    camino = busqueda_online(grafo, origen, destino, max_pasos)
    
    # Verificar si se encontró un camino al destino
    if camino:
        # Si se encontró, mostrar el camino completo
        print(f"Camino alcanzado: {camino}")
    else:
        # Si no se encontró, informar al usuario
        print(f"No se alcanzó el destino {destino} mediante exploración online.")

def main():
    """Función principal que ejecuta el programa."""
    # Definir grafo de ejemplo con nodos A-F y sus conexiones
    # Cada nodo mapea a una lista de nodos adyacentes (vecinos)
    grafo_ejemplo = {
        'A': ['B', 'C'],      # A conecta con B y C
        'B': ['D', 'E'],      # B conecta con D y E
        'C': ['E', 'F'],      # C conecta con E y F
        'D': ['C'],           # D conecta solo con C
        'E': ['F'],           # E conecta solo con F
        'F': []               # F no tiene vecinos (nodo terminal)
    }

    # Mostrar menú de opciones al usuario
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    
    # Solicitar al usuario que elija una opción
    opcion = input("Ingrese el número de opción: ").strip()
    
    # Ejecutar el modo correspondiente según la opción elegida
    if opcion == '1':
        # Ejecutar modo demo con parámetros predefinidos
        modo_demo(grafo_ejemplo)
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario configura parámetros
        modo_interactivo(grafo_ejemplo)
    else:
        # Mostrar mensaje de error si la opción no es válida
        print("Opción no válida.")

# Punto de entrada del programa
if __name__ == "__main__":
    # Ejecutar la función principal
    main()
