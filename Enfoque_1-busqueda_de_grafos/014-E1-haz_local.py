"""
014-E1-haz_local.py
--------------------------------
Este script implementa el algoritmo de Búsqueda por Haz Local:
- Mantiene una “haz” de k estados candidatas en cada iteración.
- A partir de todos los sucesores de los estados de la haz, se seleccionan los k mejores para la próxima iteración.
- Termina cuando se alcanza un número máximo de iteraciones o cuando se encuentra un estado objetivo (opcional).
- Incluye dos modos de ejecución:
    1. MODO DEMO: parámetros fijos, muestra el progreso por terminal.
    2. MODO INTERACTIVO: usuario define el estado inicial (o haz inicial), tamaño de la haz (k), número máximo de iteraciones, etc.

Autor: Alejandro Aguirre Díaz
"""

import random

def funcion_objetivo(estado):
    """
    Función que queremos maximizar (o minimizar) en el espacio de estados.
    :parametro estado: lista o valor que representa un estado
    :return: valor numérico que indica la “bondad” del estado
    """
    # Ejemplo sencillo: maximizar f(x) = -(x-5)^2 + 10 (máximo en x = 5)
    # Nota: cuanto más cerca esté x de 5, mayor es el valor; p.ej. x=5 → 10, x=4 → 9, x=0 → -15
    return - (estado - 5) ** 2 + 10

def generar_vecinos(estado, paso=1.0, num_vecinos=5):
    """
    Genera vecinos del estado dado mediante pequeñas variaciones.
    :parametro estado: valor actual (float)
    :parametro paso: magnitud del cambio
    :parametro num_vecinos: cuántos vecinos generar
    :return: lista de estados vecino
    """
    # Crear 'num_vecinos' variaciones del estado en el rango [-paso, paso]
    return [estado + random.uniform(-paso, paso) for _ in range(num_vecinos)]

def busqueda_haz_local(haz_inicial, k=3, paso=1.0, num_vecinos=5, max_iteraciones=20):
    """
    Ejecuta la búsqueda por haz local.
    :parametro haz_inicial: lista de estados (floats) que constituyen la selección inicial
    :parametro k: tamaño de la haz (número de estados que se conservan cada iteración)
    :parametro paso: magnitud de variación al generar vecinos
    :parametro num_vecinos: número de vecinos generados por cada estado de la haz
    :parametro max_iteraciones: número máximo de iteraciones permitidas
    :return: mejor_estado encontrado y valor objetivo
    """
    # Copiar la haz inicial para no modificar la lista original
    haz = haz_inicial.copy()
    # Calcular la valoración (función objetivo) de cada estado inicial
    valores = [(estado, funcion_objetivo(estado)) for estado in haz]
    # Guardar el mejor estado y su valor hasta ahora
    mejor_estado, mejor_valor = max(valores, key=lambda x: x[1])

    # Cabecera de trazas para observar el progreso por iteración
    print("Iter | Estados de la haz (estado:valor) | Mejor hasta ahora")
    for iteracion in range(1, max_iteraciones + 1):
        # Generar todos los vecinos de todos los estados de la haz
        candidatos = []
        for estado in haz:
            # Para cada estado, generamos 'num_vecinos' vecinos con variaciones aleatorias
            vecinos = generar_vecinos(estado, paso, num_vecinos)
            for v in vecinos:
                # Evaluar el valor objetivo de cada vecino y guardarlo como candidato
                candidatos.append((v, funcion_objetivo(v)))
        # Seleccionar los k mejores candidatos (mayor valor objetivo)
        candidatos.sort(key=lambda x: x[1], reverse=True)
        haz = [estado for estado, valor in candidatos[:k]]
        mejor_iter_estado, mejor_iter_valor = candidatos[0]

        # Actualizar el mejor global si se supera el mejor valor conocido
        if mejor_iter_valor > mejor_valor:
            mejor_estado, mejor_valor = mejor_iter_estado, mejor_iter_valor

        # Mostrar estado de la iteración: los k estados actuales y el mejor global
        estados_str = ", ".join(f"{e:.2f}:{funcion_objetivo(e):.2f}" for e in haz)
        print(f"{iteracion:4d} | {estados_str} | {mejor_estado:.2f}:{mejor_valor:.2f}")

    print(f"\nResultado final: mejor estado = {mejor_estado:.2f} | valor = {mejor_valor:.2f}")
    return mejor_estado, mejor_valor

def modo_demo():
    """Modo demostrativo con parámetros fijos."""
    # Anunciar modo demo y fijar parámetros reproducibles
    print("\n--- MODO DEMO ---")
    # Conjunto de estados iniciales variados para explorar el espacio
    haz_inicial = [0.0, 10.0, -5.0]
    # Tamaño de la haz: cuántos mejores estados conservamos en cada iteración
    k = 3
    # Amplitud de las variaciones al generar vecinos
    paso = 1.0
    # Vecinos a generar por cada estado de la haz
    num_vecinos = 5
    # Límite de iteraciones para detener la búsqueda
    max_iter = 10
    print(f"Haz inicial = {haz_inicial}, tamaño haz = {k}, paso = {paso}, num_vecinos = {num_vecinos}, max_iter = {max_iter}\n")
    # Ejecutar el algoritmo con los parámetros definidos
    busqueda_haz_local(haz_inicial, k, paso, num_vecinos, max_iter)

def modo_interactivo():
    """Modo interactivo donde el usuario define parámetros."""
    # Anunciar modo interactivo y solicitar parámetros al usuario
    print("\n--- MODO INTERACTIVO ---")
    try:
        # Leer estados iniciales separados por comas y convertirlos a float
        entrada = input("Introduce los estados iniciales de la haz separados por comas (ej: 0,10,-5): ").strip()
        haz_inicial = [float(x) for x in entrada.split(",")]
    except Exception:
        print("Entrada inválida. Se usará [0.0,10.0,-5.0]")
        haz_inicial = [0.0, 10.0, -5.0]
    try:
        # Leer tamaño de la haz (k) y validar que sea al menos 1
        k = int(input("Introduce el tamaño de la haz (k): ").strip())
        if k < 1:
            raise ValueError
    except ValueError:
        print("Entrada inválida. Se usará 3")
        k = 3
    try:
        # Leer tamaño del paso de variación
        paso = float(input("Introduce el tamaño del paso para generar vecinos: ").strip())
    except ValueError:
        print("Entrada inválida. Se usará 1.0")
        paso = 1.0
    try:
        # Leer número de vecinos a generar por cada estado y validar que sea >= 1
        num_vecinos = int(input("Introduce cuántos vecinos generar por estado: ").strip())
        if num_vecinos < 1:
            raise ValueError
    except ValueError:
        print("Entrada inválida. Se usará 5")
        num_vecinos = 5
    try:
        # Leer el máximo de iteraciones y validar que sea >= 1
        max_iter = int(input("Introduce el número máximo de iteraciones: ").strip())
        if max_iter < 1:
            raise ValueError
    except ValueError:
        print("Entrada inválida. Se usará 20")
        max_iter = 20

    print(f"\nHaz inicial = {haz_inicial}, tamaño haz = {k}, paso = {paso}, num_vecinos = {num_vecinos}, max_iter = {max_iter}\n")
    # Ejecutar el algoritmo con los parámetros proporcionados por el usuario
    busqueda_haz_local(haz_inicial, k, paso, num_vecinos, max_iter)

def main():
    # Menú simple para seleccionar modo de ejecución
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()

    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        # Manejo de opción inválida
        print("Opción no válida.")

if __name__ == "__main__":
    main()
