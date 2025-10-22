"""
015-E1-algoritmos_geneticos.py
--------------------------------
Este script implementa una versión simplificada de Algoritmos Genéticos (AG):
- Representa una población de soluciones, las evalúa con una función de aptitud, selecciona progenitores, aplica cruce y mutación para generar nuevas generaciones.
- Incluye visualización por terminal de la población, aptitudes y evolución.
- Modos de ejecución:
    1. MODO DEMO: parámetros fijos, población inicial generada aleatoriamente.
    2. MODO INTERACTIVO: usuario define tamaño de población, número de generaciones, probabilidad de mutación, etc.

Autor: Alejandro Aguirre Díaz
"""

import random

def generar_individuo(longitud=10, rango_valores=(0,1)):
    """
    Genera un individuo (solución candidata) representado como lista de bits 0/1 o valores en rango.
    :parametro longitud: número de genes
    :parametro rango_valores: tupla (min, max) para generación aleatoria de cada gen
    :return: lista que representa el cromosoma
    """
    # Crear un cromosoma con 'longitud' genes, cada uno con valor aleatorio en el rango especificado
    return [random.uniform(rango_valores[0], rango_valores[1]) for _ in range(longitud)]

def aptitud(individuo):
    """
    Función de aptitud (fitness) que queremos maximizar.
    Aquí: ejemplo f(x) = suma de genes (más alto mejor).
    :parametro individuo: lista de valores
    :return: valor numérico de aptitud
    """
    # Calcular la aptitud como la suma de todos los genes del individuo
    # Cuanto mayor sea la suma, mejor es la solución
    return sum(individuo)

def seleccion(poblacion, aptitudes, num_seleccionados):
    """
    Selecciona individuos según método de ruleta proporcional (probabilidad proporcional a aptitud).
    :parametro poblacion: lista de individuos
    :parametro aptitudes: lista de aptitudes correspondientes
    :parametro num_seleccionados: número de individuos a seleccionar
    :return: lista de individuos seleccionados
    """
    total = sum(aptitudes)
    seleccionados = []
    for _ in range(num_seleccionados):
        r = random.uniform(0, total)
        acumulado = 0
        for individuo, apt in zip(poblacion, aptitudes):
            acumulado += apt
            if acumulado >= r:
                seleccionados.append(individuo.copy())
                break
    return seleccionados

def cruce(padre1, padre2):
    """
    Cruce de un punto entre dos padres, devuelve dos hijos.
    :parametro padre1: lista
    :parametro padre2: lista
    :return: hijo1, hijo2
    """
    punto = random.randint(1, len(padre1)-1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

def mutacion(individuo, prob_mutacion=0.1, rango_mutacion=(-1,1)):
    """
    Aplica mutación aleatoria a cada gen con una probabilidad dada.
    :parametro individuo: lista de valores
    :parametro prob_mutacion: probabilidad de mutar cada gen
    :parametro rango_mutacion: intervalo de variación para la mutación
    :return: individuo mutado
    """
    nuevo = individuo.copy()
    for i in range(len(nuevo)):
        if random.random() < prob_mutacion:
            nuevo[i] += random.uniform(rango_mutacion[0], rango_mutacion[1])
    return nuevo

def algoritmo_genetico(tamaño_poblacion=20, num_generaciones=50, prob_mutacion=0.05, longitud_individuo=10):
    """
    Ejecuta el algoritmo genético completo.
    :parametro tamaño_poblacion: número de individuos por generación
    :parametro num_generaciones: cuántas generaciones correr
    :parametro prob_mutacion: probabilidad de mutación por gen
    :parametro longitud_individuo: longitud de cada cromosoma
    :return: mejor individuo encontrado y su aptitud
    """
    # Inicialización
    poblacion = [generar_individuo(longitud_individuo) for _ in range(tamaño_poblacion)]
    mejor = None
    mejor_apt = float('-inf')
    mejor_generacion = None  # para registrar en qué generación se encontró el mejor

    print("Gen | Mejor aptitud | Promedio aptitud")
    for gen in range(1, num_generaciones+1):
        aptitudes = [aptitud(ind) for ind in poblacion]
        promedio = sum(aptitudes) / len(aptitudes)
        max_apt = max(aptitudes)
        idx_max = aptitudes.index(max_apt)
        if max_apt > mejor_apt:
            mejor_apt = max_apt
            mejor = poblacion[idx_max].copy()
            mejor_generacion = gen  # guardar número de generación del mejor hasta ahora

        # Mostrar número de generación, mejor aptitud global hasta ahora y aptitud promedio actual
        print(f"{gen:3d} | {mejor_apt:.2f}     | {promedio:.2f}")

        # Selección
        padres = seleccion(poblacion, aptitudes, tamaño_poblacion // 2)
        # Reproducción / cruce
        hijos = []
        for i in range(0, len(padres), 2):
            if i+1 < len(padres):
                h1, h2 = cruce(padres[i], padres[i+1])
                hijos.append(h1)
                hijos.append(h2)
        # Mutación
        poblacion = [mutacion(h, prob_mutacion) for h in hijos]
        # Si quedan menos que el tamaño deseado lo completamos aleatoriamente
        while len(poblacion) < tamaño_poblacion:
            poblacion.append(generar_individuo(longitud_individuo))

    # Incluir también la generación en la que se encontró el mejor individuo
    print(f"\nMejor individuo encontrado (generación {mejor_generacion}): {mejor} | Aptitud = {mejor_apt:.2f}")
    return mejor, mejor_apt

def modo_demo():
    """Modo demostrativo con parámetros fijos."""
    print("\n--- MODO DEMO ---")
    tamaño_poblacion = 20
    num_generaciones = 30
    prob_mut = 0.1
    longitud_ind = 10
    print(f"Tamaño población = {tamaño_poblacion}, generaciones = {num_generaciones}, prob_mutación = {prob_mut}, longitud individuo = {longitud_ind}\n")
    algoritmo_genetico(tamaño_poblacion, num_generaciones, prob_mut, longitud_ind)

def modo_interactivo():
    """Modo interactivo para definir parámetros."""
    print("\n--- MODO INTERACTIVO ---")
    try:
        tamaño_poblacion = int(input("Introduce tamaño de la población: ").strip())
    except ValueError:
        tamaño_poblacion = 20
    try:
        num_generaciones = int(input("Introduce número de generaciones: ").strip())
    except ValueError:
        num_generaciones = 30
    try:
        prob_mut = float(input("Introduce probabilidad de mutación (0-1): ").strip())
        if not (0 <= prob_mut <= 1):
            raise ValueError
    except ValueError:
        prob_mut = 0.1
    try:
        longitud_ind = int(input("Introduce longitud del individuo (número de genes): ").strip())
    except ValueError:
        longitud_ind = 10

    print(f"\nTamaño población = {tamaño_poblacion}, generaciones = {num_generaciones}, prob_mutación = {prob_mut}, longitud individuo = {longitud_ind}\n")
    algoritmo_genetico(tamaño_poblacion, num_generaciones, prob_mut, longitud_ind)

def main():
    print("Seleccione modo de ejecución:")
    print("1. Modo DEMO")
    print("2. Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()

    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
