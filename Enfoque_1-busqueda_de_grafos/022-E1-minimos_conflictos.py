"""
022-E1-minimos_conflictos.py
--------------------------------
Este script implementa el algoritmo de Mínimos Conflictos (Min-Conflicts) para resolver CSP:
- Es un algoritmo de **reparación iterativa** que comienza con una asignación completa (posiblemente inconsistente)
- En cada iteración, selecciona una variable en conflicto y le asigna el valor que minimiza el número de conflictos
- Es especialmente eficiente para problemas como N-Reinas y es el método de elección para CSPs grandes
- A diferencia del backtracking, no construye soluciones parciales sino que repara soluciones completas
- Incluye dos modos:
    1. MODO DEMO: problema N-Reinas (8 reinas) que muestra el proceso de reparación
    2. MODO INTERACTIVO: usuario selecciona entre varios problemas predefinidos
- Muestra paso a paso qué variable se selecciona, qué conflictos tiene, y cómo se repara

Autor: Alejandro Aguirre Díaz
"""

import random

# Contador global de iteraciones (mide el esfuerzo de búsqueda)
# Se incrementa cada vez que reparamos una variable en conflicto
iteraciones = 0

def asignacion_completa_aleatoria(variables, dominios):
    """
    Genera una asignación completa aleatoria para todas las variables.
    Esta es la asignación inicial desde la cual comenzará la reparación.
    
    :parametro variables: lista de variables del CSP
    :parametro dominios: dict variable->lista de valores posibles
    :return: dict con asignación variable->valor aleatorio
    """
    # Para cada variable, seleccionar aleatoriamente un valor de su dominio
    asignacion = {}
    for var in variables:
        # random.choice selecciona un elemento aleatorio del dominio de la variable
        asignacion[var] = random.choice(dominios[var])
    
    return asignacion

def contar_conflictos(asignacion, variable, valor, restricciones):
    """
    Cuenta cuántas restricciones viola una variable si se le asigna un valor específico.
    
    :parametro asignacion: dict variable->valor asignado
    :parametro variable: variable que estamos evaluando
    :parametro valor: valor candidato para la variable
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :return: número de conflictos (restricciones violadas)
    """
    # Contador de conflictos inicializado a cero
    conflictos = 0
    
    # Guardar temporalmente el valor actual de la variable
    valor_original = asignacion.get(variable)
    # Asignar temporalmente el nuevo valor para evaluación
    asignacion[variable] = valor
    
    # Recorrer todas las restricciones binarias del problema
    for (v1, v2, restr) in restricciones:
        # Verificar si esta restricción involucra a nuestra variable
        # y si la otra variable también está asignada
        if (variable == v1 and v2 in asignacion) or (variable == v2 and v1 in asignacion):
            # Asegurarse de que ambas variables estén asignadas
            if v1 in asignacion and v2 in asignacion:
                # Verificar si la restricción se viola
                if not restr(asignacion[v1], asignacion[v2]):
                    conflictos += 1
    
    # Restaurar el valor original de la variable
    if valor_original is not None:
        asignacion[variable] = valor_original
    else:
        # Si la variable no tenía valor antes, removerla
        del asignacion[variable]
    
    return conflictos

def obtener_variables_en_conflicto(asignacion, variables, restricciones):
    """
    Identifica todas las variables que están involucradas en al menos un conflicto.
    
    :parametro asignacion: dict variable->valor asignado
    :parametro variables: lista de variables del CSP
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :return: lista de variables en conflicto
    """
    # Conjunto para almacenar variables conflictivas (evita duplicados)
    en_conflicto = set()
    
    # Recorrer todas las restricciones para identificar las violadas
    for (v1, v2, restr) in restricciones:
        # Verificar si ambas variables están asignadas
        if v1 in asignacion and v2 in asignacion:
            # Si la restricción no se cumple, ambas variables están en conflicto
            if not restr(asignacion[v1], asignacion[v2]):
                en_conflicto.add(v1)
                en_conflicto.add(v2)
    
    # Convertir el conjunto a lista y retornar
    return list(en_conflicto)

def valor_minimos_conflictos(asignacion, variable, dominio, restricciones):
    """
    Encuentra el valor del dominio que minimiza el número de conflictos para una variable.
    Si hay empate, selecciona aleatoriamente entre los valores empatados.
    
    :parametro asignacion: dict variable->valor asignado
    :parametro variable: variable que queremos reparar
    :parametro dominio: lista de valores posibles para esta variable
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :return: valor que minimiza conflictos
    """
    # Calcular el número de conflictos para cada valor del dominio
    conflictos_por_valor = []
    
    for valor in dominio:
        # Contar conflictos si asignamos este valor a la variable
        num_conflictos = contar_conflictos(asignacion, variable, valor, restricciones)
        conflictos_por_valor.append((valor, num_conflictos))
    
    # Encontrar el mínimo número de conflictos
    min_conflictos = min(conflictos_por_valor, key=lambda x: x[1])[1]
    
    # Obtener todos los valores que tienen el mínimo número de conflictos
    mejores_valores = [valor for valor, conf in conflictos_por_valor if conf == min_conflictos]
    
    # Si hay empate, seleccionar aleatoriamente (esto ayuda a escapar de mínimos locales)
    return random.choice(mejores_valores)

def minimos_conflictos(variables, dominios, restricciones, max_iteraciones=1000, verbose=True):
    """
    Implementa el algoritmo de Mínimos Conflictos para resolver un CSP.
    
    Estrategia:
    1. Comenzar con una asignación completa aleatoria
    2. Mientras haya conflictos y no se alcance el máximo de iteraciones:
       a. Seleccionar aleatoriamente una variable en conflicto
       b. Asignarle el valor que minimiza el número de conflictos
    3. Si no hay conflictos, retornar la solución
    
    :parametro variables: lista de variables del CSP
    :parametro dominios: dict variable->lista de valores posibles
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :parametro max_iteraciones: número máximo de iteraciones permitidas
    :parametro verbose: si True, muestra información paso a paso
    :return: tupla (solución, exito) donde solución es dict y exito es booleano
    """
    global iteraciones
    iteraciones = 0
    
    # Paso 1: Generar una asignación completa aleatoria inicial
    asignacion = asignacion_completa_aleatoria(variables, dominios)
    
    if verbose:
        print(f"\n[Inicio] Asignación aleatoria inicial: {asignacion}")
    
    # Paso 2: Iterar hasta encontrar solución o alcanzar máximo de iteraciones
    for i in range(max_iteraciones):
        iteraciones += 1
        
        # Identificar variables en conflicto
        vars_conflicto = obtener_variables_en_conflicto(asignacion, variables, restricciones)
        
        # Si no hay conflictos, hemos encontrado una solución
        if not vars_conflicto:
            if verbose:
                print(f"\n[EXITO] Solución encontrada en iteración {iteraciones}")
                print(f"Asignación final: {asignacion}")
            return (asignacion, True)
        
        if verbose:
            print(f"\n[Iteración {iteraciones}] Variables en conflicto: {vars_conflicto} (total: {len(vars_conflicto)})")
        
        # Seleccionar aleatoriamente una variable en conflicto
        var = random.choice(vars_conflicto)
        
        # Calcular el número de conflictos actual de esta variable
        conflictos_actuales = contar_conflictos(asignacion, var, asignacion[var], restricciones)
        
        if verbose:
            print(f"  → Variable seleccionada: '{var}' (valor actual: {asignacion[var]}, conflictos: {conflictos_actuales})")
        
        # Encontrar el valor que minimiza conflictos para esta variable
        mejor_valor = valor_minimos_conflictos(asignacion, var, dominios[var], restricciones)
        
        # Calcular conflictos con el nuevo valor
        nuevos_conflictos = contar_conflictos(asignacion, var, mejor_valor, restricciones)
        
        if verbose:
            print(f"  → Asignando '{var}' = {mejor_valor} (conflictos: {nuevos_conflictos})")
        
        # Realizar la asignación (reparación)
        asignacion[var] = mejor_valor
    
    # Si llegamos aquí, alcanzamos el máximo de iteraciones sin encontrar solución
    if verbose:
        print(f"\n[FALLO] Se alcanzó el máximo de iteraciones ({max_iteraciones})")
        vars_conflicto_final = obtener_variables_en_conflicto(asignacion, variables, restricciones)
        print(f"Variables aún en conflicto: {len(vars_conflicto_final)}")
    
    return (asignacion, False)

def modo_demo():
    """
    Modo demostrativo con el problema de las N-Reinas (8 reinas).
    Este es un problema clásico donde Min-Conflicts es muy eficiente.
    """
    print("\n" + "="*70)
    print("--- MODO DEMO: Problema de las 8 Reinas ---")
    print("="*70)
    
    # Definir el problema de las 8 reinas
    n = 8
    variables = [f'Q{i}' for i in range(n)]  # Q0, Q1, ..., Q7
    
    # Cada reina (variable) puede estar en cualquier fila (dominio {0,1,...,7})
    dominios = {var: list(range(n)) for var in variables}
    
    # Restricciones: ninguna reina puede atacar a otra
    # Dos reinas se atacan si están en la misma fila, columna o diagonal
    restricciones = []
    for i in range(n):
        for j in range(i + 1, n):
            var_i = f'Q{i}'
            var_j = f'Q{j}'
            # Restricción: no pueden estar en la misma fila ni en la misma diagonal
            # var_i representa la columna i, su valor es la fila
            # var_j representa la columna j, su valor es la fila
            def no_ataque(fila_i, fila_j, col_i=i, col_j=j):
                # Misma fila
                if fila_i == fila_j:
                    return False
                # Misma diagonal (diferencia de filas = diferencia de columnas)
                if abs(fila_i - fila_j) == abs(col_i - col_j):
                    return False
                return True
            
            restricciones.append((var_i, var_j, no_ataque))
    
    print("\nProblema:")
    print(f"  Colocar {n} reinas en un tablero de {n}x{n} sin que se ataquen")
    print(f"  Variables: {n} reinas (Q0 a Q7), cada una en una columna")
    print(f"  Dominio: cada reina puede estar en cualquier fila (0-7)")
    print(f"  Restricciones: no pueden atacarse (misma fila, columna o diagonal)")
    print(f"\nMin-Conflicts es muy eficiente para este problema.\n")
    
    # Ejecutar el algoritmo de mínimos conflictos
    solucion, exito = minimos_conflictos(variables, dominios, restricciones, max_iteraciones=1000, verbose=True)
    
    # Mostrar estadísticas finales
    print("\n" + "="*70)
    print(f"Total de iteraciones: {iteraciones}")
    print("="*70)
    
    if exito:
        print("\n[EXITO] Solución encontrada:")
        # Mostrar el tablero de forma visual
        print("\nTablero (cada número indica la fila de la reina en esa columna):")
        for i in range(n):
            fila = solucion[f'Q{i}']
            linea = ['.' if j != fila else 'Q' for j in range(n)]
            print(f"  Fila {i}: " + " ".join(linea))
        print(f"\nAsignación: {solucion}")
    else:
        print("\n[FALLO] No se encontró solución en el límite de iteraciones.")
        print("Intenta ejecutar de nuevo (el algoritmo es probabilístico).")

def modo_interactivo():
    """
    Modo interactivo donde el usuario selecciona entre datasets predefinidos.
    """
    print("\n" + "="*70)
    print("--- MODO INTERACTIVO: Seleccione un problema ---")
    print("="*70)
    
    print("\nDatasets disponibles:")
    print("1) N-Reinas=4")
    print("2) N-Reinas=8")
    print("3) All-Different (5 variables, dominio {1,2,3,4,5})")
    print("4) Coloreado de mapa (4 regiones en ciclo, 3 colores)")
    
    opcion = input("\nIntroduce el número del dataset: ").strip()
    
    if opcion == '1':
        # Dataset 1: 4 reinas
        n = 4
        variables = [f'Q{i}' for i in range(n)]
        dominios = {var: list(range(n)) for var in variables}
        restricciones = []
        for i in range(n):
            for j in range(i + 1, n):
                var_i = f'Q{i}'
                var_j = f'Q{j}'
                def no_ataque(fila_i, fila_j, col_i=i, col_j=j):
                    if fila_i == fila_j:
                        return False
                    if abs(fila_i - fila_j) == abs(col_i - col_j):
                        return False
                    return True
                restricciones.append((var_i, var_j, no_ataque))
        print(f"\n[Dataset 1] {n}-Reinas: tablero {n}x{n}")
        
    elif opcion == '2':
        # Dataset 2: 8 reinas
        n = 8
        variables = [f'Q{i}' for i in range(n)]
        dominios = {var: list(range(n)) for var in variables}
        restricciones = []
        for i in range(n):
            for j in range(i + 1, n):
                var_i = f'Q{i}'
                var_j = f'Q{j}'
                def no_ataque(fila_i, fila_j, col_i=i, col_j=j):
                    if fila_i == fila_j:
                        return False
                    if abs(fila_i - fila_j) == abs(col_i - col_j):
                        return False
                    return True
                restricciones.append((var_i, var_j, no_ataque))
        print(f"\n[Dataset 2] {n}-Reinas: tablero {n}x{n}")
        
    elif opcion == '3':
        # Dataset 3: All-Different
        variables = ['X1', 'X2', 'X3', 'X4', 'X5']
        dominios = {v: [1, 2, 3, 4, 5] for v in variables}
        restricciones = []
        # Todas las variables deben ser diferentes
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                restricciones.append((variables[i], variables[j], lambda x, y: x != y))
        print("\n[Dataset 3] All-Different: 5 variables con dominio {1,2,3,4,5}, todas diferentes")
        
    elif opcion == '4':
        # Dataset 4: Coloreado de mapa
        variables = ['R1', 'R2', 'R3', 'R4']
        colores = ['Rojo', 'Verde', 'Azul']
        dominios = {v: colores.copy() for v in variables}
        restricciones = [
            ('R1', 'R2', lambda x, y: x != y),  # R1 y R2 son adyacentes
            ('R2', 'R3', lambda x, y: x != y),  # R2 y R3 son adyacentes
            ('R3', 'R4', lambda x, y: x != y),  # R3 y R4 son adyacentes
            ('R4', 'R1', lambda x, y: x != y)   # R4 y R1 son adyacentes (ciclo)
        ]
        print("\n[Dataset 4] Coloreado de 4 regiones en ciclo con 3 colores")
        
    else:
        # Opción inválida, usar dataset 2 por defecto
        print("\nOpción no válida. Usando Dataset 2 (8-Reinas) por defecto.")
        n = 8
        variables = [f'Q{i}' for i in range(n)]
        dominios = {var: list(range(n)) for var in variables}
        restricciones = []
        for i in range(n):
            for j in range(i + 1, n):
                var_i = f'Q{i}'
                var_j = f'Q{j}'
                def no_ataque(fila_i, fila_j, col_i=i, col_j=j):
                    if fila_i == fila_j:
                        return False
                    if abs(fila_i - fila_j) == abs(col_i - col_j):
                        return False
                    return True
                restricciones.append((var_i, var_j, no_ataque))
        print(f"\n[Dataset 2] {n}-Reinas: tablero {n}x{n}")
    
    # Preguntar al usuario el máximo de iteraciones
    try:
        max_iter = int(input("\n¿Máximo de iteraciones? (default 1000): ").strip() or "1000")
    except ValueError:
        max_iter = 1000
        print("Entrada inválida, usando 1000 iteraciones.")
    
    # Ejecutar el algoritmo de mínimos conflictos
    print("\nIniciando búsqueda con Mínimos Conflictos...\n")
    solucion, exito = minimos_conflictos(variables, dominios, restricciones, max_iteraciones=max_iter, verbose=True)
    
    # Mostrar estadísticas finales
    print("\n" + "="*70)
    print(f"Total de iteraciones: {iteraciones}")
    print("="*70)
    
    if exito:
        print("\n[EXITO] Solución encontrada:", solucion)
    else:
        print("\n[FALLO] No se encontró solución en el límite de iteraciones.")
        print("Intenta ejecutar de nuevo o aumenta el número de iteraciones.")

def main():
    """
    Función principal que presenta el menú y ejecuta el modo seleccionado.
    """
    print("\n" + "="*70)
    print("ALGORITMO DE MÍNIMOS CONFLICTOS (Min-Conflicts) para CSP")
    print("="*70)
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (8-Reinas con reparación iterativa)")
    print("2) Modo INTERACTIVO (seleccionar entre varios problemas)\n")
    
    opcion = input("Ingrese el número de opción: ").strip()
    
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("\nOpción no válida. Ejecutando modo DEMO por defecto.\n")
        modo_demo()

# Punto de entrada del programa
if __name__ == "__main__":
    main()
