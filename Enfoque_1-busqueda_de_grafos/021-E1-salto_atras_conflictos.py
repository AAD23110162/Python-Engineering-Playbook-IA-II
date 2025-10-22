"""
021-E1-salto_atras_conflictos.py
--------------------------------
Este script implementa la técnica de Salto Atrás Dirigido por Conflictos (Conflict-Directed Backjumping, CBJ)
en el contexto de un Problema de Satisfacción de Restricciones (CSP):
- Cuando se encuentra un "callejón sin salida" (dead-end), en lugar de retroceder simplemente a la última variable asignada,
  analiza los conjuntos de conflicto acumulados para **saltar directamente** a una variable previa que causó el conflicto.
- Esto puede reducir significativamente el espacio de búsqueda al evitar explorar ramas que están condenadas al fracaso.
- Incluye dos modos:
    1. MODO DEMO: problema sencillo predefinido que muestra saltos.
    2. MODO INTERACTIVO: usuario selecciona entre datasets predefinidos.
- Muestra paso a paso qué variable se asigna, qué valor, qué conflictos se detectan, y hacia dónde se salta.

Autor: Alejandro Aguirre Díaz
"""

# Contador global de nodos explorados (mide el esfuerzo de búsqueda)
# Se incrementa cada vez que exploramos un nodo (asignación parcial)
nodos_explorados = 0

def asignacion_completa(asignacion, variables):
    """
    Verifica si todas las variables han sido asignadas.
    :parametro asignacion: dict variable->valor asignado
    :parametro variables: lista de variables del CSP
    :return: True si todas las variables tienen valor, False en caso contrario
    """
    # Devuelve True solo si cada variable en 'variables' tiene un valor en 'asignacion'
    return all(var in asignacion for var in variables)

def consistente(asignacion, variable, valor, restricciones, conflicto_set):
    """
    Comprueba si asignar 'valor' a 'variable' es consistente con las variables ya asignadas.
    En caso de inconsistencia, registra en conflicto_set las variables que causan el conflicto.
    
    :parametro asignacion: dict variable->valor asignado
    :parametro variable: variable que se va a asignar
    :parametro valor: valor candidato para la variable
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :parametro conflicto_set: set que acumula variables en conflicto
    :return: True si consistente, False si viola alguna restricción
    """
    # Recorrer todas las restricciones binarias del problema
    for (v1, v2, restr) in restricciones:
        # Caso 1: la variable actual es v1 en la restricción (v1, v2)
        if variable == v1 and v2 in asignacion:
            # Verificar si la restricción se cumple entre el valor candidato y el valor asignado a v2
            if not restr(valor, asignacion[v2]):
                # Si no se cumple, registrar v2 como causa del conflicto
                conflicto_set.add(v2)
                return False
        
        # Caso 2: la variable actual es v2 en la restricción (v1, v2)
        if variable == v2 and v1 in asignacion:
            # Verificar si la restricción se cumple entre el valor asignado a v1 y el valor candidato
            if not restr(asignacion[v1], valor):
                # Si no se cumple, registrar v1 como causa del conflicto
                conflicto_set.add(v1)
                return False
    
    # Si ninguna restricción se violó, la asignación es consistente
    return True

def backjumping_conflictos(asignacion, variables, dominios, restricciones, nivel=0):
    """
    Implementa backtracking con salto atrás dirigido por conflictos (CBJ).
    
    La diferencia clave con backtracking cronológico:
    - En lugar de retornar None y retroceder un nivel, retorna un conjunto de conflicto
    - Este conjunto se propaga hacia arriba hasta la variable culpable
    - Permite saltar varios niveles de una vez
    
    :parametro asignacion: dict variable->valor asignado
    :parametro variables: lista de variables del CSP
    :parametro dominios: dict variable->lista de valores posibles
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :parametro nivel: profundidad actual en el árbol de búsqueda (para visualización)
    :return: tupla (solución, conjunto_conflicto) donde:
             - solución es dict o None
             - conjunto_conflicto es set de variables culpables del fallo
    """
    global nodos_explorados
    nodos_explorados += 1
    
    # Indentación proporcional al nivel para visualizar el árbol de búsqueda
    indent = "  " * nivel
    
    # Mostrar estado actual: asignación parcial
    print(f"{indent}[Nodo {nodos_explorados}] Nivel {nivel}: Asignación = {asignacion}")
    
    # Caso base: si todas las variables están asignadas, tenemos una solución completa
    if asignacion_completa(asignacion, variables):
        print(f"{indent}[EXITO] Solución completa encontrada: {asignacion}")
        return (asignacion, set())  # Retornar solución con conjunto de conflicto vacío
    
    # Seleccionar la siguiente variable sin asignar
    # Nota: aquí usamos el primer no asignado, pero podríamos usar heurísticas (MRV, grado, etc.)
    var = next(v for v in variables if v not in asignacion)
    print(f"{indent}Seleccionando variable '{var}' para asignar...")
    
    # Conjunto de conflicto acumulado para esta variable
    # Guardará todas las variables que causaron fallos en todos los intentos de asignación
    conflicto_acumulado = set()
    
    # Probar cada valor del dominio de la variable actual
    for valor in dominios[var]:
        print(f"{indent}  Probando {var} = {valor}")
        
        # Conjunto de conflicto para este intento específico
        conflicto_local = set()
        
        # Verificar si este valor es consistente con las asignaciones previas
        if consistente(asignacion, var, valor, restricciones, conflicto_local):
            print(f"{indent}    {var} = {valor} es consistente con asignaciones previas")
            
            # Hacer la asignación tentativa
            asignacion[var] = valor
            
            # Llamada recursiva para continuar con la siguiente variable
            resultado, conflicto_hijo = backjumping_conflictos(
                asignacion, variables, dominios, restricciones, nivel + 1
            )
            
            # Si se encontró una solución en la recursión, propagarla hacia arriba
            if resultado is not None:
                return (resultado, set())
            
            # Si la recursión falló, analizar el conjunto de conflicto retornado
            print(f"{indent}    Fallo en recursión. Conflicto propagado desde hijo: {conflicto_hijo}")
            
            # Verificar si el conflicto incluye la variable actual
            if var in conflicto_hijo:
                # Si la variable actual está en el conflicto, removerla
                # (ya estamos retrocediendo desde ella)
                conflicto_hijo.remove(var)
                # Agregar este conflicto al acumulado
                conflicto_acumulado.update(conflicto_hijo)
                print(f"{indent}    Variable actual '{var}' está en conflicto, continuando con siguiente valor...")
            else:
                # Si la variable actual NO está en el conflicto hijo,
                # significa que debemos saltar hacia atrás más allá de esta variable
                print(f"{indent}    [SALTO] Conflicto no incluye '{var}', saltando hacia atrás hasta: {conflicto_hijo}")
                # Deshacer la asignación actual
                del asignacion[var]
                # Propagar el conflicto hacia arriba (salto atrás)
                return (None, conflicto_hijo)
            
            # Deshacer la asignación para probar el siguiente valor
            del asignacion[var]
        else:
            # El valor no es consistente, registrar las variables conflictivas
            print(f"{indent}    {var} = {valor} NO es consistente. Conflicto con: {conflicto_local}")
            # Acumular los conflictos de este intento
            conflicto_acumulado.update(conflicto_local)
    
    # Si llegamos aquí, ningún valor del dominio funcionó para esta variable
    print(f"{indent}[FALLO] Todos los valores de {var} agotados. Conflicto acumulado: {conflicto_acumulado}")
    
    # Agregar la variable actual al conjunto de conflicto antes de propagarlo
    conflicto_acumulado.add(var)
    
    # Retornar fallo con el conjunto de conflicto acumulado
    return (None, conflicto_acumulado)

def modo_demo():
    """
    Modo demostrativo con un problema predefinido que muestra claramente el CBJ.
    Problema: 4 variables con dominios pequeños y restricciones que fuerzan saltos.
    """
    print("\n" + "="*70)
    print("--- MODO DEMO: Conflict-Directed Backjumping ---")
    print("="*70)
    
    # Definir variables del CSP
    variables = ['A', 'B', 'C', 'D']
    
    # Definir dominios para cada variable
    # Dominios pequeños para forzar conflictos y mostrar saltos
    dominios = {
        'A': [1, 2],
        'B': [1, 2],
        'C': [1, 2],
        'D': [1, 2]
    }
    
    # Definir restricciones binarias
    # Restricción: todas las variables deben ser diferentes
    # Además, A y D deben tener la misma paridad (ambas pares o ambas impares)
    restricciones = [
        ('A', 'B', lambda x, y: x != y),  # A diferente de B
        ('A', 'C', lambda x, y: x != y),  # A diferente de C
        ('B', 'C', lambda x, y: x != y),  # B diferente de C
        ('B', 'D', lambda x, y: x != y),  # B diferente de D
        ('C', 'D', lambda x, y: x != y),  # C diferente de D
        ('A', 'D', lambda x, y: (x % 2) == (y % 2))  # A y D misma paridad
    ]
    
    print("\nProblema:")
    print("  Variables: A, B, C, D")
    print("  Dominios: {1, 2} para cada variable")
    print("  Restricciones:")
    print("    - Todas diferentes entre sí")
    print("    - A y D deben tener la misma paridad")
    print("\nEste problema NO tiene solución, verás cómo CBJ detecta conflictos y salta.\n")
    
    # Reiniciar contador de nodos
    global nodos_explorados
    nodos_explorados = 0
    
    # Ejecutar el algoritmo CBJ
    solucion, conflicto_final = backjumping_conflictos({}, variables, dominios, restricciones)
    
    # Mostrar estadísticas finales
    print("\n" + "="*70)
    print(f"Total de nodos explorados: {nodos_explorados}")
    print("="*70)
    
    if solucion:
        print("[EXITO] Solución encontrada:", solucion)
    else:
        print("[FALLO] No existe solución para este CSP.")
        print(f"Conjunto de conflicto final: {conflicto_final}")

def modo_interactivo():
    """
    Modo interactivo donde el usuario selecciona entre datasets predefinidos.
    """
    print("\n" + "="*70)
    print("--- MODO INTERACTIVO: Seleccione un problema ---")
    print("="*70)
    
    print("\nDatasets disponibles:")
    print("1) All-Different simple (3 variables, dominios {1,2,3}) - Tiene solución")
    print("2) Coloreado de mapa (4 regiones en ciclo, 3 colores) - Tiene solución")
    print("3) Problema imposible (4 variables, conflictos forzados) - Sin solución")
    
    opcion = input("\nIntroduce el número del dataset: ").strip()
    
    if opcion == '1':
        # Dataset 1: All-Different simple con solución
        variables = ['X', 'Y', 'Z']
        dominios = {v: [1, 2, 3] for v in variables}
        restricciones = [
            ('X', 'Y', lambda x, y: x != y),
            ('Y', 'Z', lambda x, y: x != y),
            ('X', 'Z', lambda x, y: x != y)
        ]
        print("\n[Dataset 1] All-Different: X, Y, Z ∈ {1,2,3}, todas diferentes.")
        
    elif opcion == '2':
        # Dataset 2: Coloreado de mapa con 4 regiones
        variables = ['R1', 'R2', 'R3', 'R4']
        colores = ['Rojo', 'Verde', 'Azul']
        dominios = {v: colores.copy() for v in variables}
        restricciones = [
            ('R1', 'R2', lambda x, y: x != y),  # R1 y R2 son adyacentes
            ('R2', 'R3', lambda x, y: x != y),  # R2 y R3 son adyacentes
            ('R3', 'R4', lambda x, y: x != y),  # R3 y R4 son adyacentes
            ('R4', 'R1', lambda x, y: x != y)   # R4 y R1 son adyacentes (ciclo)
        ]
        print("\n[Dataset 2] Coloreado de 4 regiones en ciclo con 3 colores.")
        
    elif opcion == '3':
        # Dataset 3: Problema sin solución para demostrar CBJ
        variables = ['A', 'B', 'C', 'D']
        dominios = {
            'A': [1, 2],
            'B': [1, 2],
            'C': [1, 2],
            'D': [1, 2]
        }
        restricciones = [
            ('A', 'B', lambda x, y: x != y),
            ('A', 'C', lambda x, y: x != y),
            ('B', 'C', lambda x, y: x != y),
            ('B', 'D', lambda x, y: x != y),
            ('C', 'D', lambda x, y: x != y),
            ('A', 'D', lambda x, y: (x % 2) == (y % 2))  # Paridad forzada
        ]
        print("\n[Dataset 3] Problema sin solución con conflictos de paridad.")
        
    else:
        # Opción inválida, usar dataset 1 por defecto
        print("\nOpción no válida. Usando Dataset 1 por defecto.")
        variables = ['X', 'Y', 'Z']
        dominios = {v: [1, 2, 3] for v in variables}
        restricciones = [
            ('X', 'Y', lambda x, y: x != y),
            ('Y', 'Z', lambda x, y: x != y),
            ('X', 'Z', lambda x, y: x != y)
        ]
        print("\n[Dataset 1] All-Different: X, Y, Z ∈ {1,2,3}, todas diferentes.")
    
    # Reiniciar contador de nodos
    global nodos_explorados
    nodos_explorados = 0
    
    # Ejecutar el algoritmo CBJ
    print("\nIniciando búsqueda con Conflict-Directed Backjumping...\n")
    solucion, conflicto_final = backjumping_conflictos({}, variables, dominios, restricciones)
    
    # Mostrar estadísticas finales
    print("\n" + "="*70)
    print(f"Total de nodos explorados: {nodos_explorados}")
    print("="*70)
    
    if solucion:
        print("[EXITO] Solución encontrada:", solucion)
    else:
        print("[FALLO] No existe solución para este CSP.")
        print(f"Conjunto de conflicto final: {conflicto_final}")

def main():
    """
    Función principal que presenta el menú y ejecuta el modo seleccionado.
    """
    print("\n" + "="*70)
    print("CONFLICT-DIRECTED BACKJUMPING (CBJ) para CSP")
    print("="*70)
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (problema predefinido que muestra saltos)")
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
