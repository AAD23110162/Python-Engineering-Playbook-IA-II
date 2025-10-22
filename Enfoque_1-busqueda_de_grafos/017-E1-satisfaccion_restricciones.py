"""
017-E1-satisfaccion_restricciones.py
------------------------------------
Este script implementa una versión simplificada de un Problema de Satisfacción de Restricciones (CSP):
- Define un conjunto de variables, cada una con su dominio de posibles valores.
- Define restricciones binarias entre variables.
- Usa un algoritmo de retroceso (backtracking) con “chequeo hacia delante” (forward checking) para encontrar una asignación que cumpla todas las restricciones.
- Muestra estadísticas de nodos explorados.
- Incluye dos modos de ejecución:
    1. MODO DEMO: problema predefinido.
    2. MODO INTERACTIVO: el usuario selecciona un dataset predefinido.

Autor: Alejandro Aguirre Díaz
"""

def asignacion_completa(asignacion, variables):
    # Verificar si todas las variables ya tienen un valor asignado
    # Devuelve True si cada variable en 'variables' aparece como clave en 'asignacion'
    return all(var in asignacion for var in variables)

def consistente(asignacion, variable, valor, restricciones):
    # Comprobar que asignar 'valor' a 'variable' no viola ninguna restricción con variables ya asignadas
    # Las restricciones son binarias (entre pares de variables)
    for (v1, v2, restr) in restricciones:
        if variable == v1 and v2 in asignacion:
            # Si la restricción es (variable, v2), verificar que restr(valor, asignacion[v2]) sea True
            if not restr(valor, asignacion[v2]):
                return False
        if variable == v2 and v1 in asignacion:
            # Si la restricción es (v1, variable), verificar que restr(asignacion[v1], valor) sea True
            if not restr(asignacion[v1], valor):
                return False
    return True

def forward_checking(dominios, variable, valor, restricciones):
    # Aplicar "chequeo hacia delante": reducir dominios de variables no asignadas
    # en base a la nueva asignación (variable = valor)
    nuevos_dom = {v: list(dominios[v]) for v in dominios}
    for (v1, v2, restr) in restricciones:
        if v1 == variable and v2 in nuevos_dom:
            # Mantener en el dominio de v2 solo valores compatibles con la restricción restr(valor, y)
            nuevos_dom[v2] = [y for y in nuevos_dom[v2] if restr(valor, y)]
            if not nuevos_dom[v2]:
                # Si algún dominio queda vacío, no hay solución bajo esta asignación
                return None
        if v2 == variable and v1 in nuevos_dom:
            # Mantener en el dominio de v1 solo valores compatibles con la restricción restr(x, valor)
            nuevos_dom[v1] = [x for x in nuevos_dom[v1] if restr(x, valor)]
            if not nuevos_dom[v1]:
                return None
    return nuevos_dom

# Global variable para contar nodos explorados
nodos_explorados = 0

def backtracking_fc(asignacion, variables, dominios, restricciones):
    global nodos_explorados
    # Incrementar conteo de nodos explorados cada vez que entramos a esta función
    nodos_explorados += 1

    # Caso base: si todas las variables están asignadas, devolvemos la asignación completa
    if asignacion_completa(asignacion, variables):
        return asignacion

    # Selección de variable sin asignar (estrategia simple)
    # Aquí elegimos la primera variable no asignada; se podría mejorar con heurísticas (MRV, Grado, etc.)
    var = next(v for v in variables if v not in asignacion)

    for valor in dominios[var]:
        # Probar asignar un valor del dominio actual de 'var'
        if consistente(asignacion, var, valor, restricciones):
            # Si es consistente con las asignaciones actuales, asignamos tentativamente
            asignacion[var] = valor
            # Reducimos dominios de variables relacionadas mediante forward checking
            nuevos_dom = forward_checking(dominios, var, valor, restricciones)
            if nuevos_dom is not None:
                # Continuamos con la siguiente variable
                resultado = backtracking_fc(asignacion, variables, nuevos_dom, restricciones)
                if resultado is not None:
                    # Si se encontró solución en el subárbol, propagarla hacia arriba
                    return resultado
            # Si no hubo solución o forward checking vació algún dominio, deshacer asignación (backtrack)
            del asignacion[var]
    return None

def modo_demo():
    print("\n--- MODO DEMO ---")
    # Definir variables y dominios para un problema all-different simple
    variables = ['X1', 'X2', 'X3']
    dominios = {
        'X1': [1, 2, 3],
        'X2': [1, 2, 3],
        'X3': [1, 2, 3]
    }
    # Restricciones: todas las variables deben tomar valores distintos entre sí
    restricciones = [
        ('X1', 'X2', lambda x, y: x != y),
        ('X1', 'X3', lambda x, y: x != y),
        ('X2', 'X3', lambda x, y: x != y)
    ]
    print("Problema: variables X1, X2, X3 con dominio {1,2,3} y restricción all-different.\n")
    global nodos_explorados
    nodos_explorados = 0
    # Ejecutar backtracking con forward checking
    solucion = backtracking_fc({}, variables, dominios, restricciones)
    print(f"Nodos explorados: {nodos_explorados}")
    if solucion:
        # Mostrar asignación completa si se encontró
        print("Solución encontrada:", solucion)
    else:
        print("No se encontró solución.")

def modo_interactivo():
    print("\n--- MODO INTERACTIVO ---")
    # Permitir seleccionar entre datasets predefinidos para CSP
    print("Seleccione un dataset:")
    print("1) Dataset 1: All-Different 3 variables (igual que demo)")
    print("2) Dataset 2: Colores de mapa (4 regiones, 3 colores)")
    print("3) Dataset 3: Sudoku simplificado (4 variables, dominio {1,2})")
    opcion = input("Introduce el número de dataset: ").strip()

    if opcion == '1':
        variables = ['X1', 'X2', 'X3']
        dominios = {
            'X1': [1,2,3],
            'X2': [1,2,3],
            'X3': [1,2,3]
        }
        restricciones = [
            ('X1','X2', lambda x,y: x!=y),
            ('X1','X3', lambda x,y: x!=y),
            ('X2','X3', lambda x,y: x!=y)
        ]
    elif opcion == '2':
        # Problema de coloreado de mapa en ciclo de 4 regiones
        variables = ['R1','R2','R3','R4']
        dominios = {
            'R1': ['Rojo','Verde','Azul'],
            'R2': ['Rojo','Verde','Azul'],
            'R3': ['Rojo','Verde','Azul'],
            'R4': ['Rojo','Verde','Azul']
        }
        restricciones = [
            ('R1','R2', lambda x,y: x!=y),
            ('R2','R3', lambda x,y: x!=y),
            ('R3','R4', lambda x,y: x!=y),
            ('R4','R1', lambda x,y: x!=y)
        ]
        print("Problema: 4 regiones conectadas en ciclo, color distinto para vecinos.\n")
    elif opcion == '3':
        # Sudoku 2x2 simplificado con restricciones de adyacencia distinta
        variables = ['A','B','C','D']
        dominios = {
            'A': [1,2],
            'B': [1,2],
            'C': [1,2],
            'D': [1,2]
        }
        restricciones = [
            ('A','B', lambda x,y: x!=y),
            ('A','C', lambda x,y: x!=y),
            ('B','D', lambda x,y: x!=y),
            ('C','D', lambda x,y: x!=y)
        ]
        print("Problema: Sudoku simplificado 2×2 (variables A-D, dominio {1,2}, restricción de adyacencia distinta).\n")
    else:
        # Opción inválida: usar dataset por defecto igual que demo
        print("Opción no válida. Se usará dataset 1.")
        variables = ['X1', 'X2', 'X3']
        dominios = {
            'X1': [1,2,3],
            'X2': [1,2,3],
            'X3': [1,2,3]
        }
        restricciones = [
            ('X1','X2', lambda x,y: x!=y),
            ('X1','X3', lambda x,y: x!=y),
            ('X2','X3', lambda x,y: x!=y)
        ]

    global nodos_explorados
    nodos_explorados = 0
    # Ejecutar el solver con el dataset seleccionado
    solucion = backtracking_fc({}, variables, dominios, restricciones)
    print(f"Nodos explorados: {nodos_explorados}")
    if solucion:
        print("Solución encontrada:", solucion)
    else:
        print("No se encontró solución.")

def main():
    # Menú simple para ejecutar en modo demo o interactivo
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
