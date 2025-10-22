"""
019-E1-comprobacion_hacia_adelante.py
--------------------------------------
Este script implementa la técnica de Comprobación hacia adelante (Forward Checking) para un CSP,
y muestra en la terminal **cada paso** del algoritmo: asignaciones parciales, dominios actualizados, decisiones de retroceso.

Autor: Alejandro Aguirre Díaz
"""

# Contador global para nodos explorados (mide el esfuerzo de búsqueda)
# Se incrementa cada vez que se explora un nodo (asignación parcial)
nodos_explorados = 0

def asignacion_completa(asignacion, variables):
    # Verificar si todas las variables han sido asignadas
    # Devuelve True solo si cada variable en 'variables' tiene un valor en 'asignacion'
    return all(var in asignacion for var in variables)

def consistente(asignacion, variable, valor, restricciones):
    """
    Comprueba que asignar 'valor' a 'variable' no viole ninguna restricción con variables ya asignadas.
    """
    # Recorrer todas las restricciones binarias del problema
    for (v1, v2, restr) in restricciones:
        # Caso 1: la restricción es (variable, v2) y v2 ya está asignada
        if variable == v1 and v2 in asignacion:
            # Verificar que la restricción restr(valor, asignacion[v2]) se cumple
            if not restr(valor, asignacion[v2]):
                return False
        # Caso 2: la restricción es (v1, variable) y v1 ya está asignada
        if variable == v2 and v1 in asignacion:
            # Verificar que la restricción restr(asignacion[v1], valor) se cumple
            if not restr(asignacion[v1], valor):
                return False
    # Si todas las restricciones se cumplen, la asignación es consistente
    return True

def forward_checking(dominios, variable, valor, restricciones):
    """
    Aplica filtrado (forward checking): tras asignar variable=valor,
    reduce los dominios de las variables no asignadas que tienen restricción con la variable.
    Devuelve nuevos dominios o None si algún dominio queda vacío.
    """
    # Crear una copia de los dominios actuales para no modificar el original
    nuevos_dom = {v: list(dominios[v]) for v in dominios}
    # Fijar el dominio de la variable recién asignada a un único valor
    nuevos_dom[variable] = [valor]
    
    # Informar sobre el proceso de filtrado
    print(f"  → Filtrando dominios tras asignar {variable} = {valor}")
    
    # Recorrer todas las restricciones para actualizar dominios
    for (v1, v2, restr) in restricciones:
        # Caso 1: la restricción es (variable, v2) - filtrar dominio de v2
        if v1 == variable and v2 in nuevos_dom:
            # Guardar el dominio anterior para mostrar el cambio
            antes = list(nuevos_dom[v2])
            # Mantener solo los valores de v2 que son compatibles con 'valor'
            nuevos_dom[v2] = [y for y in nuevos_dom[v2] if restr(valor, y)]
            # Mostrar el filtrado realizado
            print(f"     Dominio {v2} antes: {antes}, tras filtrado: {nuevos_dom[v2]}")
            # Si el dominio quedó vacío, no hay solución posible con esta asignación
            if not nuevos_dom[v2]:
                print(f"     ✘ Dominio de {v2} quedó vacío → fallo inmediato")
                return None
        
        # Caso 2: la restricción es (v1, variable) - filtrar dominio de v1
        if v2 == variable and v1 in nuevos_dom:
            # Guardar el dominio anterior para mostrar el cambio
            antes = list(nuevos_dom[v1])
            # Mantener solo los valores de v1 que son compatibles con 'valor'
            nuevos_dom[v1] = [x for x in nuevos_dom[v1] if restr(x, valor)]
            # Mostrar el filtrado realizado
            print(f"     Dominio {v1} antes: {antes}, tras filtrado: {nuevos_dom[v1]}")
            # Si el dominio quedó vacío, no hay solución posible con esta asignación
            if not nuevos_dom[v1]:
                print(f"     ✘ Dominio de {v1} quedó vacío → fallo inmediato")
                return None
    
    # Retornar los nuevos dominios filtrados si todos son válidos
    return nuevos_dom

def backtracking_fc(asignacion, variables, dominios, restricciones, nivel=0):
    global nodos_explorados
    # Incrementar el contador cada vez que exploramos un nodo (asignación parcial)
    nodos_explorados += 1

    # Mostrar el estado actual: qué variables están asignadas y qué dominios quedan
    print(f"Nodo {nodos_explorados}: Asignación parcial = {asignacion}, dominios = {dominios}")
    
    # Caso base: verificar si todas las variables ya están asignadas
    if asignacion_completa(asignacion, variables):
        print(f"✔ Solución completa hallada: {asignacion}")
        return asignacion

    # Selección de variable sin asignar (estrategia simple: primera sin asignar)
    # Se podría mejorar con heurísticas como MRV (Minimum Remaining Values)
    var = next(v for v in variables if v not in asignacion)
    print(f"Seleccionando variable '{var}' para asignar…")

    # Probar cada valor posible en el dominio actual de la variable seleccionada
    for valor in dominios[var]:
        print(f"  Intentando {var} = {valor}")
        
        # Verificar que el valor sea consistente con las asignaciones actuales
        if consistente(asignacion, var, valor, restricciones):
            print(f"    {var} = {valor} es consistente con la asignación actual")
            
            # Asignar tentativamente el valor a la variable
            asignacion[var] = valor
            
            # Aplicar forward checking: reducir dominios de variables relacionadas
            nuevos_dom = forward_checking(dominios, var, valor, restricciones)
            
            # Si el forward checking no vació ningún dominio, continuar con la búsqueda
            if nuevos_dom is not None:
                # Llamada recursiva para asignar la siguiente variable
                resultado = backtracking_fc(asignacion, variables, nuevos_dom, restricciones, nivel+1)
                # Si se encontró una solución completa, propagarla hacia arriba
                if resultado is not None:
                    return resultado
            else:
                # El forward checking falló: algún dominio quedó vacío
                print(f"    Retrocediendo: filtrado de dominios falló para {var} = {valor}")
        else:
            # El valor no es consistente con las asignaciones actuales
            print(f"    {var} = {valor} no es consistente → omitiendo")
        
        # Revertir la asignación (backtrack) para probar el siguiente valor
        if var in asignacion:
            del asignacion[var]
        print(f"  Retroceso tras intentar {var} = {valor}")
    
    # Si ningún valor funcionó para esta variable, retornar None (fallo)
    return None

def modo_demo():
    """Modo demostrativo con problema predefinido."""
    print("\n--- MODO DEMO ---")
    
    # Definir variables del CSP: 3 variables que deben tomar valores diferentes
    variables = ['X1','X2','X3']
    
    # Definir dominios: cada variable puede tomar valores {1, 2, 3}
    dominios = {
        'X1': [1,2,3],
        'X2': [1,2,3],
        'X3': [1,2,3]
    }
    
    # Definir restricciones: todas las variables deben ser diferentes (all-different)
    # Las restricciones son binarias: comparan pares de variables
    restricciones = [
        ('X1','X2', lambda x,y: x != y),  # X1 debe ser diferente de X2
        ('X1','X3', lambda x,y: x != y),  # X1 debe ser diferente de X3
        ('X2','X3', lambda x,y: x != y)   # X2 debe ser diferente de X3
    ]
    
    print("Problema: 3 variables, dominio {1,2,3}, restricción all-different.\n")
    
    global nodos_explorados
    # Reiniciar el contador de nodos explorados para esta ejecución
    nodos_explorados = 0
    
    # Ejecutar backtracking con forward checking
    # Comenzamos con asignación vacía {}
    solucion = backtracking_fc({}, variables, dominios, restricciones)
    
    # Mostrar estadísticas finales
    print(f"\nTotal nodos explorados: {nodos_explorados}")
    if solucion:
        print("Solución final:", solucion)
    else:
        print("No se encontró solución.")

def modo_interactivo():
    """Modo interactivo donde el usuario selecciona un dataset."""
    print("\n--- MODO INTERACTIVO ---")
    
    # Menú de selección de problemas predefinidos
    print("Seleccione un dataset:")
    print("1) All-Different 3 variables")
    print("2) Colores de mapa: 4 regiones, 3 colores")
    opcion = input("Introduce el número del dataset: ").strip()

    # Configurar el problema según la opción seleccionada
    if opcion == '1':
        # Problema 1: All-Different con 3 variables y dominio {1,2,3}
        variables = ['X1','X2','X3']
        dominios = {'X1': [1,2,3], 'X2': [1,2,3], 'X3': [1,2,3]}
        restricciones = [
            ('X1','X2', lambda x,y: x != y),
            ('X1','X3', lambda x,y: x != y),
            ('X2','X3', lambda x,y: x != y)
        ]
    elif opcion == '2':
        # Problema 2: Coloreado de mapa con 4 regiones conectadas en ciclo
        variables = ['R1','R2','R3','R4']
        colores = ['Rojo','Verde','Azul']
        # Cada región puede tomar cualquiera de los 3 colores
        dominios = {v: colores.copy() for v in variables}
        # Restricciones: regiones adyacentes deben tener colores diferentes
        # Las regiones están conectadas en ciclo: R1-R2-R3-R4-R1
        restricciones = [
            ('R1','R2', lambda x,y: x != y),  # R1 y R2 son adyacentes
            ('R2','R3', lambda x,y: x != y),  # R2 y R3 son adyacentes
            ('R3','R4', lambda x,y: x != y),  # R3 y R4 son adyacentes
            ('R4','R1', lambda x,y: x != y)   # R4 y R1 son adyacentes (cierre del ciclo)
        ]
        print("Problema: 4 regiones conectadas en ciclo, colores distintos para vecinos.\n")
    else:
        # Opción inválida: usar dataset predeterminado (dataset 1)
        print("Opción no válida: se usará dataset 1.")
        variables = ['X1','X2','X3']
        dominios = {'X1': [1,2,3], 'X2': [1,2,3], 'X3': [1,2,3]}
        restricciones = [
            ('X1','X2', lambda x,y: x != y),
            ('X1','X3', lambda x,y: x != y),
            ('X2','X3', lambda x,y: x != y)
        ]

    global nodos_explorados
    # Reiniciar el contador de nodos explorados para esta ejecución
    nodos_explorados = 0
    
    # Ejecutar backtracking con forward checking
    solucion = backtracking_fc({}, variables, dominios, restricciones)
    
    # Mostrar estadísticas finales
    print(f"\nTotal nodos explorados: {nodos_explorados}")
    if solucion:
        print("Solución final:", solucion)
    else:
        print("No se encontró solución.")

def main():
    """Función principal que ejecuta el programa."""
    # Mostrar menú principal para que el usuario seleccione el modo
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    
    # Leer la opción del usuario
    opcion = input("Ingrese el número de opción: ").strip()
    
    # Ejecutar el modo correspondiente según la selección
    if opcion == '1':
        # Ejecutar modo demo con problema predefinido
        modo_demo()
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario selecciona un dataset
        modo_interactivo()
    else:
        # Manejar opción inválida: usar modo demo por defecto
        print("Opción no válida. Se usará MODO DEMO.")
        modo_demo()

# Punto de entrada del programa
if __name__ == "__main__":
    main()
