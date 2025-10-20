"""
020-E1-propagacion_de_restricciones.py
--------------------------------------
Este script implementa el mecanismo de **Propagación de Restricciones** (Constraint Propagation) dentro de un CSP (Problema de Satisfacción de Restricciones):
- Al asignar una variable, se propagan los efectos a través del grafo de restricciones para reducir los dominios de muchas variables (no sólo vecinas directas).
- Usa una estrategia simple de consistencia de arcos (Arc Consistency) cada vez que se hace una asignación.
- Incluye dos modos de ejecución:
    1. MODO DEMO: problema pequeño predefinido.
    2. MODO INTERACTIVO: el usuario elige un dataset sencillo predefinido (no ingresa todo manualmente).
- Muestra paso a paso la reducción de dominios y los retrocesos.

Autor: Alejandro Aguirre Díaz
"""

# Contador global para nodos explorados (mide el esfuerzo de búsqueda)
# Se incrementa cada vez que se explora un nodo (asignación parcial)
nodos_explorados = 0

def asignacion_completa(asignacion, variables):
    # Verificar si todas las variables han sido asignadas
    # Devuelve True solo si cada variable en 'variables' tiene un valor en 'asignacion'
    return all(var in asignacion for var in variables)

def revisa_arco(dominios, v_i, v_j, restr):
    """
    Aplica el test de consistencia de arco (v_i → v_j): elimina valores de dominio de v_i para los cuales
    no existe valor en dominio de v_j que satisfaga la restricción.
    :parametro dominios: dict variable → lista de valores posibles
    :parametro v_i: variable en el arco
    :parametro v_j: variable en el arco
    :parametro restr: función (valor_i, valor_j) → bool que indica si la asignación es válida
    :return: True si dominio de v_i cambió, False si no
    """
    cambiado = False
    nuevo_dom_i = []
    
    # Revisar cada valor en el dominio de v_i
    for valor_i in dominios[v_i]:
        # Verificar si existe algún valor en el dominio de v_j que satisface la restricción
        # con valor_i (es decir, si valor_i tiene soporte en v_j)
        existe = any(restr(valor_i, valor_j) for valor_j in dominios[v_j])
        
        if existe:
            # Si existe soporte, conservar valor_i en el dominio
            nuevo_dom_i.append(valor_i)
        else:
            # Si no existe soporte, eliminar valor_i (marcar cambio)
            cambiado = True
    
    # Si el dominio cambió, actualizar con el nuevo dominio filtrado
    if cambiado:
        dominios[v_i] = nuevo_dom_i
    
    return cambiado

def arc_consistency(dominios, restricciones, variables):
    """
    Algoritmo AC-3 para propagación de restricciones:
    - Inicia una cola con todos los arcos (v_i, v_j) para cada restricción binaria.
    - Mientras la cola no esté vacía:
        * extrae un arco (vi, vj)
        * aplica revisa_arco
        * si el dominio de vi cambió: por cada vecino vk de vi distinto de vj, añade (vk, vi) a la cola
    - Si en algún momento algún dominio queda vacío: falla (no solución en esta rama).
    :parametro dominios: dict variable → lista de valores posibles (actualizados)
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :parametro variables: lista de variables
    :return: True si alcanzó consistencia sin dominios vacíos, False si encontró dominio vacío
    """
    from collections import deque
    cola = deque()
    # Inicializar todos los arcos (ambas direcciones para cada restricción)
    for (v1, v2, restr) in restricciones:
        # Arco directo: (v1, v2) con la restricción original
        cola.append((v1, v2, restr))
        # Arco reverso: (v2, v1) con la restricción invertida
        # Crear una función nueva que invierte los argumentos
        def crear_restr_inversa(r):
            return lambda y, x: r(x, y)
        cola.append((v2, v1, crear_restr_inversa(restr)))
    
    while cola:
        v_i, v_j, restr = cola.popleft()
        # Revisar si el arco (v_i, v_j) requiere reducción del dominio de v_i
        if revisa_arco(dominios, v_i, v_j, restr):
            # Si el dominio de v_i quedó vacío, no hay solución
            if not dominios[v_i]:
                return False
            # Añadir arcos (vk, v_i) para todos los vecinos vk de v_i excepto v_j
            for (vk, vl, restr2) in restricciones:
                if vl == v_i and vk != v_j:
                    cola.append((vk, v_i, restr2))
    return True

def backtracking_propagacion(asignacion, variables, dominios, restricciones):
    """
    Búsqueda de vuelta atrás con propagación de restricciones:
    :parametro asignacion: dict variable→valor asignado hasta ahora
    :parametro variables: lista de variables
    :parametro dominios: dict variable→lista valores posibles (actualizados)
    :parametro restricciones: lista de tuplas (v1, v2, función_restricción)
    :return: asignación completa o None si no hay solución
    """
    global nodos_explorados
    # Incrementar el contador cada vez que exploramos un nodo (asignación parcial)
    nodos_explorados += 1
    
    # Mostrar el estado actual: qué variables están asignadas y qué dominios quedan
    print(f"Nodo {nodos_explorados}: Asignación parcial = {asignacion}, dominios = {dominios}")
    
    # Caso base: verificar si todas las variables ya están asignadas
    if asignacion_completa(asignacion, variables):
        print(f"✔ Solución completa encontrada: {asignacion}")
        return asignacion

    # Seleccionar variable sin asignar (estrategia simple: primera sin asignar)
    # Se podría mejorar con heurísticas como MRV (Minimum Remaining Values)
    var = next(v for v in variables if v not in asignacion)
    print(f"Seleccionando variable '{var}' para asignar…")

    # Probar cada valor posible en el dominio actual de la variable seleccionada
    for valor in list(dominios[var]):
        print(f"  Intentando {var} = {valor}")
        
        # Crear una copia de seguridad de los dominios antes de modificarlos
        # Esto permite restaurar el estado si la propagación falla
        dominios_backup = {v: list(dominios[v]) for v in dominios}
        
        # Asignar tentativamente el valor a la variable
        asignacion[var] = valor
        
        # Fijar el dominio de la variable asignada a un único valor
        dominios[var] = [valor]

        # Propagar restricciones: aplicar AC-3 para reducir dominios de otras variables
        # Esto detecta inconsistencias de forma temprana
        if arc_consistency(dominios, restricciones, variables):
            print(f"    Propagación completada tras {var} = {valor}")
            
            # Llamada recursiva para asignar la siguiente variable
            resultado = backtracking_propagacion(asignacion, variables, dominios, restricciones)
            
            # Si se encontró una solución completa, propagarla hacia arriba
            if resultado is not None:
                return resultado
        else:
            # La propagación falló: algún dominio quedó vacío
            print(f"    ✘ Propagación falló tras {var} = {valor} → dominio vacío detectado")

        # Restaurar asignación y dominios (backtrack)
        del asignacion[var]
        dominios.clear()
        dominios.update({v: list(dominios_backup[v]) for v in dominios_backup})
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
    restricciones = [
        ('X1','X2', lambda x,y: x != y),  # X1 debe ser diferente de X2
        ('X1','X3', lambda x,y: x != y),  # X1 debe ser diferente de X3
        ('X2','X3', lambda x,y: x != y)   # X2 debe ser diferente de X3
    ]
    
    print("Problema demo: 3 variables, dominio {1,2,3}, restricción all-different.\n")
    
    global nodos_explorados
    # Reiniciar el contador de nodos explorados para esta ejecución
    nodos_explorados = 0
    
    # Ejecutar backtracking con propagación de restricciones
    # Crear copias de dominios para no modificar el original
    solucion = backtracking_propagacion({}, variables, {v:list(dominios[v]) for v in dominios}, restricciones)
    
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
    opcion = input("Introduce el número de dataset: ").strip()

    # Configurar el problema según la opción seleccionada
    if opcion == '1':
        # Problema 1: All-Different con 3 variables y dominio {1,2,3}
        variables = ['X1','X2','X3']
        dominios = {v:[1,2,3] for v in variables}
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
        dominios = {v:list(colores) for v in variables}
        # Restricciones: regiones adyacentes deben tener colores diferentes
        # Las regiones están conectadas en ciclo: R1-R2-R3-R4-R1
        restricciones = [
            ('R1','R2', lambda x,y: x != y),  # R1 y R2 son adyacentes
            ('R2','R3', lambda x,y: x != y),  # R2 y R3 son adyacentes
            ('R3','R4', lambda x,y: x != y),  # R3 y R4 son adyacentes
            ('R4','R1', lambda x,y: x != y)   # R4 y R1 son adyacentes (cierre del ciclo)
        ]
        print("Problema: 4 regiones en ciclo, colores distintos para vecinos.\n")
    else:
        # Opción inválida: usar dataset predeterminado (dataset 1)
        print("Opción inválida. Se usará dataset 1.")
        variables = ['X1','X2','X3']
        dominios = {v:[1,2,3] for v in variables}
        restricciones = [
            ('X1','X2', lambda x,y: x != y),
            ('X1','X3', lambda x,y: x != y),
            ('X2','X3', lambda x,y: x != y)
        ]

    global nodos_explorados
    # Reiniciar el contador de nodos explorados para esta ejecución
    nodos_explorados = 0
    
    # Ejecutar backtracking con propagación de restricciones
    # Crear copias de dominios para no modificar el original
    solucion = backtracking_propagacion({}, variables, {v:list(dominios[v]) for v in dominios}, restricciones)
    
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
