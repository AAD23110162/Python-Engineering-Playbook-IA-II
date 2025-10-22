"""
018-E1-busqueda_vuelta_atras.py
--------------------------------
Este script implementa el esquema de Búsqueda de Vuelta Atrás (Backtracking):
- A partir de una **cadena de datos** que el usuario proporciona (por ejemplo: "A,B,C,D"),
  el tamaño del problema se ajusta automáticamente (longitud = número de elementos).
- El dominio será los mismos datos que el usuario ingresó (o una variante según problema).
- Se busca **todas** las combinaciones/permutaciones según la configuración.
- Se muestra cuántos nodos (soluciones parciales) han sido explorados.
- Incluye dos modos de ejecución:
    1. MODO DEMO: datos fijos predefinidos.
    2. MODO INTERACTIVO: el usuario introduce una cadena de valores separados por comas, y el script ajusta el tamaño automáticamente.

Autor: Alejandro Aguirre Díaz
"""

# Contador global para nodos explorados (mide el esfuerzo de búsqueda)
# Se incrementa cada vez que se explora una solución parcial
nodos_explorados = 0

def es_solucion(solucion_parcial, longitud_objetivo):
    """
    Verifica si la solución parcial ha alcanzado la longitud deseada.
    :parametro solucion_parcial: lista con valores asignados hasta ahora
    :parametro longitud_objetivo: número total de elementos a asignar
    :return: True si se ha completado
    """
    # Comparar la longitud actual de la solución con el objetivo
    # Si son iguales, hemos construido una solución completa
    return len(solucion_parcial) == longitud_objetivo

def es_factible(solucion_parcial):
    """
    Comprueba si la asignación parcial es factible según el problema concreto.
    Aquí, como ejemplo simple, no hay restricción adicional (si quisieras, puedes modificar).
    :parametro solucion_parcial: lista de valores asignados hasta ahora
    :return: True si es aceptable continuar
    """
    # Verificar que no haya elementos repetidos (restricción para permutaciones)
    # Comparamos la longitud de la lista con la longitud del conjunto (sin duplicados)
    # Si son iguales, significa que todos los elementos son únicos
    return len(solucion_parcial) == len(set(solucion_parcial))

def vuelta_atras(solucion_parcial, dominio, longitud_objetivo, todas=True, soluciones=None):
    """
    Función recursiva para buscar soluciones usando backtracking.
    :parametro solucion_parcial: lista con los valores asignados actualmente
    :parametro dominio: lista de todos los valores posibles
    :parametro longitud_objetivo: longitud que debe alcanzar cada solución
    :parametro todas: booleano. Si True, recolecta todas las soluciones; si False, detiene tras la primera.
    :parametro soluciones: lista donde se agregan las soluciones cuando todas=True
    :return: si todas=False devuelve la primera solución encontrada o None.
    """
    global nodos_explorados
    # Incrementar el contador cada vez que exploramos un nodo (solución parcial)
    nodos_explorados += 1

    # Caso base: verificar si hemos alcanzado una solución completa
    if es_solucion(solucion_parcial, longitud_objetivo):
        if todas:
            # Si queremos todas las soluciones, guardar una copia y continuar buscando
            soluciones.append(solucion_parcial.copy())
            return None
        else:
            # Si solo queremos una solución, retornarla inmediatamente
            return solucion_parcial.copy()

    # Probar cada valor del dominio para el siguiente nivel
    for valor in dominio:
        # Verificar si el valor ya está en la solución parcial (evitar repeticiones)
        if valor in solucion_parcial:
            # Saltar este valor para mantener permutaciones sin repetición
            continue
        
        # Agregar el valor candidato a la solución parcial
        solucion_parcial.append(valor)
        
        # Verificar si la solución parcial sigue siendo factible
        if es_factible(solucion_parcial):
            # Llamada recursiva para explorar el siguiente nivel
            resultado = vuelta_atras(solucion_parcial, dominio, longitud_objetivo, todas, soluciones)
            # Si solo buscamos una solución y la encontramos, retornarla
            if (not todas) and (resultado is not None):
                return resultado
        
        # Retroceder (backtrack): quitar el último valor para probar el siguiente
        solucion_parcial.pop()
    
    # Si no se encontró solución en este subárbol, retornar según el modo
    return None if (not todas) else soluciones

def modo_demo():
    """Modo demostrativo con valores fijos."""
    # Encabezado del modo demo
    print("\n--- MODO DEMO ---")
    
    # Definir dominio predefinido: 3 elementos ['A', 'B', 'C']
    # Esto generará todas las permutaciones de estos 3 elementos (3! = 6 permutaciones)
    datos = ['A', 'B', 'C']
    longitud = len(datos)
    print(f"Dominio = {datos}")
    
    global nodos_explorados
    # Reiniciar el contador de nodos para esta ejecución
    nodos_explorados = 0
    
    # Lista para almacenar todas las soluciones encontradas
    soluciones = []
    
    # Ejecutar backtracking con solución inicial vacía
    vuelta_atras([], datos, longitud, todas=True, soluciones=soluciones)
    
    # Mostrar estadísticas de la búsqueda
    print(f"Nodos explorados: {nodos_explorados}")
    print(f"Se han encontrado {len(soluciones)} soluciones:")
    
    # Imprimir cada solución (permutación) encontrada
    for sol in soluciones:
        print(sol)

def modo_interactivo():
    """Modo interactivo donde el usuario define los datos separados por comas."""
    # Encabezado del modo interactivo
    print("\n--- MODO INTERACTIVO ---")
    
    # Solicitar al usuario que ingrese los valores del dominio separados por comas
    entrada = input("Introduce los valores separados por comas (ej: A,B,C,D): ").strip()
    
    # Validar la entrada del usuario
    if not entrada:
        # Si la entrada está vacía, usar dominio predeterminado
        print("Entrada vacía. Se usará el dominio predeterminado ['A','B','C'].")
        datos = ['A', 'B', 'C']
    else:
        # Procesar la entrada: dividir por comas y eliminar espacios en blanco
        datos = [x.strip() for x in entrada.split(",") if x.strip()]
        
        # Verificar que se hayan detectado valores válidos
        if len(datos) < 1:
            print("No se detectaron valores válidos. Se usará ['A','B','C'].")
            datos = ['A', 'B', 'C']

    # La longitud objetivo será el número de elementos ingresados
    # Esto significa que buscaremos permutaciones de todos los elementos
    longitud = len(datos)
    print(f"Dominio = {datos}  (longitud objetivo = {longitud})\n")
    
    global nodos_explorados
    # Reiniciar el contador de nodos para esta ejecución
    nodos_explorados = 0
    
    # Lista para almacenar todas las soluciones encontradas
    soluciones = []
    
    # Ejecutar backtracking con solución inicial vacía
    vuelta_atras([], datos, longitud, todas=True, soluciones=soluciones)
    
    # Mostrar estadísticas de la búsqueda
    print(f"Nodos explorados: {nodos_explorados}")
    print(f"Se han encontrado {len(soluciones)} soluciones:")
    
    # Imprimir cada solución (permutación) encontrada
    for sol in soluciones:
        print(sol)

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
        # Ejecutar modo demo con valores predefinidos
        modo_demo()
    elif opcion == '2':
        # Ejecutar modo interactivo donde el usuario define los datos
        modo_interactivo()
    else:
        # Manejar opción inválida: usar modo demo por defecto
        print("Opción no válida. Se usará MODO DEMO.")
        modo_demo()

# Punto de entrada del programa
if __name__ == "__main__":
    main()
