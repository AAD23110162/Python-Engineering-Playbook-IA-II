"""
001-E2-incertidumbre_probabilidad.py
--------------------------------
Este script introduce los conceptos fundamentales de Incertidumbre y Probabilidad:
- Define eventos aleatorios y espacios muestrales
- Calcula probabilidades básicas de eventos simples y compuestos
- Implementa operaciones con eventos: unión, intersección y complemento
- Muestra ejemplos de eventos mutuamente excluyentes e independientes
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos predefinidos con diferentes tipos de eventos
2. INTERACTIVO: permite al usuario definir espacios muestrales y calcular probabilidades

Autor: Alejandro Aguirre Díaz
"""

def calcular_probabilidad(evento, espacio_muestral):
    """
    Calcula la probabilidad de un evento en un espacio muestral equiprobable.
    :parametro evento: conjunto de resultados favorables
    :parametro espacio_muestral: conjunto de todos los resultados posibles
    :return: probabilidad del evento (float entre 0 y 1)
    """
    if len(espacio_muestral) == 0:
        return 0.0
    return len(evento) / len(espacio_muestral)

def union(evento_a, evento_b):
    """
    Calcula la unión de dos eventos (A ∪ B).
    :parametro evento_a: conjunto que representa el evento A
    :parametro evento_b: conjunto que representa el evento B
    :return: conjunto unión
    """
    return evento_a | evento_b

def interseccion(evento_a, evento_b):
    """
    Calcula la intersección de dos eventos (A ∩ B).
    :parametro evento_a: conjunto que representa el evento A
    :parametro evento_b: conjunto que representa el evento B
    :return: conjunto intersección
    """
    return evento_a & evento_b

def complemento(evento, espacio_muestral):
    """
    Calcula el complemento de un evento (A').
    :parametro evento: conjunto que representa el evento A
    :parametro espacio_muestral: espacio muestral total
    :return: conjunto complemento
    """
    return espacio_muestral - evento

def son_mutuamente_excluyentes(evento_a, evento_b):
    """
    Verifica si dos eventos son mutuamente excluyentes (A ∩ B = ∅).
    :parametro evento_a: conjunto que representa el evento A
    :parametro evento_b: conjunto que representa el evento B
    :return: True si son mutuamente excluyentes, False en caso contrario
    """
    return len(interseccion(evento_a, evento_b)) == 0

def son_independientes(evento_a, evento_b, espacio_muestral):
    """
    Verifica si dos eventos son independientes: P(A ∩ B) = P(A) × P(B).
    :parametro evento_a: conjunto que representa el evento A
    :parametro evento_b: conjunto que representa el evento B
    :parametro espacio_muestral: espacio muestral total
    :return: True si son independientes, False en caso contrario
    """
    p_a = calcular_probabilidad(evento_a, espacio_muestral)
    p_b = calcular_probabilidad(evento_b, espacio_muestral)
    p_interseccion = calcular_probabilidad(interseccion(evento_a, evento_b), espacio_muestral)
    
    # Comparación con tolerancia para errores de punto flotante
    return abs(p_interseccion - (p_a * p_b)) < 1e-9

def probabilidad_union(evento_a, evento_b, espacio_muestral):
    """
    Calcula P(A ∪ B) = P(A) + P(B) - P(A ∩ B).
    :parametro evento_a: conjunto que representa el evento A
    :parametro evento_b: conjunto que representa el evento B
    :parametro espacio_muestral: espacio muestral total
    :return: probabilidad de la unión
    """
    p_a = calcular_probabilidad(evento_a, espacio_muestral)
    p_b = calcular_probabilidad(evento_b, espacio_muestral)
    p_interseccion = calcular_probabilidad(interseccion(evento_a, evento_b), espacio_muestral)
    return p_a + p_b - p_interseccion

def modo_demo():
    """Ejecuta el modo demostrativo con ejemplos predefinidos."""
    print("\n" + "="*70)
    print("MODO DEMO: Incertidumbre y Probabilidad")
    print("="*70)
    
    # Ejemplo 1: Lanzamiento de un dado
    print("\n--- EJEMPLO 1: Lanzamiento de un dado ---")
    espacio_dado = {1, 2, 3, 4, 5, 6}
    evento_par = {2, 4, 6}
    evento_mayor_3 = {4, 5, 6}
    evento_primo = {2, 3, 5}
    
    print(f"Espacio muestral: {sorted(espacio_dado)}")
    print(f"\nEvento A (número par): {sorted(evento_par)}")
    print(f"P(A) = {calcular_probabilidad(evento_par, espacio_dado):.4f}")
    
    print(f"\nEvento B (mayor que 3): {sorted(evento_mayor_3)}")
    print(f"P(B) = {calcular_probabilidad(evento_mayor_3, espacio_dado):.4f}")
    
    print(f"\nEvento C (número primo): {sorted(evento_primo)}")
    print(f"P(C) = {calcular_probabilidad(evento_primo, espacio_dado):.4f}")
    
    # Operaciones con eventos
    print(f"\n--- Operaciones entre A (par) y B (mayor que 3) ---")
    union_ab = union(evento_par, evento_mayor_3)
    interseccion_ab = interseccion(evento_par, evento_mayor_3)
    complemento_a = complemento(evento_par, espacio_dado)
    
    print(f"A ∪ B = {sorted(union_ab)}")
    print(f"P(A ∪ B) = {probabilidad_union(evento_par, evento_mayor_3, espacio_dado):.4f}")
    
    print(f"\nA ∩ B = {sorted(interseccion_ab)}")
    print(f"P(A ∩ B) = {calcular_probabilidad(interseccion_ab, espacio_dado):.4f}")
    
    print(f"\nA' (complemento de A) = {sorted(complemento_a)}")
    print(f"P(A') = {calcular_probabilidad(complemento_a, espacio_dado):.4f}")
    
    # Verificación de propiedades
    print(f"\n--- Propiedades ---")
    print(f"¿A y B son mutuamente excluyentes? {son_mutuamente_excluyentes(evento_par, evento_mayor_3)}")
    print(f"¿A y B son independientes? {son_independientes(evento_par, evento_mayor_3, espacio_dado)}")
    
    # Ejemplo 2: Baraja de cartas simplificada
    print("\n--- EJEMPLO 2: Baraja simplificada (12 cartas) ---")
    # 3 palos (♥, ♦, ♠) con 4 valores cada uno (A, 2, 3, 4)
    palos = ['♥', '♦', '♠']
    valores = ['A', '2', '3', '4']
    espacio_baraja = {f"{v}{p}" for v in valores for p in palos}
    
    evento_corazones = {f"{v}♥" for v in valores}
    evento_ases = {f"A{p}" for p in palos}
    evento_pares = {f"2{p}" for p in palos}
    
    print(f"Espacio muestral (12 cartas): {sorted(espacio_baraja)}")
    print(f"\nEvento D (corazones ♥): {sorted(evento_corazones)}")
    print(f"P(D) = {calcular_probabilidad(evento_corazones, espacio_baraja):.4f}")
    
    print(f"\nEvento E (ases): {sorted(evento_ases)}")
    print(f"P(E) = {calcular_probabilidad(evento_ases, espacio_baraja):.4f}")
    
    print(f"\nEvento F (doses): {sorted(evento_pares)}")
    print(f"P(F) = {calcular_probabilidad(evento_pares, espacio_baraja):.4f}")
    
    # Independencia
    print(f"\n--- Verificación de Independencia ---")
    print(f"D ∩ E = {sorted(interseccion(evento_corazones, evento_ases))}")
    print(f"P(D ∩ E) = {calcular_probabilidad(interseccion(evento_corazones, evento_ases), espacio_baraja):.4f}")
    print(f"P(D) × P(E) = {calcular_probabilidad(evento_corazones, espacio_baraja) * calcular_probabilidad(evento_ases, espacio_baraja):.4f}")
    print(f"¿D y E son independientes? {son_independientes(evento_corazones, evento_ases, espacio_baraja)}")
    
    # Mutuamente excluyentes
    print(f"\n¿E (ases) y F (doses) son mutuamente excluyentes? {son_mutuamente_excluyentes(evento_ases, evento_pares)}")
    print(f"(Tienen intersección vacía: {sorted(interseccion(evento_ases, evento_pares))})")

def modo_interactivo():
    """Ejecuta el modo interactivo con entrada del usuario."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Incertidumbre y Probabilidad")
    print("="*70)
    
    print("\nElige un escenario predefinido:")
    print("1. Lanzamiento de moneda (2 resultados)")
    print("2. Lanzamiento de dado (6 resultados)")
    print("3. Extracción de carta de baraja española (40 cartas)")
    
    opcion = input("\nIngresa el número de opción (1-3): ").strip()
    
    if opcion == '1':
        espacio = {'Cara', 'Cruz'}
        print(f"\nEspacio muestral: {espacio}")
        evento_input = input("Define el evento (ej: Cara): ").strip()
        evento = {evento_input} if evento_input in espacio else set()
        
    elif opcion == '2':
        espacio = {1, 2, 3, 4, 5, 6}
        print(f"\nEspacio muestral: {sorted(espacio)}")
        print("Define el evento ingresando números separados por comas (ej: 1,3,5)")
        evento_input = input("Evento: ").strip()
        try:
            evento = {int(x.strip()) for x in evento_input.split(',') if x.strip().isdigit()}
            evento = evento & espacio  # Solo elementos válidos
        except:
            evento = set()
            
    elif opcion == '3':
        palos = ['oros', 'copas', 'espadas', 'bastos']
        valores = list(range(1, 13))  # 1-12 (sin 8 y 9 en española, pero simplificamos)
        espacio = {f"{v}-{p}" for v in valores for p in palos}
        print(f"\nEspacio muestral: 40 cartas (valores 1-12, 4 palos)")
        print("Ejemplos de eventos:")
        print("  - Palo específico: ingresa 'oros', 'copas', 'espadas' o 'bastos'")
        print("  - Valor específico: ingresa un número del 1 al 12")
        
        evento_input = input("\nDefine el evento: ").strip().lower()
        if evento_input in palos:
            evento = {f"{v}-{evento_input}" for v in valores}
        elif evento_input.isdigit() and int(evento_input) in valores:
            evento = {f"{evento_input}-{p}" for p in palos}
        else:
            evento = set()
    else:
        print("Opción no válida.")
        return
    
    if not evento:
        print("Evento vacío o no válido.")
        return
    
    print(f"\nEvento definido: {sorted(evento) if len(evento) <= 10 else f'{len(evento)} elementos'}")
    prob = calcular_probabilidad(evento, espacio)
    print(f"P(Evento) = {prob:.4f} = {prob*100:.2f}%")
    
    # Complemento
    comp = complemento(evento, espacio)
    print(f"\nComplemento del evento: {len(comp)} elementos")
    print(f"P(Complemento) = {calcular_probabilidad(comp, espacio):.4f}")

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("INCERTIDUMBRE Y PROBABILIDAD")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (ejemplos predefinidos)")
    print("2. INTERACTIVO (define tus propios eventos)")
    
    opcion = input("\nIngresa el número de opción (1 o 2): ").strip()
    
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida. Ejecutando modo DEMO por defecto...")
        modo_demo()
    
    print("\n" + "="*70)
    print("FIN DEL PROGRAMA")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
