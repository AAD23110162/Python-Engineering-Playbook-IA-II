"""
032-E1-teoria_juegos.py
--------------------------------
Este script implementa un modelo educativo de Teoría de Juegos:
- Permite definir o usar un ejemplo predefinido de juego de dos jugadores con matriz de pagos.
- Calcula para cada jugador la mejor respuesta dada una estrategia del otro.
- Identifica los perfiles de estrategias que constituyen un Equilibrio de Nash (o varios) en estrategias puras.
- Modo DEMO: ejemplo clásico predefinido.
- Modo INTERACTIVO: usuario puede configurar pagos o elegir juego predefinido.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def mejores_respuestas(matriz_pagos, jugador, otras_estrategias):
    """
    Dada la matriz de pagos, devuelve para un jugador las mejores respuestas
    frente a cada combinación de la(s) otra(s) estrategia(s).
    :parametro matriz_pagos: dict (estr1, estr2) -> (pago_jugador1, pago_jugador2)
    :parametro jugador: 1 o 2
    :parametro otras_estrategias: lista de posibles estrategias del otro jugador
    :return: dict mapa de estr_otros -> lista de mejores respuestas para jugador
    """
    # Diccionario para almacenar las mejores respuestas: estrategia_oponente -> [mejores_estrategias]
    respuestas = {}
    
    # Extraer todas las estrategias únicas de cada jugador de las claves de la matriz
    estrategias_j1 = sorted({e1 for (e1, _) in matriz_pagos.keys()})
    estrategias_j2 = sorted({e2 for (_, e2) in matriz_pagos.keys()})
    
    # Calcular mejores respuestas según el jugador especificado
    if jugador == 1:
        # Para el Jugador 1: encontrar la mejor estrategia frente a cada estrategia del Jugador 2
        for e2 in otras_estrategias:
            max_pago = float('-inf')  # Inicializar con el peor pago posible
            mejores = []  # Lista de estrategias que dan el mejor pago
            
            # Revisar todas las estrategias posibles del Jugador 1
            for e1 in estrategias_j1:
                # Obtener el pago del Jugador 1 para esta combinación (índice 0)
                pago = matriz_pagos[(e1, e2)][0]
                
                # Si encontramos un pago mejor, reiniciar la lista de mejores estrategias
                if pago > max_pago:
                    max_pago = pago
                    mejores = [e1]
                # Si el pago es igual al mejor, agregar a la lista (pueden haber empates)
                elif pago == max_pago:
                    mejores.append(e1)
            
            # Guardar las mejores respuestas para esta estrategia del Jugador 2
            respuestas[e2] = mejores
    else:
        # Para el Jugador 2: encontrar la mejor estrategia frente a cada estrategia del Jugador 1
        for e1 in otras_estrategias:
            max_pago = float('-inf')  # Inicializar con el peor pago posible
            mejores = []  # Lista de estrategias que dan el mejor pago
            
            # Revisar todas las estrategias posibles del Jugador 2
            for e2 in estrategias_j2:
                # Obtener el pago del Jugador 2 para esta combinación (índice 1)
                pago = matriz_pagos[(e1, e2)][1]
                
                # Si encontramos un pago mejor, reiniciar la lista de mejores estrategias
                if pago > max_pago:
                    max_pago = pago
                    mejores = [e2]
                # Si el pago es igual al mejor, agregar a la lista (pueden haber empates)
                elif pago == max_pago:
                    mejores.append(e2)
            
            # Guardar las mejores respuestas para esta estrategia del Jugador 1
            respuestas[e1] = mejores
    
    return respuestas

def encontrar_equilibrios_nash(matriz_pagos):
    """
    Busca todos los perfiles (e1,e2) que son Equilibrio de Nash en estrategias puras:
    un perfil es equilibrio si e1 está en las mejores respuestas del jugador1 frente a e2
    Y e2 está en las mejores respuestas del jugador2 frente a e1.
    :parametro matriz_pagos: dict (e1,e2) -> (p1,p2)
    :return: list de perfiles (e1,e2)
    """
    # Extraer todas las estrategias únicas de cada jugador
    estrategias_j1 = sorted({e1 for (e1, _) in matriz_pagos.keys()})
    estrategias_j2 = sorted({e2 for (_, e2) in matriz_pagos.keys()})
    
    # Calcular las mejores respuestas de cada jugador
    # mr1: para cada estrategia del J2, cuáles son las mejores estrategias del J1
    mr1 = mejores_respuestas(matriz_pagos, 1, estrategias_j2)
    # mr2: para cada estrategia del J1, cuáles son las mejores estrategias del J2
    mr2 = mejores_respuestas(matriz_pagos, 2, estrategias_j1)
    
    # Lista para almacenar los equilibrios de Nash encontrados
    equilibrios = []
    
    # Revisar todas las combinaciones posibles de estrategias
    for e1 in estrategias_j1:
        for e2 in estrategias_j2:
            # Un perfil (e1,e2) es Equilibrio de Nash si:
            # - e1 es mejor respuesta del J1 cuando J2 juega e2 Y
            # - e2 es mejor respuesta del J2 cuando J1 juega e1
            # Esto significa que ningún jugador puede mejorar cambiando unilateralmente
            if e1 in mr1[e2] and e2 in mr2[e1]:
                equilibrios.append((e1, e2))
    
    return equilibrios

def modo_demo():
    print("\n--- MODO DEMO: Juego clásico «Dilema del Prisionero» simplificado ---")
    
    # Definir las estrategias disponibles: «Cooperar» (C), «Traicionar» (T)
    estrategias = ['C','T']
    
    # Definir la matriz de pagos: (estrategia_J1, estrategia_J2) -> (pago_J1, pago_J2)
    # Si ambos cooperan (C,C): ambos reciben 3
    # Si uno traiciona y otro coopera: el traidor recibe 5, el cooperador 0
    # Si ambos traicionan (T,T): ambos reciben 1 (peor que cooperar mutuamente)
    matriz_pagos = {
        ('C','C'): (3,3),  # Cooperación mutua
        ('C','T'): (0,5),  # J1 coopera, J2 traiciona
        ('T','C'): (5,0),  # J1 traiciona, J2 coopera
        ('T','T'): (1,1),  # Traición mutua
    }
    
    # Mostrar la información del juego
    print("Estrategias disponibles para cada jugador:", estrategias)
    print("Matriz de pagos (jugador1, jugador2):")
    for (e1,e2), (p1,p2) in matriz_pagos.items():
        print(f"  (Jugador1: {e1}, Jugador2: {e2}) → pagos = ({p1}, {p2})")
    
    # Calcular los equilibrios de Nash del juego
    equilibrios = encontrar_equilibrios_nash(matriz_pagos)
    
    # Mostrar las mejores respuestas de cada jugador
    print("\nMejores respuestas jugador1 frente a jugador2:")
    print(mejores_respuestas(matriz_pagos, 1, estrategias))
    print("Mejores respuestas jugador2 frente a jugador1:")
    print(mejores_respuestas(matriz_pagos, 2, estrategias))
    
    # Mostrar los equilibrios encontrados
    print("\nEquilibrio(s) de Nash encontrado(s):", equilibrios)
    if equilibrios:
        for eq in equilibrios:
            print(f"  → Perfil de estrategia Equilibrio de Nash: {eq}")
    else:
        print("  No se encontró equilibrio en estrategias puras.")
    
    # Explicar el resultado del dilema del prisionero
    print("\nInterpretación: En este juego puro-estrategia el equilibrio es (T,T) con pagos (1,1).")

def modo_interactivo():
    print("\n--- MODO INTERACTIVO: Define tu juego de 2 jugadores ---")
    
    # Solicitar el número de estrategias para el Jugador 1
    n1 = int(input("¿Cuántas estrategias tiene el Jugador 1? ").strip())
    estrategias_j1 = []
    # Recopilar las etiquetas de cada estrategia del Jugador 1
    for i in range(n1):
        estrategias_j1.append(input(f"  Etiqueta estrategia {i+1} jugador1: ").strip())
    
    # Solicitar el número de estrategias para el Jugador 2
    n2 = int(input("¿Cuántas estrategias tiene el Jugador 2? ").strip())
    estrategias_j2 = []
    # Recopilar las etiquetas de cada estrategia del Jugador 2
    for j in range(n2):
        estrategias_j2.append(input(f"  Etiqueta estrategia {j+1} jugador2: ").strip())
    
    # Construir la matriz de pagos pidiendo el pago para cada combinación de estrategias
    matriz_pagos = {}
    print("\nAhora introduce los pagos (jugador1, jugador2) para cada combinación:")
    # Iterar sobre todas las combinaciones posibles de estrategias
    for e1 in estrategias_j1:
        for e2 in estrategias_j2:
            # Solicitar el pago que recibe cada jugador en esta combinación
            p1 = float(input(f"  Pago jugador1 para ({e1}, {e2}): ").strip())
            p2 = float(input(f"  Pago jugador2 para ({e1}, {e2}): ").strip())
            # Almacenar en la matriz: (estrategia1, estrategia2) -> (pago1, pago2)
            matriz_pagos[(e1, e2)] = (p1, p2)
    
    # Mostrar la matriz de pagos completa que el usuario definió
    print("\nTu matriz de pagos:")
    for (e1,e2),(p1,p2) in matriz_pagos.items():
        print(f"  ({e1},{e2}) → ({p1},{p2})")
    
    # Calcular y mostrar los equilibrios de Nash del juego definido
    equilibrios = encontrar_equilibrios_nash(matriz_pagos)
    print("\nEquilibrio(s) de Nash encontrado(s):", equilibrios)
    if equilibrios:
        # Listar todos los perfiles de estrategia que son equilibrios
        for eq in equilibrios:
            print(f"  → Perfil de estrategia Equilibrio de Nash: {eq}")
    else:
        # Informar si no hay equilibrios en estrategias puras
        print("  No se encontró equilibrio en estrategias puras.")
    
    print("\nPrograma terminado.")

def main():
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()
    if opcion == '2':
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
