"""
027-E1-iteracion_de_valores_MDP.py
--------------------------------------
Este script implementa el algoritmo de **Iteración de Valores** para un Proceso de Decisión de Márkov (MDP):
- Define un conjunto de estados, acciones, probabilidades de transición y recompensas.
- Ejecuta la iteración de valores para encontrar la función de valor óptima V* y derivar la política óptima π*.
- Modo DEMO: problema predefinido con visualización paso por paso.
- Modo INTERACTIVO: usuario define estados, acciones, probabilidades, recompensas y observa cada iteración.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def inicializar_valores(estados, valor_inicial=0.0):
    """
    Inicializa la función de valor V(s) = valor_inicial para todos los estados.
    :parametro estados: lista de etiquetas de estados
    :parametro valor_inicial: valor numérico para iniciar
    :return: dict estado -> valor
    """
    # Crea un diccionario con cada estado y el valor inicial
    return {s: valor_inicial for s in estados}

def actualizar_valor(estado, acciones, P, R, V_ant, gamma):
    """
    Calcula el nuevo valor de V(estado) usando la ecuación de Bellman-óptima:
    V'(s) = max_a Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ·V_ant(s') ]
    :parametro estado: etiqueta de estado
    :parametro acciones: lista de acciones posibles
    :parametro P: dict: (s,a) -> dict de s'->probabilidad
    :parametro R: dict: (s,a,s') -> recompensa
    :parametro V_ant: dict estado->valor de la iteración anterior
    :parametro gamma: factor de descuento (0 ≤ γ < 1)
    :return: nuevo valor numérico de V(estado)
    """
    mejor = float('-inf')
    # Para cada acción posible desde el estado actual
    for a in acciones:
        suma = 0.0
        # Obtiene las transiciones y probabilidades para (estado, acción)
        trans = P.get((estado,a), {})
        # Para cada estado siguiente y su probabilidad
        for s2, p in trans.items():
            # Obtiene la recompensa por la transición
            r = R.get((estado,a,s2), 0.0)
            # Suma la contribución: probabilidad * (recompensa + valor descontado del siguiente estado)
            suma += p * (r + gamma * V_ant.get(s2, 0.0))
        # Actualiza el mejor valor si esta acción es superior
        if suma > mejor:
            mejor = suma
    # Si no hay acciones válidas, retorna 0.0
    return mejor if mejor != float('-inf') else 0.0

def derivar_politica(estados, acciones, P, R, V, gamma):
    """
    Deriva la política óptima π(s) = argmax_a Σ_{s'} P(s'|s,a)[ R(s,a,s') + γ·V(s') ]
    :parametro estados: lista de estados
    :parametro acciones: lista de acciones
    :parametro P, R, V, gamma: como anteriormente
    :return: dict estado->acción óptima
    """
    politica = {}
    # Para cada estado, busca la acción óptima
    for s in estados:
        mejor_acc = None
        mejor_val = float('-inf')
        # Evalúa cada acción posible
        for a in acciones:
            suma = 0.0
            # Obtiene las transiciones y probabilidades para (estado, acción)
            trans = P.get((s,a), {})
            # Suma la contribución de cada transición
            for s2, p in trans.items():
                r = R.get((s,a,s2), 0.0)
                suma += p * (r + gamma * V.get(s2,0.0))
            # Si esta acción es mejor, la guarda
            if suma > mejor_val:
                mejor_val = suma
                mejor_acc = a
        # Asigna la mejor acción encontrada para el estado
        politica[s] = mejor_acc
    return politica

def iteracion_de_valores(estados, acciones, P, R, gamma=0.9, theta=1e-6, max_iter=1000):
    """
    Ejecuta el algoritmo de iteración de valores mostrando el proceso paso-a-paso:
    :parametro estados: lista de estados
    :parametro acciones: lista de acciones
    :parametro P: dict (s,a)->dict s'->probabilidad
    :parametro R: dict (s,a,s')→recompensa
    :parametro gamma: factor de descuento
    :parametro theta: umbral de convergencia
    :parametro max_iter: número máximo de iteraciones permitidas
    :return: (V, π) donde V es dict estado->valor óptimo, π estado->acción óptima
    """
    V = inicializar_valores(estados)
    # Muestra la inicialización de valores
    print("Iteración 0:", V)
    for k in range(1, max_iter+1):
        delta = 0.0
        V_new = V.copy()
        # Muestra el número de iteración
        print(f"\n*** Iteración {k} ***")
        for s in estados:
            v_ant = V[s]
            # Calcula el nuevo valor para el estado usando Bellman
            v_nuevo = actualizar_valor(s, acciones, P, R, V, gamma)
            V_new[s] = v_nuevo
            # Calcula el cambio absoluto para verificar convergencia
            cambio = abs(v_ant - v_nuevo)
            delta = max(delta, cambio)
            # Muestra el valor anterior, nuevo y el cambio
            print(f"  Estado {s}: V_old = {v_ant:.4f}, V_new = {v_nuevo:.4f}, cambio = {cambio:.4f}")
        # Actualiza los valores para la siguiente iteración
        V = V_new
        # Muestra el mayor cambio observado en esta iteración
        print(f"  Valor máximo de cambio (δ) = {delta:.6f}")
        # Si el cambio es menor al umbral, se considera convergido
        if delta < theta:
            print("Convergencia alcanzada (δ < θ).")
            break
    # Deriva la política óptima a partir de los valores finales
    politica = derivar_politica(estados, acciones, P, R, V, gamma)
    # Muestra los resultados finales
    print("\nResultado final de V:", V)
    print("Política óptima derivada π*:", politica)
    return V, politica

def modo_demo():
    print("\n--- MODO DEMO ---")
    # Define un MDP de ejemplo con 3 estados y 2 acciones
    estados = ['s0','s1','s2']
    acciones = ['a0','a1']
    # Diccionario de probabilidades de transición P(s,a→s')
    P = {
        ('s0','a0'): {'s0': 0.5, 's1': 0.5},
        ('s0','a1'): {'s1': 1.0},
        ('s1','a0'): {'s2': 1.0},
        ('s1','a1'): {'s0': 0.3, 's2': 0.7},
        ('s2','a0'): {'s2': 1.0},
        ('s2','a1'): {'s0': 1.0},
    }
    # Diccionario de recompensas R(s,a,s')
    R = {
        ('s0','a0','s0'): 1,   ('s0','a0','s1'): 0,
        ('s0','a1','s1'): 5,
        ('s1','a0','s2'): 10,
        ('s1','a1','s0'): -1,  ('s1','a1','s2'): 2,
        ('s2','a0','s2'): 0,
        ('s2','a1','s0'): 0,
    }
    # Factor de descuento
    gamma = 0.9
    # Ejecuta la iteración de valores y muestra resultados
    V, π = iteracion_de_valores(estados, acciones, P, R, gamma)
    print(f"\nValor inicial (estado {estados[0]}): V({estados[0]}) ≈ {V[estados[0]]:.4f}")

def modo_interactivo():
    print("\n--- MODO INTERACTIVO SIMPLIFICADO ---")
    # Ofrece al usuario elegir entre datasets precargados
    print("Elige un dataset de MDP precargado para ejecutar la iteración de valores:\n")
    datasets = {
        "1": {
            "nombre": "Ejemplo clásico (3 estados, 2 acciones)",
            "estados": ['s0','s1','s2'],
            "acciones": ['a0','a1'],
            "P": {
                ('s0','a0'): {'s0': 0.5, 's1': 0.5},
                ('s0','a1'): {'s1': 1.0},
                ('s1','a0'): {'s2': 1.0},
                ('s1','a1'): {'s0': 0.3, 's2': 0.7},
                ('s2','a0'): {'s2': 1.0},
                ('s2','a1'): {'s0': 1.0},
            },
            "R": {
                ('s0','a0','s0'): 1,   ('s0','a0','s1'): 0,
                ('s0','a1','s1'): 5,
                ('s1','a0','s2'): 10,
                ('s1','a1','s0'): -1,  ('s1','a1','s2'): 2,
                ('s2','a0','s2'): 0,
                ('s2','a1','s0'): 0,
            },
            "gamma": 0.9
        },
        "2": {
            "nombre": "MDP simple (2 estados, 2 acciones)",
            "estados": ['A','B'],
            "acciones": ['ir','esperar'],
            "P": {
                ('A','ir'): {'B': 1.0},
                ('A','esperar'): {'A': 1.0},
                ('B','ir'): {'A': 0.5, 'B': 0.5},
                ('B','esperar'): {'B': 1.0},
            },
            "R": {
                ('A','ir','B'): 10,
                ('A','esperar','A'): 0,
                ('B','ir','A'): 5, ('B','ir','B'): -2,
                ('B','esperar','B'): 1,
            },
            "gamma": 0.8
        },
        "3": {
            "nombre": "MDP de inventario (3 estados, 2 acciones)",
            "estados": ['bajo','medio','alto'],
            "acciones": ['ordenar','no_ordenar'],
            "P": {
                ('bajo','ordenar'): {'medio': 1.0},
                ('bajo','no_ordenar'): {'bajo': 1.0},
                ('medio','ordenar'): {'alto': 0.7, 'medio': 0.3},
                ('medio','no_ordenar'): {'bajo': 0.4, 'medio': 0.6},
                ('alto','ordenar'): {'alto': 1.0},
                ('alto','no_ordenar'): {'medio': 1.0},
            },
            "R": {
                ('bajo','ordenar','medio'): 5,
                ('bajo','no_ordenar','bajo'): -2,
                ('medio','ordenar','alto'): 10, ('medio','ordenar','medio'): 2,
                ('medio','no_ordenar','bajo'): -3, ('medio','no_ordenar','medio'): 1,
                ('alto','ordenar','alto'): 0,
                ('alto','no_ordenar','medio'): -1,
            },
            "gamma": 0.85
        }
    }
    # Muestra las opciones disponibles
    for k, v in datasets.items():
        print(f"{k}) {v['nombre']}")
    # Solicita al usuario la elección
    eleccion = input("\nElige el número de dataset: ").strip()
    if eleccion not in datasets:
        print("Opción no válida. Usando el dataset 1 por defecto.")
        eleccion = "1"
    ds = datasets[eleccion]
    # Ejecuta la iteración de valores sobre el dataset elegido
    print(f"\nEjecutando iteración de valores para: {ds['nombre']}")
    V, π = iteracion_de_valores(ds['estados'], ds['acciones'], ds['P'], ds['R'], ds['gamma'])
    print(f"\nValor estimado para estado inicial {ds['estados'][0]}: V({ds['estados'][0]}) ≈ {V[ds['estados'][0]]:.4f}")

def main():
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida. Se usará MODO DEMO.")
        modo_demo()

if __name__ == "__main__":
    main()
