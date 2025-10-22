"""
028-E1-iteracion_de_politicas.py
---------------------------------
Este script implementa el algoritmo de Iteración de Políticas para un MDP:
- Alterna entre: Evaluación de política (policy evaluation) y Mejora de política (policy improvement). :contentReference[oaicite:1]{index=1}
- Modo DEMO: escenario predefinido sencillo con visualización de cada iteración.
- Modo INTERACTIVO: el usuario define estados, acciones, transiciones y recompensas, y se muestra el proceso completo.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def inicializar_politica(estados, acciones):
    """
    Inicializa una política arbitraria (por ejemplo la primera acción para cada estado).
    :parametro estados: lista de estados
    :parametro acciones: lista de acciones posibles
    :return: dict estado -> acción
    """
    politica = {}
    # Asigna la primera acción a todos los estados como política inicial
    for s in estados:
        politica[s] = acciones[0]
    return politica

def evaluacion_politica(politica, estados, acciones, P, R, gamma=0.9, theta=1e-6, max_iter=1000):
    """
    Evalúa la política dada, calculando V^π hasta converger.
    :parametro politica: dict estado->acción
    :return: dict estado->valor
    """
    # Inicializa todos los valores a 0.0
    V = {s: 0.0 for s in estados}
    print("Iniciando evaluación de política:", politica)
    
    # Itera hasta convergencia o máximo de iteraciones
    for it in range(1, max_iter+1):
        delta = 0.0
        print(f"\n  Iteración de evaluación {it}")
        
        # Para cada estado, actualiza su valor basado en la política actual
        for s in estados:
            # Obtiene la acción que la política actual asigna a este estado
            a = politica[s]
            v_ant = V[s]
            suma = 0.0
            
            # Suma sobre todos los posibles estados siguientes
            for s2, p in P.get((s,a), {}).items():
                # Obtiene la recompensa de la transición
                r = R.get((s,a,s2), 0.0)
                # Acumula: probabilidad * (recompensa + valor descontado del siguiente estado)
                suma += p * (r + gamma * V[s2])
            
            # Actualiza el valor del estado
            V[s] = suma
            cambio = abs(v_ant - V[s])
            delta = max(delta, cambio)
            print(f"    Estado {s}: acción {a}, V_old={v_ant:.4f}, V_nuevo={V[s]:.4f}, Δ={cambio:.4f}")
        
        print(f"  Δ máxima = {delta:.6f}")
        # Si el cambio es menor al umbral, convergió
        if delta < theta:
            print("  → Convergencia alcanzada en evaluación de política.")
            break
    
    print("Valores finales V^π:", V)
    return V

def mejora_politica(V, estados, acciones, P, R, politica, gamma=0.9):
    """
    Mejora la política actual basándose en V.
    :parametro politica: política actual dict estado->acción
    :return: nueva política, indicador si cambio ocurrió
    """
    politica_nueva = {}
    cambio = False
    print("\nMejora de política:")
    
    # Para cada estado, encuentra la mejor acción
    for s in estados:
        mejor_accion = None
        mejor_val = float('-inf')
        
        # Evalúa todas las acciones posibles para este estado
        for a in acciones:
            suma = 0.0
            # Calcula el valor esperado de tomar la acción 'a' en el estado 's'
            for s2, p in P.get((s,a), {}).items():
                r = R.get((s,a,s2), 0.0)
                suma += p * (r + gamma * V[s2])
            print(f"    Estado {s}: acción {a} → valor estimado = {suma:.4f}")
            
            # Si esta acción es mejor, la guarda
            if suma > mejor_val:
                mejor_val = suma
                mejor_accion = a
        
        # Asigna la mejor acción encontrada
        politica_nueva[s] = mejor_accion
        
        # Verifica si hubo cambio en la política para este estado
        if politica.get(s) is None:
            # Si no existía política previa para el estado, consideramos que hubo cambio
            cambio = True
            print(f"    Nueva política para estado {s}: {mejor_accion}")
        elif mejor_accion != politica[s]:
            cambio = True
            print(f"    Cambio de política en estado {s}: {politica[s]} → {mejor_accion}")
    
    return politica_nueva, cambio
def iteracion_de_politicas(estados, acciones, P, R, gamma=0.9):
    """
    Ejecuta el algoritmo de Iteración de Políticas mostrando todo el proceso.
    :return: (política óptima, valores óptimos)
    """
    # Inicializa con una política arbitraria
    politica = inicializar_politica(estados, acciones)
    print("Política inicial:", politica)
    iter_count = 0
    
    # Ciclo principal: evaluar y mejorar hasta convergencia
    while True:
        iter_count += 1
        print(f"\n=== Iteración de política #{iter_count} ===")
        
        # Paso 1: Evaluar la política actual
        V = evaluacion_politica(politica, estados, acciones, P, R, gamma)
        
        # Paso 2: Mejorar la política basándose en V
        politica_nueva, cambio = mejora_politica(V, estados, acciones, P, R, politica, gamma)
        
        # Si no hubo cambios, la política es óptima
        if not cambio:
            print("\nNo hubo cambio en la política → política óptima encontrada.")
            break
        
        # Actualiza la política para la siguiente iteración
        politica = politica_nueva
    
    print("\nResultado final:")
    print("Política óptima π*:", politica)
    print("Valores óptimos V*:", V)
    return politica, V


def modo_demo():
    print("\n--- MODO DEMO ---")
    # Define un MDP de ejemplo con 3 estados y 2 acciones
    estados = ['s0','s1','s2']
    acciones = ['a0','a1']
    # Probabilidades de transición P(s,a→s')
    P = {
        ('s0','a0'): {'s0':0.5, 's1':0.5},
        ('s0','a1'): {'s1':1.0},
        ('s1','a0'): {'s2':1.0},
        ('s1','a1'): {'s0':0.3,'s2':0.7},
        ('s2','a0'): {'s2':1.0},
        ('s2','a1'): {'s0':1.0}
    }
    # Recompensas R(s,a,s')
    R = {
        ('s0','a0','s0'):1,   ('s0','a0','s1'):0,
        ('s0','a1','s1'):5,
        ('s1','a0','s2'):10,
        ('s1','a1','s0'):-1,  ('s1','a1','s2'):2,
        ('s2','a0','s2'):0,
        ('s2','a1','s0'):0
    }
    # Factor de descuento
    gamma = 0.9
    # Ejecuta la iteración de políticas
    politica_opt, V_opt = iteracion_de_politicas(estados, acciones, P, R, gamma)

def modo_interactivo():
    print("\n--- MODO INTERACTIVO SIMPLIFICADO ---")
    print("Elige un dataset de MDP precargado para ejecutar la iteración de políticas:\n")
    datasets = {
        "1": {
            "nombre": "Ejemplo clásico (3 estados, 2 acciones)",
            "estados": ['s0','s1','s2'],
            "acciones": ['a0','a1'],
            "P": {
                ('s0','a0'): {'s0':0.5, 's1':0.5},
                ('s0','a1'): {'s1':1.0},
                ('s1','a0'): {'s2':1.0},
                ('s1','a1'): {'s0':0.3,'s2':0.7},
                ('s2','a0'): {'s2':1.0},
                ('s2','a1'): {'s0':1.0}
            },
            "R": {
                ('s0','a0','s0'):1,   ('s0','a0','s1'):0,
                ('s0','a1','s1'):5,
                ('s1','a0','s2'):10,
                ('s1','a1','s0'):-1,  ('s1','a1','s2'):2,
                ('s2','a0','s2'):0,
                ('s2','a1','s0'):0
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
            "nombre": "MDP de navegación (4 estados, 3 acciones)",
            "estados": ['inicio','intermedio','objetivo','trampa'],
            "acciones": ['avanzar','esperar','retroceder'],
            "P": {
                ('inicio','avanzar'): {'intermedio': 1.0},
                ('inicio','esperar'): {'inicio': 1.0},
                ('inicio','retroceder'): {'inicio': 1.0},
                ('intermedio','avanzar'): {'objetivo': 0.6, 'trampa': 0.4},
                ('intermedio','esperar'): {'intermedio': 1.0},
                ('intermedio','retroceder'): {'inicio': 1.0},
                ('objetivo','avanzar'): {'objetivo': 1.0},
                ('objetivo','esperar'): {'objetivo': 1.0},
                ('objetivo','retroceder'): {'intermedio': 1.0},
                ('trampa','avanzar'): {'trampa': 1.0},
                ('trampa','esperar'): {'trampa': 1.0},
                ('trampa','retroceder'): {'intermedio': 1.0},
            },
            "R": {
                ('inicio','avanzar','intermedio'): 0,
                ('inicio','esperar','inicio'): -1,
                ('inicio','retroceder','inicio'): -1,
                ('intermedio','avanzar','objetivo'): 100,
                ('intermedio','avanzar','trampa'): -50,
                ('intermedio','esperar','intermedio'): -1,
                ('intermedio','retroceder','inicio'): -5,
                ('objetivo','avanzar','objetivo'): 10,
                ('objetivo','esperar','objetivo'): 10,
                ('objetivo','retroceder','intermedio'): -10,
                ('trampa','avanzar','trampa'): -10,
                ('trampa','esperar','trampa'): -10,
                ('trampa','retroceder','intermedio'): -5,
            },
            "gamma": 0.9
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
    # Ejecuta la iteración de políticas sobre el dataset elegido
    print(f"\nEjecutando iteración de políticas para: {ds['nombre']}")
    politica_opt, V_opt = iteracion_de_politicas(ds['estados'], ds['acciones'], ds['P'], ds['R'], ds['gamma'])
    print(f"\nValor estimado para estado inicial {ds['estados'][0]}: V({ds['estados'][0]}) ≈ {V_opt[ds['estados'][0]]:.4f}")

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
