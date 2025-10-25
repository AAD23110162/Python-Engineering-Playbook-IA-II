"""
035-E1-QLearning.py
-------------------
Este script implementa el algoritmo Q-Learning clásico para Aprendizaje por Refuerzo:
- El agente aprende una política óptima π* interactuando con el entorno y actualizando Q(s,a).

Características:
- Entornos discretos: lista de estados, transiciones P(s'|s,a), recompensas R(s,a,s').
- Selección de acción: ε-greedy.
- Actualización Q-Learning: Q ← Q + α [ r + γ·max_a' Q(s',a') − Q ]
- Modo DEMO y modo INTERACTIVO.
- Opción verbose para ver el proceso de aprendizaje paso a paso.

Autor: Alejandro Aguirre Díaz
"""

import random
from collections import defaultdict

# ========== Funciones utilitarias ==========
def acciones_disponibles(estado, transiciones):
    """
    Devuelve la lista de acciones disponibles en un estado según el diccionario de transiciones.
    """
    disponibles = []
    # Recorremos todas las tuplas (estado, accion) en las transiciones
    for (s, a) in transiciones.keys():
        # Si el estado coincide y la acción aún no está en la lista
        if s == estado and a not in disponibles:
            disponibles.append(a)
    return disponibles

def elegir_accion_epsilon_greedy(Q, estado, transiciones, epsilon):
    """
    Selecciona una acción ε-greedy respecto a Q en el estado dado.
    """
    # Obtenemos las acciones disponibles en el estado actual
    acciones = acciones_disponibles(estado, transiciones)
    if not acciones:
        return None
    # Con probabilidad epsilon, exploramos (acción aleatoria)
    if random.random() < epsilon:
        return random.choice(acciones)
    # Con probabilidad 1-epsilon, explotamos (acción con mayor Q)
    mejor_a = max(acciones, key=lambda a: Q[(estado, a)])
    return mejor_a

# ========== Algoritmo Q-Learning ==========
def q_learning(estados, transiciones, recompensas, gamma, alpha, epsilon, episodios, max_pasos=100, verbose=False):
    """
    Algoritmo Q-Learning clásico.
    :parametro verbose: si True, imprime el proceso paso a paso.
    """
    # Inicializamos la tabla Q con valores en 0.0
    Q = defaultdict(float)
    
    # Iteramos sobre el número de episodios especificado
    for ep in range(episodios):
        # Comenzamos en un estado aleatorio
        s = random.choice(estados)
        
        # Simulamos hasta max_pasos o hasta terminar
        for t in range(max_pasos):
            # Elegimos una acción usando estrategia ε-greedy
            a = elegir_accion_epsilon_greedy(Q, s, transiciones, epsilon)
            if a is None:
                break  # No hay acciones disponibles
            
            # Obtenemos la distribución de estados siguientes para (s, a)
            distrib = transiciones.get((s, a), {})
            if not distrib:
                break  # Sin transición definida
            
            # Muestreamos el siguiente estado según la distribución de probabilidad
            estados_sig = list(distrib.keys())
            probs = list(distrib.values())
            s_prime = random.choices(estados_sig, weights=probs)[0]
            
            # Obtenemos la recompensa por la transición (s, a, s')
            r = recompensas.get((s, a, s_prime), 0.0)
            
            # Calculamos el valor máximo de Q en el estado siguiente s'
            acciones_s_prime = acciones_disponibles(s_prime, transiciones)
            max_q_s_prime = max((Q[(s_prime, ap)] for ap in acciones_s_prime), default=0.0)
            
            # Calculamos el objetivo TD: r + γ·max_a' Q(s',a')
            td_objetivo = r + gamma * max_q_s_prime
            
            # Calculamos el error TD: objetivo - Q actual
            td_error = td_objetivo - Q[(s, a)]
            
            # Actualizamos Q con la regla de Q-Learning: Q ← Q + α·TD_error
            Q[(s, a)] += alpha * td_error
            
            # Si verbose, imprimimos el proceso paso a paso
            if verbose:
                print(f"Episodio {ep+1}, paso {t+1}: s={s}, a={a}, r={r}, s'={s_prime}, TD={td_error:.3f}, Q[{s},{a}]={Q[(s,a)]:.3f}")
            
            # Transicionamos al siguiente estado
            s = s_prime
    
    # Después del entrenamiento, extraemos la política y los valores de estado
    V = {}
    politica = {}
    for s in estados:
        accs = acciones_disponibles(s, transiciones)
        if accs:
            # La mejor acción es aquella con mayor Q(s,a)
            mejor_a = max(accs, key=lambda a: Q[(s, a)])
            V[s] = Q[(s, mejor_a)]  # V(s) ≈ max_a Q(s,a)
            politica[s] = mejor_a
        else:
            V[s] = 0.0
            politica[s] = None
    
    return Q, V, politica

# ========== Modo DEMO ==========
def modo_demo():
    print("\nMODO DEMO: Q-Learning clásico")
    print("=" * 60)
    
    # Definimos el entorno: 3 estados (A, B, C)
    estados = ['A', 'B', 'C']
    
    # Definimos las transiciones: P(s'|s,a)
    # Ejemplo: desde A con acción x, vamos a A con prob 0.8 y a B con prob 0.2
    transiciones = {
        ('A','x'): {'A':0.8,'B':0.2},
        ('A','y'): {'B':1.0},
        ('B','x'): {'C':1.0},
        ('B','y'): {'A':0.5,'C':0.5},
        ('C','x'): {'C':1.0},
        ('C','y'): {'A':1.0}
    }
    
    # Definimos las recompensas: R(s,a,s')
    recompensas = {
        ('A','x','A'): 2,
        ('A','x','B'): 0,
        ('A','y','B'): 1,
        ('B','x','C'): 4,
        ('B','y','A'): -1,
        ('B','y','C'): 3,
        ('C','x','C'): 0,
        ('C','y','A'): 2
    }
    
    # Parámetros del algoritmo Q-Learning
    gamma = 0.9       # Factor de descuento
    alpha = 0.5       # Tasa de aprendizaje
    epsilon = 0.2     # Probabilidad de exploración
    episodios = 3000  # Número de episodios de entrenamiento
    verbose = False   # Sin traza detallada
    
    print("\nCONFIGURACIÓN:")
    print(f"Estados: {estados}")
    print(f"γ={gamma}, α={alpha}, ε={epsilon}, episodios={episodios}")
    
    # Ejecutamos el algoritmo Q-Learning
    print("\nEntrenando con Q-Learning...")
    Q, V, politica = q_learning(estados, transiciones, recompensas, gamma, alpha, epsilon, episodios, verbose=verbose)
    
    # Mostramos los resultados
    print("\nRESULTADOS:")
    print("Política aprendida π*(aprox):")
    for s in estados:
        print(f"  {s} -> {politica[s]}")
    print("\nValores de estado V(s) ≈ max_a Q(s,a):")
    for s in estados:
        print(f"  {s}: {V[s]:.3f}")

# ========== Modo INTERACTIVO ==========
def modo_interactivo():
    print("\nMODO INTERACTIVO: Q-Learning clásico")
    print("=" * 60)
    
    # Mostramos los escenarios disponibles
    print("\nEscenarios predefinidos:")
    print("1) Entorno simple 3 estados (A, B, C)")
    print("2) GridWorld 2x2")
    opcion = input("\nIntroduce el número de escenario: ").strip()
    
    # Configuramos el entorno según la elección del usuario
    if opcion == '2':
        # Escenario GridWorld 2x2
        estados = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
        
        # Transiciones deterministas en el grid
        transiciones = {
            ('(0,0)','derecha'): {'(0,1)':1.0},
            ('(0,0)','abajo'): {'(1,0)':1.0},
            ('(0,1)','izquierda'): {'(0,0)':1.0},
            ('(0,1)','abajo'): {'(1,1)':1.0},
            ('(1,0)','arriba'): {'(0,0)':1.0},
            ('(1,0)','derecha'): {'(1,1)':1.0},
            ('(1,1)','arriba'): {'(0,1)':1.0},
            ('(1,1)','izquierda'): {'(1,0)':1.0},
            ('(1,1)','abajo'): {'(1,1)':1.0}
        }
        
        # Recompensas: objetivo en (1,1) con recompensa +10
        recompensas = {
            ('(0,0)','derecha','(0,1)'): -1,
            ('(0,0)','abajo','(1,0)'): -1,
            ('(0,1)','izquierda','(0,0)'): -1,
            ('(0,1)','abajo','(1,1)'): 10,
            ('(1,0)','arriba','(0,0)'): -1,
            ('(1,0)','derecha','(1,1)'): 10,
            ('(1,1)','arriba','(0,1)'): -1,
            ('(1,1)','izquierda','(1,0)'): -1,
            ('(1,1)','abajo','(1,1)'): 0
        }
        print("\nHas elegido GridWorld 2x2")
    else:
        # Escenario simple de 3 estados
        estados = ['A','B','C']
        
        # Transiciones estocásticas
        transiciones = {
            ('A','x'): {'A':0.8,'B':0.2},
            ('A','y'): {'B':1.0},
            ('B','x'): {'C':1.0},
            ('B','y'): {'A':0.5,'C':0.5},
            ('C','x'): {'C':1.0},
            ('C','y'): {'A':1.0}
        }
        
        # Recompensas variadas
        recompensas = {
            ('A','x','A'): 2,
            ('A','x','B'): 0,
            ('A','y','B'): 1,
            ('B','x','C'): 4,
            ('B','y','A'): -1,
            ('B','y','C'): 3,
            ('C','x','C'): 0,
            ('C','y','A'): 2
        }
        print("\nHas elegido el entorno simple de 3 estados")
    
    print(f"\nEstados disponibles: {estados}")
    
    # Solicitamos los parámetros al usuario
    try:
        gamma = float(input("\nIntroduce factor de descuento γ (0-1, ej 0.9): ").strip())
        alpha = float(input("Introduce tasa de aprendizaje α (0-1, ej 0.5): ").strip())
        epsilon = float(input("Introduce exploración ε (0-1, ej 0.2): ").strip())
        episodios = int(input("Introduce número de episodios (ej 5000): ").strip())
        verbose = input("¿Mostrar proceso paso a paso? (s/n): ").strip().lower() == 's'
    except Exception:
        # Si hay error, usamos valores por defecto
        print("Parámetros inválidos. Usando valores por defecto.")
        gamma, alpha, epsilon, episodios, verbose = 0.9, 0.5, 0.2, 3000, False
    
    # Ejecutamos el algoritmo Q-Learning
    print("\nEntrenando con Q-Learning...")
    Q, V, politica = q_learning(estados, transiciones, recompensas, gamma, alpha, epsilon, episodios, verbose=verbose)
    
    # Mostramos los resultados
    print("\nRESULTADOS:")
    print("Política aprendida π*(aprox):")
    for s in estados:
        print(f"  {s} -> {politica[s]}")
    print("\nValores de estado V(s) ≈ max_a Q(s,a):")
    for s in estados:
        print(f"  {s}: {V[s]:.3f}")

# ========== main ==========
def main():
    # Mostramos el menú principal
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    
    # Leemos la opción del usuario
    opcion = input("Ingrese opción: ").strip()
    
    # Ejecutamos el modo correspondiente
    if opcion == '2':
        modo_interactivo()  # Modo con configuración personalizada
    else:
        modo_demo()  # Modo con valores predefinidos

if __name__ == "__main__":
    main()
