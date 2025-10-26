"""
037-E1-busqueda_politica.py
---------------------------
Este script implementa algoritmos de Búsqueda de Políticas (Policy Search) para
Aprendizaje por Refuerzo, donde optimizamos directamente la política π en lugar
de aprender una función de valor.

Métodos implementados:
1. Hill Climbing (Ascenso de Colinas) para políticas
2. Policy Gradient (Gradiente de Política) - REINFORCE
3. Cross-Entropy Method (Método de Entropía Cruzada)
4. Evolution Strategies (Estrategias Evolutivas)

Características:
- Políticas paramétricas θ: π(a|s,θ)
- Optimización directa de la recompensa esperada
- Modo DEMO y modo INTERACTIVO
- Opción verbose para ver el proceso paso a paso

Autor: Alejandro Aguirre Díaz
"""

import random
import math

# ========== Funciones utilitarias ==========
def crear_entorno_simple():
    """
    Crea un entorno simple de 3 estados con transiciones deterministas.
    
    Este entorno sirve como banco de pruebas mínimo para comparar métodos
    de búsqueda de políticas. No hay estados terminales; los episodios
    se truncan por longitud máxima.
    Estados: A, B, C
    
    """
    estados = ['A', 'B', 'C']
    
    # Acciones: 0, 1
    acciones = [0, 1]
    
    # Transiciones deterministas: (estado, accion) -> nuevo_estado
    transiciones = {
        ('A', 0): 'A',
        ('A', 1): 'B',
        ('B', 0): 'C',
        ('B', 1): 'A',
        ('C', 0): 'C',
        ('C', 1): 'A'
    }
    # Intuición: algunas acciones causan bucles (p.ej., C con 0), otras avanzan.
    
    # Recompensas: (estado, accion) -> recompensa
    recompensas = {
        ('A', 0): 1,
        ('A', 1): 2,
        ('B', 0): 5,
        ('B', 1): 0,
        ('C', 0): 0,
        ('C', 1): 3
    }
    # La señal de recompensa incentiva, por ejemplo, ir de B a C (recompensa 5)
    # y desde C volver a A con la acción 1 (recompensa 3).
    
    return estados, acciones, transiciones, recompensas

def politica_parametrizada(estado, theta, acciones):
    """
    Política paramétrica simple basada en θ.
    θ es un diccionario: θ[estado] = probabilidad de acción 1
    """
    # Nota: 'acciones' se incluye por consistencia de firma, pero esta política
    # usa un único parámetro por estado: P(a=1|s) = theta[s].
    # Obtenemos la probabilidad de tomar la acción 1
    prob_accion_1 = theta.get(estado, 0.5)
    
    # Aseguramos que esté en [0, 1]
    prob_accion_1 = max(0.0, min(1.0, prob_accion_1))
    
    # Generamos la acción según la probabilidad
    # Lanzamos una moneda sesgada con prob_accion_1 para a=1; en otro caso a=0.
    if random.random() < prob_accion_1:
        return 1
    else:
        return 0

def evaluar_politica(theta, estados, transiciones, recompensas, episodios=100, max_pasos=20):
    """
    Evalúa una política θ ejecutando episodios y calculando la recompensa promedio.
    """
    recompensa_total = 0.0
    
    # Ejecutamos múltiples episodios para obtener una estimación robusta
    for _ in range(episodios):
        # Comenzamos en un estado aleatorio
        s = random.choice(estados)
        recompensa_episodio = 0.0
        
        # Ejecutamos el episodio
        for _ in range(max_pasos):
            # Elegimos una acción según la política
            a = politica_parametrizada(s, theta, [0, 1])
            
            # Obtenemos la recompensa
            r = recompensas.get((s, a), 0.0)
            recompensa_episodio += r
            
            # Transicionamos al siguiente estado
            s = transiciones.get((s, a), s)
        
        recompensa_total += recompensa_episodio
    
    # Retornamos la recompensa promedio por episodio
    # No aplicamos descuento aquí; cada episodio aporta su suma de recompensas.
    return recompensa_total / episodios

# ========== Método 1: Hill Climbing para Políticas ==========
def hill_climbing_politica(estados, transiciones, recompensas, iteraciones=100, paso=0.1, verbose=False):
    """
    Hill Climbing para búsqueda de políticas.
    Ajusta los parámetros θ de la política en dirección ascendente.
    """
    # Inicializamos θ aleatoriamente
    theta = {s: random.uniform(0.3, 0.7) for s in estados}
    # El parámetro 'paso' controla la magnitud típica de la perturbación.
    
    # Evaluamos la política inicial
    mejor_recompensa = evaluar_politica(theta, estados, transiciones, recompensas)
    mejor_theta = theta.copy()
    
    if verbose:
        print(f"Iteración 0: θ={theta}, R={mejor_recompensa:.2f}")
    
    # Iteramos mejorando la política
    for it in range(iteraciones):
        # Generamos un vecino modificando θ ligeramente
        theta_vecino = {}
        for s in estados:
            # Perturbamos el parámetro con ruido gaussiano
            perturbacion = random.gauss(0, paso)
            theta_vecino[s] = theta[s] + perturbacion
        # No forzamos aquí el clamp a [0,1] porque la función de política
        # ya aplica ese recorte internamente antes de muestrear acciones.
        
        # Evaluamos el vecino
        recompensa_vecino = evaluar_politica(theta_vecino, estados, transiciones, recompensas)
        
        # Si el vecino es mejor, lo aceptamos
        if recompensa_vecino > mejor_recompensa:
            theta = theta_vecino
            mejor_recompensa = recompensa_vecino
            mejor_theta = theta.copy()
            
            if verbose and (it < 10 or it % 20 == 0):
                print(f"Iteración {it+1}: θ={theta}, R={mejor_recompensa:.2f} ✓ Mejora")
        else:
            if verbose and it < 10:
                print(f"Iteración {it+1}: R_vecino={recompensa_vecino:.2f} (no mejora)")
    
    return mejor_theta, mejor_recompensa

# ========== Método 2: REINFORCE (Policy Gradient) ==========
def reinforce_simple(estados, transiciones, recompensas, episodios=200, alpha=0.01, verbose=False):
    """
    Algoritmo REINFORCE simplificado (Policy Gradient).
    Actualiza θ en la dirección del gradiente de la recompensa esperada.
    """
    # Inicializamos θ
    theta = {s: 0.5 for s in estados}
    
    historial_recompensas = []
    
    for ep in range(episodios):
        # Generamos un episodio siguiendo π(θ)
        s = random.choice(estados)
        trayectoria = []  # Lista de (estado, acción, recompensa)
        recompensa_episodio = 0.0
        
        # Ejecutamos el episodio
        for _ in range(20):
            a = politica_parametrizada(s, theta, [0, 1])
            r = recompensas.get((s, a), 0.0)
            trayectoria.append((s, a, r))
            recompensa_episodio += r
            s = transiciones.get((s, a), s)
        
        historial_recompensas.append(recompensa_episodio)
        
        # Calculamos el retorno total G
        G = recompensa_episodio
        
        # Actualizamos θ usando el gradiente estimado
        # ∇θ J(θ) ≈ Σ_t G * ∇θ log π(a_t|s_t, θ)
        for (s_t, a_t, r_t) in trayectoria:
            # Gradiente simplificado: si a_t=1, aumentamos θ[s_t]; si a_t=0, disminuimos
            if a_t == 1:
                # Queremos aumentar P(a=1|s)
                theta[s_t] += alpha * G
            else:
                # Queremos aumentar P(a=0|s) = 1 - P(a=1|s)
                theta[s_t] -= alpha * G
        # Nota didáctica: esta actualización es una aproximación muy simple
        # de REINFORCE. No usamos un baseline (para reducir varianza) ni el
        # gradiente exacto de log π; el objetivo es ilustrativo.
        
        # Mantenemos θ en [0, 1]
        for s in estados:
            theta[s] = max(0.0, min(1.0, theta[s]))
        
        if verbose and (ep < 10 or ep % 50 == 0):
            print(f"Episodio {ep+1}: G={G:.2f}, θ={[f'{theta[s]:.2f}' for s in estados]}")
    
    # Evaluamos la política final
    recompensa_final = evaluar_politica(theta, estados, transiciones, recompensas)
    
    return theta, recompensa_final, historial_recompensas

# ========== Método 3: Cross-Entropy Method ==========
def cross_entropy_method(estados, transiciones, recompensas, generaciones=50, poblacion=20, elite_frac=0.2, verbose=False):
    """
    Método de Entropía Cruzada (CEM).
    Mantiene una distribución sobre θ y la actualiza basándose en las mejores muestras.
    """
    # Inicializamos la distribución: media y desviación estándar para cada estado
    mu = {s: 0.5 for s in estados}
    sigma = {s: 0.2 for s in estados}
    
    num_elite = max(1, int(poblacion * elite_frac))
    
    for gen in range(generaciones):
        # Generamos una población de políticas (θ's) muestreando de N(μ, σ²)
        poblacion_theta = []
        recompensas_poblacion = []
        
        for _ in range(poblacion):
            # Muestreamos θ de la distribución
            theta = {}
            for s in estados:
                theta[s] = random.gauss(mu[s], sigma[s])
                # Proyectamos a [0, 1]
                theta[s] = max(0.0, min(1.0, theta[s]))
            
            # Evaluamos la política
            r = evaluar_politica(theta, estados, transiciones, recompensas, episodios=20)
            
            poblacion_theta.append(theta)
            recompensas_poblacion.append(r)
        
        # Seleccionamos la élite (mejores políticas)
        indices_ordenados = sorted(range(poblacion), key=lambda i: recompensas_poblacion[i], reverse=True)
        elite_indices = indices_ordenados[:num_elite]
        elite_theta = [poblacion_theta[i] for i in elite_indices]
        mejor_recompensa = recompensas_poblacion[indices_ordenados[0]]
    # Solo la élite afecta la actualización de la distribución (mu, sigma).
        
        # Actualizamos μ y σ basándonos en la élite
        for s in estados:
            valores_elite = [theta[s] for theta in elite_theta]
            mu[s] = sum(valores_elite) / len(valores_elite)
            varianza = sum((v - mu[s])**2 for v in valores_elite) / len(valores_elite)
            sigma[s] = math.sqrt(varianza) + 1e-6  # Evitamos σ=0
        # La suma 1e-6 estabiliza el aprendizaje evitando colapso prematuro.
        
        if verbose and (gen < 10 or gen % 10 == 0):
            print(f"Generación {gen+1}: mejor_R={mejor_recompensa:.2f}, μ={[f'{mu[s]:.2f}' for s in estados]}")
    
    # Retornamos la mejor política encontrada (la media final)
    return mu, evaluar_politica(mu, estados, transiciones, recompensas)

# ========== Método 4: Evolution Strategies ==========
def evolution_strategies(estados, transiciones, recompensas, generaciones=50, poblacion=15, sigma=0.1, alpha=0.05, verbose=False):
    """
    Estrategias Evolutivas (ES) para búsqueda de políticas.
    Usa mutaciones gaussianas y actualiza en dirección del gradiente estimado.
    """
    # Inicializamos θ
    theta = {s: 0.5 for s in estados}
    
    for gen in range(generaciones):
        # Generamos perturbaciones y evaluamos
        perturbaciones = []
        recompensas_perturbadas = []
        
        for _ in range(poblacion):
            # Generamos una perturbación ε ~ N(0, σ²)
            epsilon = {s: random.gauss(0, sigma) for s in estados}
            
            # Creamos θ + ε
            theta_perturbado = {}
            for s in estados:
                theta_perturbado[s] = max(0.0, min(1.0, theta[s] + epsilon[s]))
            
            # Evaluamos
            r = evaluar_politica(theta_perturbado, estados, transiciones, recompensas, episodios=20)
            
            perturbaciones.append(epsilon)
            recompensas_perturbadas.append(r)
        
        # Calculamos el gradiente estimado: ∇θ J ≈ (1/(n·σ²)) Σ R_i · ε_i
        gradiente = {s: 0.0 for s in estados}
        for i in range(poblacion):
            for s in estados:
                gradiente[s] += recompensas_perturbadas[i] * perturbaciones[i][s]
        
        # Normalizamos el gradiente
        for s in estados:
            gradiente[s] /= (poblacion * sigma)
        # Esta normalización es una versión simplificada; variantes usan σ o σ²
        # según la formulación. El objetivo aquí es mantener la intuición.
        
        # Actualizamos θ
        for s in estados:
            theta[s] += alpha * gradiente[s]
            theta[s] = max(0.0, min(1.0, theta[s]))
        
        # Evaluamos la política actual
        recompensa_actual = evaluar_politica(theta, estados, transiciones, recompensas)
        
        if verbose and (gen < 10 or gen % 10 == 0):
            print(f"Generación {gen+1}: R={recompensa_actual:.2f}, θ={[f'{theta[s]:.2f}' for s in estados]}")
    
    return theta, recompensa_actual

# ========== Modo DEMO ==========
def modo_demo():
    print("\nMODO DEMO: Búsqueda de Políticas")
    print("=" * 70)
    
    # Creamos el entorno
    estados, acciones, transiciones, recompensas = crear_entorno_simple()
    
    print(f"\nENTORNO:")
    print(f"Estados: {estados}")
    print(f"Acciones: {acciones}")
    print(f"Transiciones y recompensas definidas")
    
    # Evaluamos una política aleatoria como baseline
    theta_random = {s: 0.5 for s in estados}
    r_random = evaluar_politica(theta_random, estados, transiciones, recompensas)
    print(f"\nPolítica aleatoria (θ=0.5): R={r_random:.2f}")
    # Este baseline sirve para contextualizar la mejora de cada método.
    
    # Método 1: Hill Climbing
    print("\n" + "="*70)
    print("MÉTODO 1: Hill Climbing")
    print("="*70)
    theta_hc, r_hc = hill_climbing_politica(estados, transiciones, recompensas, iteraciones=100, verbose=False)
    print(f"Política encontrada: θ={[f'{theta_hc[s]:.2f}' for s in estados]}")
    print(f"Recompensa promedio: {r_hc:.2f}")
    
    # Método 2: REINFORCE
    print("\n" + "="*70)
    print("MÉTODO 2: REINFORCE (Policy Gradient)")
    print("="*70)
    theta_pg, r_pg, _ = reinforce_simple(estados, transiciones, recompensas, episodios=200, verbose=False)
    print(f"Política encontrada: θ={[f'{theta_pg[s]:.2f}' for s in estados]}")
    print(f"Recompensa promedio: {r_pg:.2f}")
    # Nota: REINFORCE aquí no usa baseline; puede mostrar mayor varianza.
    
    # Método 3: Cross-Entropy Method
    print("\n" + "="*70)
    print("MÉTODO 3: Cross-Entropy Method")
    print("="*70)
    theta_cem, r_cem = cross_entropy_method(estados, transiciones, recompensas, generaciones=30, verbose=False)
    print(f"Política encontrada: θ={[f'{theta_cem[s]:.2f}' for s in estados]}")
    print(f"Recompensa promedio: {r_cem:.2f}")
    
    # Método 4: Evolution Strategies
    print("\n" + "="*70)
    print("MÉTODO 4: Evolution Strategies")
    print("="*70)
    theta_es, r_es = evolution_strategies(estados, transiciones, recompensas, generaciones=40, verbose=False)
    print(f"Política encontrada: θ={[f'{theta_es[s]:.2f}' for s in estados]}")
    print(f"Recompensa promedio: {r_es:.2f}")
    
    # Comparación final
    print("\n" + "="*70)
    print("COMPARACIÓN DE MÉTODOS")
    print("="*70)
    print(f"Política aleatoria:      R = {r_random:.2f}")
    print(f"Hill Climbing:           R = {r_hc:.2f}")
    print(f"REINFORCE:               R = {r_pg:.2f}")
    print(f"Cross-Entropy Method:    R = {r_cem:.2f}")
    print(f"Evolution Strategies:    R = {r_es:.2f}")

# ========== Modo INTERACTIVO ==========
def modo_interactivo():
    print("\nMODO INTERACTIVO: Búsqueda de Políticas")
    print("=" * 70)
    
    # Creamos el entorno
    estados, acciones, transiciones, recompensas = crear_entorno_simple()
    
    print(f"\nENTORNO:")
    print(f"Estados: {estados}")
    print(f"Acciones: {acciones} (0 o 1)")
    print(f"Política π(a|s,θ): P(a=1|s) = θ[s]")
    # Es decir, una política Bernoulli por estado, parametrizada por θ.
    
    # Seleccionamos método
    print("\nMétodos de búsqueda de políticas disponibles:")
    print("1) Hill Climbing")
    print("2) REINFORCE (Policy Gradient)")
    print("3) Cross-Entropy Method")
    print("4) Evolution Strategies")
    
    opcion = input("\nSeleccione método [1-4]: ").strip()
    
    # Ejecutamos el método seleccionado
    if opcion == '1':
        iteraciones_str = input("Número de iteraciones [ej: 100]: ").strip()
        iteraciones = int(iteraciones_str) if iteraciones_str else 100
        paso_str = input("Tamaño de paso [ej: 0.1]: ").strip()
        paso = float(paso_str) if paso_str else 0.1
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        
        print(f"\nEjecutando Hill Climbing...")
        theta, r = hill_climbing_politica(estados, transiciones, recompensas, iteraciones, paso, verbose)
        
    elif opcion == '2':
        episodios_str = input("Número de episodios [ej: 200]: ").strip()
        episodios = int(episodios_str) if episodios_str else 200
        alpha_str = input("Tasa de aprendizaje α [ej: 0.01]: ").strip()
        alpha = float(alpha_str) if alpha_str else 0.01
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        
        print(f"\nEjecutando REINFORCE...")
        theta, r, _ = reinforce_simple(estados, transiciones, recompensas, episodios, alpha, verbose)
        
    elif opcion == '3':
        generaciones_str = input("Número de generaciones [ej: 50]: ").strip()
        generaciones = int(generaciones_str) if generaciones_str else 50
        poblacion_str = input("Tamaño de población [ej: 20]: ").strip()
        poblacion = int(poblacion_str) if poblacion_str else 20
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        
        print(f"\nEjecutando Cross-Entropy Method...")
        theta, r = cross_entropy_method(estados, transiciones, recompensas, generaciones, poblacion, verbose=verbose)
        
    elif opcion == '4':
        generaciones_str = input("Número de generaciones [ej: 50]: ").strip()
        generaciones = int(generaciones_str) if generaciones_str else 50
        poblacion_str = input("Tamaño de población [ej: 15]: ").strip()
        poblacion = int(poblacion_str) if poblacion_str else 15
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        
        print(f"\nEjecutando Evolution Strategies...")
        theta, r = evolution_strategies(estados, transiciones, recompensas, generaciones, poblacion, verbose=verbose)
        
    else:
        print("Opción inválida. Usando Hill Climbing por defecto.")
        theta, r = hill_climbing_politica(estados, transiciones, recompensas, iteraciones=100, verbose=False)
    
    # Mostramos resultados
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    print(f"Política óptima encontrada:")
    for s in estados:
        print(f"  Estado {s}: P(acción=1) = {theta[s]:.3f}")
    print(f"\nRecompensa promedio por episodio: {r:.2f}")
    
    # Evaluamos política aleatoria para comparación
    theta_random = {s: 0.5 for s in estados}
    r_random = evaluar_politica(theta_random, estados, transiciones, recompensas)
    print(f"Política aleatoria (baseline):    {r_random:.2f}")
    print(f"Mejora obtenida:                  {r - r_random:.2f} ({((r/r_random - 1)*100):.1f}%)")

# ========== main ==========
def main():
    # Mostramos el menú principal
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           BÚSQUEDA DE POLÍTICAS (Policy Search)                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (compara todos los métodos)")
    print("2) Modo INTERACTIVO (configura y prueba un método)\n")
    
    # Leemos la opción del usuario
    opcion = input("Ingrese opción: ").strip()
    
    # Ejecutamos el modo correspondiente
    if opcion == '2':
        modo_interactivo()  # Modo con configuración personalizada
    else:
        modo_demo()  # Modo con comparación de todos los métodos

if __name__ == "__main__":
    main() 
