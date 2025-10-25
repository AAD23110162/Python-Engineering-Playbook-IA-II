"""
036-E1-exploracion_vs_explotacion.py
-------------------------------------
Este script implementa y compara diferentes estrategias de exploración vs explotación
en el contexto del problema de Multi-Armed Bandits (MAB) y Aprendizaje por Refuerzo.

Estrategias implementadas:
1. ε-greedy: Explora con probabilidad ε, explota con probabilidad 1-ε
2. ε-decreciente: ε disminuye con el tiempo
3. UCB (Upper Confidence Bound): Balance usando incertidumbre
4. Softmax/Boltzmann: Selección probabilística basada en temperatura

Características:
- Problema de bandidos multi-brazo (k brazos con recompensas estocásticas)
- Comparación de estrategias en términos de recompensa acumulada
- Modo DEMO y modo INTERACTIVO
- Opción verbose para ver el proceso paso a paso

Autor: Alejandro Aguirre Díaz
"""

import random
import math

# ========== Funciones utilitarias ==========
def crear_bandidos(k, medias=None, varianzas=None):
    """
    Crea k bandidos (brazos) con distribuciones normales.
    Si no se especifican medias/varianzas, se generan aleatoriamente.
    """
    if medias is None:
        # Generamos medias aleatorias entre 0 y 10
        medias = [random.uniform(0, 10) for _ in range(k)]
    if varianzas is None:
        # Usamos varianza fija de 1.0 para todos
        varianzas = [1.0] * k
    
    # Cada bandido devuelve una función que genera recompensas
    bandidos = []
    for i in range(k):
        # Creamos una función lambda que captura media y varianza
        bandido = lambda mu=medias[i], sigma=math.sqrt(varianzas[i]): random.gauss(mu, sigma)
        bandidos.append(bandido)
    
    return bandidos, medias

def obtener_recompensa(bandidos, brazo):
    """
    Obtiene una recompensa del brazo especificado.
    """
    # El brazo devuelve una recompensa según su distribución
    return bandidos[brazo]()

# ========== Estrategia 1: ε-greedy ==========
def epsilon_greedy(bandidos, epsilon, pasos, verbose=False):
    """
    Estrategia ε-greedy: explora con probabilidad ε, explota con probabilidad 1-ε.
    """
    k = len(bandidos)  # Número de brazos
    Q = [0.0] * k      # Estimaciones de valor para cada brazo
    N = [0] * k        # Número de veces que se ha elegido cada brazo
    recompensa_total = 0.0
    historial_recompensas = []
    
    for t in range(pasos):
        # Decidimos si exploramos o explotamos
        if random.random() < epsilon:
            # Exploración: elegimos un brazo al azar
            brazo = random.randint(0, k - 1)
        else:
            # Explotación: elegimos el brazo con mayor Q estimado
            brazo = Q.index(max(Q))
        
        # Obtenemos la recompensa del brazo elegido
        r = obtener_recompensa(bandidos, brazo)
        
        # Actualizamos el contador del brazo
        N[brazo] += 1
        
        # Actualizamos la estimación Q usando promedio incremental
        # Q_nuevo = Q_anterior + (1/N) * (r - Q_anterior)
        Q[brazo] += (1.0 / N[brazo]) * (r - Q[brazo])
        
        # Acumulamos la recompensa total
        recompensa_total += r
        historial_recompensas.append(recompensa_total)
        
        # Si verbose, mostramos el proceso paso a paso
        if verbose and (t < 10 or t % 100 == 0):
            print(f"Paso {t+1}: brazo={brazo}, r={r:.2f}, Q={[round(q,2) for q in Q]}, N={N}")
    
    return recompensa_total, historial_recompensas, Q, N

# ========== Estrategia 2: ε-decreciente ==========
def epsilon_decreciente(bandidos, epsilon_inicial, pasos, verbose=False):
    """
    Estrategia ε-decreciente: ε disminuye con el tiempo según 1/t.
    """
    k = len(bandidos)
    Q = [0.0] * k
    N = [0] * k
    recompensa_total = 0.0
    historial_recompensas = []
    
    for t in range(pasos):
        # Calculamos epsilon decreciente: ε(t) = ε_0 / (1 + t)
        epsilon = epsilon_inicial / (1.0 + t / 100.0)
        
        # Decidimos si exploramos o explotamos
        if random.random() < epsilon:
            brazo = random.randint(0, k - 1)
        else:
            brazo = Q.index(max(Q))
        
        # Obtenemos recompensa y actualizamos
        r = obtener_recompensa(bandidos, brazo)
        N[brazo] += 1
        Q[brazo] += (1.0 / N[brazo]) * (r - Q[brazo])
        recompensa_total += r
        historial_recompensas.append(recompensa_total)
        
        if verbose and (t < 10 or t % 100 == 0):
            print(f"Paso {t+1}: ε={epsilon:.4f}, brazo={brazo}, r={r:.2f}, Q={[round(q,2) for q in Q]}")
    
    return recompensa_total, historial_recompensas, Q, N

# ========== Estrategia 3: UCB (Upper Confidence Bound) ==========
def ucb(bandidos, c, pasos, verbose=False):
    """
    Estrategia UCB: selecciona el brazo con mayor valor UCB.
    UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
    """
    k = len(bandidos)
    Q = [0.0] * k
    N = [0] * k
    recompensa_total = 0.0
    historial_recompensas = []
    
    # Primero, jugamos cada brazo una vez para inicializar
    for brazo in range(k):
        r = obtener_recompensa(bandidos, brazo)
        N[brazo] = 1
        Q[brazo] = r
        recompensa_total += r
        historial_recompensas.append(recompensa_total)
        
        if verbose:
            print(f"Inicialización brazo {brazo}: r={r:.2f}")
    
    # Ahora aplicamos UCB para los pasos restantes
    for t in range(k, pasos):
        # Calculamos UCB para cada brazo
        ucb_values = []
        for a in range(k):
            # UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
            bonus = c * math.sqrt(math.log(t + 1) / N[a])
            ucb_val = Q[a] + bonus
            ucb_values.append(ucb_val)
        
        # Elegimos el brazo con mayor UCB
        brazo = ucb_values.index(max(ucb_values))
        
        # Obtenemos recompensa y actualizamos
        r = obtener_recompensa(bandidos, brazo)
        N[brazo] += 1
        Q[brazo] += (1.0 / N[brazo]) * (r - Q[brazo])
        recompensa_total += r
        historial_recompensas.append(recompensa_total)
        
        if verbose and (t < 10 or t % 100 == 0):
            print(f"Paso {t+1}: brazo={brazo}, UCB={[round(u,2) for u in ucb_values]}, r={r:.2f}")
    
    return recompensa_total, historial_recompensas, Q, N

# ========== Estrategia 4: Softmax/Boltzmann ==========
def softmax_boltzmann(bandidos, temperatura, pasos, verbose=False):
    """
    Estrategia Softmax/Boltzmann: selección probabilística basada en temperatura.
    P(a) = exp(Q(a)/τ) / Σ_b exp(Q(b)/τ)
    """
    k = len(bandidos)
    Q = [0.0] * k
    N = [0] * k
    recompensa_total = 0.0
    historial_recompensas = []
    
    for t in range(pasos):
        # Calculamos probabilidades softmax
        if temperatura > 0:
            # Calculamos exp(Q(a)/τ) para cada brazo
            exp_values = [math.exp(q / temperatura) for q in Q]
            suma_exp = sum(exp_values)
            
            # Calculamos probabilidades
            if suma_exp > 0:
                probs = [e / suma_exp for e in exp_values]
            else:
                # Si suma es 0, probabilidad uniforme
                probs = [1.0 / k] * k
        else:
            # Temperatura 0: siempre explota (greedy puro)
            max_q = max(Q)
            probs = [1.0 if q == max_q else 0.0 for q in Q]
            # Normalizamos en caso de empates
            suma_probs = sum(probs)
            if suma_probs > 0:
                probs = [p / suma_probs for p in probs]
        
        # Seleccionamos un brazo según las probabilidades
        brazo = random.choices(range(k), weights=probs)[0]
        
        # Obtenemos recompensa y actualizamos
        r = obtener_recompensa(bandidos, brazo)
        N[brazo] += 1
        Q[brazo] += (1.0 / N[brazo]) * (r - Q[brazo])
        recompensa_total += r
        historial_recompensas.append(recompensa_total)
        
        if verbose and (t < 10 or t % 100 == 0):
            print(f"Paso {t+1}: brazo={brazo}, P={[round(p,3) for p in probs]}, r={r:.2f}")
    
    return recompensa_total, historial_recompensas, Q, N

# ========== Modo DEMO ==========
def modo_demo():
    print("\nMODO DEMO: Exploración vs Explotación")
    print("=" * 60)
    
    # Configuramos el problema de bandidos
    k = 5  # Número de brazos
    medias = [3.0, 5.0, 2.0, 7.0, 4.0]  # Medias reales (desconocidas para el agente)
    pasos = 1000  # Número de pasos
    
    print(f"\nPROBLEMA: {k} bandidos con medias reales: {medias}")
    print(f"Mejor brazo: {medias.index(max(medias))} (media={max(medias)})")
    print(f"Pasos: {pasos}\n")
    
    # Creamos los bandidos (misma semilla para comparación justa)
    random.seed(42)
    
    # Estrategia 1: ε-greedy
    print("\n" + "="*60)
    print("Estrategia 1: ε-greedy (ε=0.1)")
    print("="*60)
    bandidos, _ = crear_bandidos(k, medias)
    r_total, _, Q, N = epsilon_greedy(bandidos, epsilon=0.1, pasos=pasos, verbose=False)
    print(f"Recompensa total: {r_total:.2f}")
    print(f"Q estimados: {[round(q,2) for q in Q]}")
    print(f"Veces elegido cada brazo: {N}")
    
    # Estrategia 2: ε-decreciente
    print("\n" + "="*60)
    print("Estrategia 2: ε-decreciente (ε₀=1.0)")
    print("="*60)
    random.seed(42)
    bandidos, _ = crear_bandidos(k, medias)
    r_total, _, Q, N = epsilon_decreciente(bandidos, epsilon_inicial=1.0, pasos=pasos, verbose=False)
    print(f"Recompensa total: {r_total:.2f}")
    print(f"Q estimados: {[round(q,2) for q in Q]}")
    print(f"Veces elegido cada brazo: {N}")
    
    # Estrategia 3: UCB
    print("\n" + "="*60)
    print("Estrategia 3: UCB (c=2.0)")
    print("="*60)
    random.seed(42)
    bandidos, _ = crear_bandidos(k, medias)
    r_total, _, Q, N = ucb(bandidos, c=2.0, pasos=pasos, verbose=False)
    print(f"Recompensa total: {r_total:.2f}")
    print(f"Q estimados: {[round(q,2) for q in Q]}")
    print(f"Veces elegido cada brazo: {N}")
    
    # Estrategia 4: Softmax
    print("\n" + "="*60)
    print("Estrategia 4: Softmax/Boltzmann (τ=1.0)")
    print("="*60)
    random.seed(42)
    bandidos, _ = crear_bandidos(k, medias)
    r_total, _, Q, N = softmax_boltzmann(bandidos, temperatura=1.0, pasos=pasos, verbose=False)
    print(f"Recompensa total: {r_total:.2f}")
    print(f"Q estimados: {[round(q,2) for q in Q]}")
    print(f"Veces elegido cada brazo: {N}")

# ========== Modo INTERACTIVO ==========
def modo_interactivo():
    print("\nMODO INTERACTIVO: Exploración vs Explotación")
    print("=" * 60)
    
    # Solicitamos configuración del problema
    try:
        k = int(input("\nNúmero de bandidos (brazos) [ej: 5]: ").strip())
        pasos = int(input("Número de pasos [ej: 1000]: ").strip())
        usar_medias = input("¿Definir medias manualmente? (s/n) [n]: ").strip().lower()
        
        if usar_medias == 's':
            medias = []
            for i in range(k):
                media = float(input(f"Media del brazo {i} [ej: 5.0]: ").strip())
                medias.append(media)
        else:
            medias = None  # Se generarán aleatoriamente
    except Exception:
        print("Parámetros inválidos. Usando valores por defecto.")
        k, pasos, medias = 5, 1000, [3.0, 5.0, 2.0, 7.0, 4.0]
    
    # Creamos los bandidos
    bandidos, medias_reales = crear_bandidos(k, medias)
    
    print(f"\nPROBLEMA: {k} bandidos con medias: {[round(m,2) for m in medias_reales]}")
    print(f"Mejor brazo: {medias_reales.index(max(medias_reales))} (media={max(medias_reales):.2f})")
    
    # Seleccionamos estrategia
    print("\nEstrategias disponibles:")
    print("1) ε-greedy")
    print("2) ε-decreciente")
    print("3) UCB (Upper Confidence Bound)")
    print("4) Softmax/Boltzmann")
    
    opcion = input("\nSeleccione estrategia [1-4]: ").strip()
    
    # Ejecutamos la estrategia seleccionada
    if opcion == '1':
        epsilon_str = input("Valor de ε [0-1, ej: 0.1]: ").strip()
        epsilon = float(epsilon_str) if epsilon_str else 0.1
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        print(f"\nEjecutando ε-greedy con ε={epsilon}...")
        r_total, _, Q, N = epsilon_greedy(bandidos, epsilon, pasos, verbose)
    elif opcion == '2':
        epsilon_0_str = input("Valor inicial de ε [ej: 1.0]: ").strip()
        epsilon_0 = float(epsilon_0_str) if epsilon_0_str else 1.0
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        print(f"\nEjecutando ε-decreciente con ε₀={epsilon_0}...")
        r_total, _, Q, N = epsilon_decreciente(bandidos, epsilon_0, pasos, verbose)
    elif opcion == '3':
        c_str = input("Parámetro c [ej: 2.0]: ").strip()
        c = float(c_str) if c_str else 2.0
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        print(f"\nEjecutando UCB con c={c}...")
        r_total, _, Q, N = ucb(bandidos, c, pasos, verbose)
    elif opcion == '4':
        temp_str = input("Temperatura τ [ej: 1.0]: ").strip()
        temp = float(temp_str) if temp_str else 1.0
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        print(f"\nEjecutando Softmax con τ={temp}...")
        r_total, _, Q, N = softmax_boltzmann(bandidos, temp, pasos, verbose)
    else:
        print("Opción inválida. Usando ε-greedy por defecto.")
        verbose = input("¿Mostrar proceso paso a paso? (s/n) [n]: ").strip().lower() == 's'
        r_total, _, Q, N = epsilon_greedy(bandidos, 0.1, pasos, verbose)
    
    # Mostramos resultados
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"Recompensa total obtenida: {r_total:.2f}")
    print(f"Recompensa promedio por paso: {r_total/pasos:.2f}")
    print(f"\nValores Q estimados: {[round(q,2) for q in Q]}")
    print(f"Medias reales: {[round(m,2) for m in medias_reales]}")
    print(f"\nVeces que se eligió cada brazo: {N}")
    print(f"Brazo más elegido: {N.index(max(N))}")
    print(f"Brazo óptimo real: {medias_reales.index(max(medias_reales))}")

# ========== main ==========
def main():
    # Mostramos el menú principal
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     EXPLORACIÓN vs EXPLOTACIÓN - Multi-Armed Bandits       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (compara todas las estrategias)")
    print("2) Modo INTERACTIVO (configura y prueba una estrategia)\n")
    
    # Leemos la opción del usuario
    opcion = input("Ingrese opción: ").strip()
    
    # Ejecutamos el modo correspondiente
    if opcion == '2':
        modo_interactivo()  # Modo con configuración personalizada
    else:
        modo_demo()  # Modo con comparación de todas las estrategias

if __name__ == "__main__":
    main()
