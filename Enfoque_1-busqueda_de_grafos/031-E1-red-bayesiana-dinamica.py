"""
031-E1-red_bayesiana_dinamica.py
--------------------------------
Este script implementa un modelo simplificado de Red Bayesiana Dinámica (DBN):
- Modela una variable de estado que evoluciona en el tiempo con dependencia temporal (t-1 → t).  
- Incluye modo DEMO: problema sencillo predefinido, articulando la evolución de la creencia del estado a través de filtros de tiempo.  
- Incluye modo INTERACTIVO: usuario puede seleccionar entre escenarios precargados y ver cómo cambia la creencia con el tiempo.  
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def inicializar_belief(estados, prob_inicial=None):
    """
    Inicializa la distribución de creencia para el primer instante de tiempo.
    :parametro estados: lista de etiquetas de los estados posibles
    :parametro prob_inicial: lista de probabilidades que suman 1 (opcional)
    :return: dict estado→probabilidad
    """
    if prob_inicial and abs(sum(prob_inicial) - 1.0) < 1e-8 and len(prob_inicial) == len(estados):
        return {s: p for s, p in zip(estados, prob_inicial)}
    else:
        uniforme = 1.0 / len(estados)
        return {s: uniforme for s in estados}

def actualizar_belief(belief_anterior, P, O, observacion, estados, verbose=False):
    """
    Aplica el paso de filtrado: primero predicción usando transición, luego actualización con la observación.
    belief′(s′) = η · O(s′, observacion) · Σ_{s} P(s → s′) · belief(s)
    :parametro belief_anterior: dict estado→probabilidad en t-1
    :parametro P: dict (s → s′) probabilidad de transición de t-1 a t
    :parametro O: dict (s′, observacion) probabilidad de observar 'observacion' dado s′
    :parametro observacion: etiqueta de la observación recibida en t
    :parametro estados: lista de estados
    :parametro verbose: si True, imprime el proceso paso a paso
    :return: nueva creencia dict estado→probabilidad
    """
    if verbose:
        print("\n  INICIO DEL FILTRADO BAYESIANO")
        print(f"\n  Creencia anterior (t-1): {belief_anterior}")
        print(f"  Observación recibida: {observacion}")
        print("\n  PASO 1: PREDICCIÓN (Aplicar modelo de transición P)")
    
    # Paso 1: Predicción usando el modelo de transición
    pred = {s2: 0.0 for s2 in estados}
    for s in estados:
        for s2, p_trans in P.get(s, {}).items():
            contribucion = belief_anterior[s] * p_trans
            pred[s2] += contribucion
            if verbose:
                print(f"     {s} → {s2}: P({s}→{s2})={p_trans:.3f} × belief({s})={belief_anterior[s]:.3f} = {contribucion:.4f}")
    
    if verbose:
        print(f"\n  Distribución predicha (antes de observación): {pred}")
        print("\n  PASO 2: ACTUALIZACIÓN (Incorporar observación con O)")
    
    # Paso 2: Actualización con la observación
    norm = 0.0
    belief_nueva = {}
    for s2 in estados:
        p_obs = O.get((s2, observacion), 0.0)
        belief_nueva[s2] = p_obs * pred[s2]
        norm += belief_nueva[s2]
        if verbose:
            print(f"     Estado {s2}: P(obs={observacion}|{s2})={p_obs:.3f} × pred({s2})={pred[s2]:.4f} = {belief_nueva[s2]:.4f}")
    
    if verbose:
        print(f"\n  Suma total (antes de normalizar): {norm:.4f}")
        print("\n  PASO 3: NORMALIZACIÓN (Asegurar que sume 1.0)")
    
    if norm <= 0.0:
        # repartir uniformemente en caso de degeneración
        if verbose:
            print("  ADVERTENCIA: Suma = 0, distribuyendo uniformemente")
        return {s: 1.0/len(estados) for s in estados}
    
    # Paso 3: Normalización
    belief_normalizada = {s2: belief_nueva[s2]/norm for s2 in estados}
    
    if verbose:
        for s2 in estados:
            print(f"     {s2}: {belief_nueva[s2]:.4f} / {norm:.4f} = {belief_normalizada[s2]:.4f}")
        print(f"\n  Creencia normalizada (t): {belief_normalizada}\n")
    
    return belief_normalizada

def modo_demo():
    print("\nMODO DEMO - Red Bayesiana Dinámica - Filtrado Temporal\n")
    
    estados = ['Limpio','Sucio']
    # Probabilidades de transición: del estado t-1 al estado t
    P = {
        'Limpio': {'Limpio': 0.7, 'Sucio': 0.3},
        'Sucio':  {'Limpio': 0.4, 'Sucio': 0.6}
    }
    # Probabilidades de observación: dado estado actual, probabilidad de ver 'Brillante' o 'Opaco'
    O = {
        ('Limpio','VerBrillante'): 0.8,
        ('Limpio','VerOpaco'):      0.2,
        ('Sucio','VerBrillante'):  0.3,
        ('Sucio','VerOpaco'):       0.7
    }
    
    print("CONFIGURACIÓN DEL MODELO:")
    print(f"Estados posibles: {estados}")
    print("\nModelo de Transición P(s'|s):")
    for s in estados:
        print(f"  {s} → {P[s]}")
    print("\nModelo de Observación P(obs|s'):")
    for key, val in O.items():
        print(f"  P({key[1]}|{key[0]}) = {val}")
    
    belief = inicializar_belief(estados)
    print(f"\nCreencia inicial (t=0): {belief}")
    
    # Supongamos observaciones secuenciales
    secuencia_obs = ['VerBrillante', 'VerOpaco', 'VerBrillante']
    for t, obs in enumerate(secuencia_obs, start=1):
        print(f"\nTIEMPO t = {t}")
        belief = actualizar_belief(belief, P, O, obs, estados, verbose=True)

    print("\nSimulación DEMO completada.")
    print(f"Creencia final: {belief}")

def modo_interactivo():
    print("\n" + "="*70)
    estados = ['Limpio','Sucio']
    # Probabilidades de transición: del estado t-1 al estado t
    P = {
        'Limpio': {'Limpio': 0.7, 'Sucio': 0.3},
        'Sucio':  {'Limpio': 0.4, 'Sucio': 0.6}
    }
    # Probabilidades de observación: dado estado actual, probabilidad de ver ‘Brillante’ o ‘Opaco’
    O = {
        ('Limpio','VerBrillante'): 0.8,
        ('Limpio','VerOpaco'):      0.2,
        ('Sucio','VerBrillante'):  0.3,
        ('Sucio','VerOpaco'):       0.7
    }
    belief = inicializar_belief(estados)
    print("Creencia inicial:", belief)
    # Supongamos observaciones secuenciales
    secuencia_obs = ['VerBrillante', 'VerOpaco', 'VerBrillante']
    for t, obs in enumerate(secuencia_obs, start=1):
        print(f"\nTiempo t = {t}: Observación recibida = {obs}")
        belief = actualizar_belief(belief, P, O, obs, estados)
        print("  Creencia actualizada:", belief)

    print("\nSimulación DEMO completada.")

def modo_interactivo():
    print("\nMODO INTERACTIVO - Red Bayesiana Dinámica - Filtrado Temporal\n")
    print("Escenarios predefinidos:")
    print("1) Estado limpieza (Limpio/Sucio)")
    print("2) Escenario vigilancia (Normal/Intruso)")
    opcion = input("\nIntroduce el número de escenario: ").strip()
    
    if opcion == '2':
        estados = ['Normal','Intruso']
        P = {
            'Normal': {'Normal':0.85, 'Intruso':0.15},
            'Intruso':{'Normal':0.25, 'Intruso':0.75}
        }
        O = {
            ('Normal','VerOk'): 0.9,
            ('Normal','VerAlerta'):0.1,
            ('Intruso','VerOk'):   0.2,
            ('Intruso','VerAlerta'):0.8
        }
        print("\nHas elegido escenario vigilancia.")
    else:
        estados = ['Limpio','Sucio']
        P = {
            'Limpio': {'Limpio': 0.7, 'Sucio': 0.3},
            'Sucio':  {'Limpio': 0.4, 'Sucio': 0.6}
        }
        O = {
            ('Limpio','VerBrillante'):0.8,
            ('Limpio','VerOpaco'):     0.2,
            ('Sucio','VerBrillante'): 0.3,
            ('Sucio','VerOpaco'):      0.7
        }
        print("\nHas elegido escenario limpieza.")

    print("\nCONFIGURACIÓN DEL MODELO:")
    print(f"Estados posibles: {estados}")
    print("\nModelo de Transición P(s'|s):")
    for s in estados:
        print(f"  {s} → {P[s]}")
    print("\nModelo de Observación P(obs|s'):")
    for key, val in O.items():
        print(f"  P({key[1]}|{key[0]}) = {val}")

    belief = inicializar_belief(estados)
    print(f"\nCreencia inicial (t=0): {belief}")
    
    pasos = int(input("\n¿Cuántos pasos deseas simular? ").strip())
    verbose_input = input("¿Mostrar proceso detallado paso a paso? (s/n): ").strip().lower()
    verbose = verbose_input == 's'
    
    import random
    # Inicializar estado real oculto
    estado_real = random.choices(estados, weights=[belief[s] for s in estados])[0]
    
    for t in range(1, pasos+1):
        print(f"\nTIEMPO t = {t}")
        
        # Simular transición del estado real
        trans = P.get(estado_real, {})
        if trans:
            estados_sig = list(trans.keys())
            probs_sig = list(trans.values())
            estado_siguiente = random.choices(estados_sig, weights=probs_sig)[0]
        else:
            estado_siguiente = estado_real
        
        # Generar observación según el estado real
        possibles = {obs: prob for (s_, obs), prob in O.items() if s_ == estado_siguiente}
        obs_list, probs = zip(*possibles.items())
        observacion = random.choices(obs_list, weights=probs)[0]
        
        print(f"(Mundo real) Estado: {estado_real} → {estado_siguiente}")
        print(f"Observación generada: {observacion}")
        
        # Actualizar creencia del agente
        belief = actualizar_belief(belief, P, O, observacion, estados, verbose=verbose)
        
        if not verbose:
            print(f"Creencia actualizada: {belief}")
        
        estado_real = estado_siguiente

    print("\nSimulación interactiva completada.")
    print(f"Creencia final: {belief}")

def main():
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    opcion = input("Ingrese opción: ").strip()
    if opcion == '2':
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
