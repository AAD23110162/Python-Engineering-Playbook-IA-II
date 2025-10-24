"""
030-E1-POMDP.py
-----------------------------
Este script implementa un modelo simplificado de un Proceso de Decisión de Márkov Parcialmente Observable (POMDP):
- Define estados ocultos, acciones, modelo de transición, modelo de observación, recompensas.
- El agente mantiene una **creencia** (distribución de probabilidad sobre los estados) y la actualiza tras cada acción-observación. :contentReference[oaicite:0]{index=0}
- Incluye dos modos de ejecución:
    1. MODO DEMO: con un problema predefinido sencillo.
    2. MODO INTERACTIVO: el usuario selecciona entre escenarios precargados y el sistema genera automáticamente las observaciones; el usuario define el número de pasos.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def inicializar_creencia(estados, prob_inicial=None):
    """
    Crea un estado de creencia inicial (distribución uniforme o especificada).
    :parametro estados: lista de etiquetas de estados
    :parametro prob_inicial: lista de probabilidades que suman 1 o None
    :return: dict estado -> probabilidad
    """
    if prob_inicial and len(prob_inicial) == len(estados) and abs(sum(prob_inicial) - 1.0) < 1e-8:
        return {s: p for s, p in zip(estados, prob_inicial)}
    else:
        uniforme = 1.0 / len(estados)
        return {s: uniforme for s in estados}

def actualizar_creencia(creencia_anterior, accion, observacion, P, O, estados):
    """
    Actualiza la distribución de creencia tras tomar acción y recibir una observación.
    b'(s') = η · O(s', accion, observacion) · Σ_s P(s → s' | accion) · b(s)
    :parametro creencia_anterior: dict estado -> probabilidad
    :parametro accion: acción ejecutada
    :parametro observacion: observación recibida
    :parametro P: dict (s, accion) -> dict s' -> probabilidad de transición
    :parametro O: dict (s', accion, observacion) -> probabilidad de observar 'observacion'
    :parametro estados: lista de estados
    :return: nueva creencia dict estado -> probabilidad (normalizada)
    """
    nueva = {}
    factor = 0.0
    for s2 in estados:
        suma_trans = 0.0
        for s in estados:
            suma_trans += creencia_anterior[s] * P.get((s, accion), {}).get(s2, 0.0)
        prob_obs = O.get((s2, accion, observacion), 0.0)
        nueva[s2] = prob_obs * suma_trans
        factor += nueva[s2]
    if factor <= 0.0:
        # Si factor es 0 o extremadamente pequeño, repartir uniformemente
        return {s: 1.0/len(estados) for s in estados}
    # Normalizar
    return {s: nueva[s]/factor for s in estados}

def calcular_valor_creencia(creencia, acciones, P, R, gamma, estados):
    """
    Simplificación: calcula una estimación del valor de cada acción dadas la creencia actual.
    Luego selecciona la mejor acción (modelo miope: solo recompensa inmediata).
    :parametro creencia: dict estado -> probabilidad
    :parametro acciones: lista de acciones posibles
    :parametro P: dict de transición como antes
    :parametro R: dict (s, accion) -> recompensa inmediata
    :parametro gamma: factor de descuento (no se usa profundiad en esta simplificación)
    :parametro estados: lista de estados
    :return: (acción seleccionada, valor estimado)
    """
    mejor_accion = None
    mejor_valor = float('-inf')
    for a in acciones:
        valor_a = 0.0
        for s in estados:
            valor_s = R.get((s, a), 0.0)
            valor_a += creencia[s] * valor_s
        if valor_a > mejor_valor:
            mejor_valor = valor_a
            mejor_accion = a
    return mejor_accion, mejor_valor

def modo_demo():
    print("\n--- MODO DEMO ---")
    estados = ['Bueno', 'Malo']
    acciones = ['Inspeccionar', 'Reparar']
    P = {
        ('Bueno','Inspeccionar'): {'Bueno':0.9, 'Malo':0.1},
        ('Bueno','Reparar'):      {'Bueno':0.6, 'Malo':0.4},
        ('Malo','Inspeccionar'): {'Bueno':0.2, 'Malo':0.8},
        ('Malo','Reparar'):      {'Bueno':0.3, 'Malo':0.7},
    }
    O = {
        ('Bueno','Inspeccionar','VerBueno'):   0.8,
        ('Bueno','Inspeccionar','VerMalo'):    0.2,
        ('Malo','Inspeccionar','VerBueno'):    0.3,
        ('Malo','Inspeccionar','VerMalo'):     0.7,
        ('Bueno','Reparar','VerBueno'):        1.0,
        ('Malo','Reparar','VerBueno'):         0.5,
    }
    R = {
        ('Bueno','Inspeccionar'): 1,
        ('Bueno','Reparar'):      5,
        ('Malo','Inspeccionar'): -1,
        ('Malo','Reparar'):      2,
    }
    gamma = 0.9

    creencia = inicializar_creencia(estados)
    print("Creencia inicial:", creencia)

    for paso in range(3):
        print(f"\nPaso {paso+1}:")
        print("  Creencia actual:", creencia)
        accion, valor = calcular_valor_creencia(creencia, acciones, P, R, gamma, estados)
        print(f"  → Acción elegida: {accion} (valor estimado = {valor:.3f})")
        # En demo escogemos una observación arbitraria para ilustración
        if accion == 'Inspeccionar':
            observacion = 'VerBueno'
        else:
            observacion = 'VerBueno'
        print(f"  → Observación recibida (demo): {observacion}")
        creencia = actualizar_creencia(creencia, accion, observacion, P, O, estados)

    print("\nSimulación DEMO completada. Creencia final:", creencia)

def modo_interactivo():
    print("\n--- MODO INTERACTIVO: Escenario POMDP sencillo ---")
    print("Elige un escenario previsto:")
    print(" 1) Máquina (estados Bueno/Malo)")
    print(" 2) Seguridad doméstica (estados Normal/Intrusión)")
    opcion = input("Introduce el número del escenario (1 o 2): ").strip()

    if opcion == '2':
        estados = ['Normal', 'Intrusion']
        acciones = ['Observar', 'Alerta']
        P = {
            ('Normal','Observar'):   {'Normal':0.95, 'Intrusion':0.05},
            ('Normal','Alerta'):     {'Normal':0.90, 'Intrusion':0.10},
            ('Intrusion','Observar'):{'Normal':0.10, 'Intrusion':0.90},
            ('Intrusion','Alerta'):  {'Normal':0.20, 'Intrusion':0.80},
        }
        O = {
            ('Normal','Observar','VerNormal'):   0.9,
            ('Normal','Observar','VerIntrusion'):0.1,
            ('Intrusion','Observar','VerNormal'):0.3,
            ('Intrusion','Observar','VerIntrusion'):0.7,
            ('Normal','Alerta','VerNormal'):    1.0,
            ('Intrusion','Alerta','VerNormal'):  0.5,
        }
        R = {
            ('Normal','Observar'):   0,
            ('Normal','Alerta'):    -5,
            ('Intrusion','Observar'):-10,
            ('Intrusion','Alerta'):  20,
        }
        print("Has elegido escenario de seguridad doméstica.")
    else:
        estados = ['Bueno', 'Malo']
        acciones = ['Inspeccionar', 'Reparar']
        P = {
            ('Bueno','Inspeccionar'): {'Bueno':0.9, 'Malo':0.1},
            ('Bueno','Reparar'):      {'Bueno':0.6, 'Malo':0.4},
            ('Malo','Inspeccionar'): {'Bueno':0.2, 'Malo':0.8},
            ('Malo','Reparar'):      {'Bueno':0.3, 'Malo':0.7},
        }
        O = {
            ('Bueno','Inspeccionar','VerBueno'):   0.8,
            ('Bueno','Inspeccionar','VerMalo'):    0.2,
            ('Malo','Inspeccionar','VerBueno'):    0.3,
            ('Malo','Inspeccionar','VerMalo'):     0.7,
            ('Bueno','Reparar','VerBueno'):        1.0,
            ('Malo','Reparar','VerBueno'):         0.5,
        }
        R = {
            ('Bueno','Inspeccionar'): 1,
            ('Bueno','Reparar'):      5,
            ('Malo','Inspeccionar'): -1,
            ('Malo','Reparar'):      2,
        }
        print("Has elegido escenario de la máquina (Bueno/Malo).")

    gamma = float(input("Introduce el factor de descuento γ (entre 0 y 1): ").strip())
    creencia = inicializar_creencia(estados)
    print(f"Creencia inicial: {creencia}")
    pasos = int(input("¿Cuántos pasos deseas simular? ").strip())

    # Inicializa un estado real oculto muestreando desde la creencia actual
    import random
    estado_real = random.choices(estados, weights=[creencia[s] for s in estados])[0]

    for paso in range(pasos):
        print(f"\nPaso {paso+1}:")
        print("  Creencia actual:", creencia)
        accion, valor = calcular_valor_creencia(creencia, acciones, P, R, gamma, estados)
        print(f"  → Acción sugerida: {accion} (valor estimado = {valor:.3f})")

        # Muestrear la transición real s -> s' según P(estado_real, accion)
        trans = P.get((estado_real, accion), {})
        if trans:
            estados_siguiente = list(trans.keys())
            probs_siguiente = list(trans.values())
            estado_siguiente = random.choices(estados_siguiente, weights=probs_siguiente)[0]
        else:
            estado_siguiente = estado_real  # sin modelo de transición, permanece

        # Elegir observación automáticamente según O[(estado_siguiente, accion, obs)]
        posibles_obs = {obs: prob for (s_, a_, obs), prob in O.items() if s_ == estado_siguiente and a_ == accion}
        if posibles_obs:
            obs_list, obs_probs = zip(*posibles_obs.items())
            observacion = random.choices(obs_list, weights=obs_probs)[0]
        else:
            # Fallback: usar la primera observación definida para esa acción, si existe
            obs_candidates = [obs for (s_, a_, obs) in O.keys() if a_ == accion]
            observacion = obs_candidates[0] if obs_candidates else (next(iter(O)))[2]

        print(f"  (mundo) Estado real: {estado_real} → {estado_siguiente} | Observación: {observacion}")

        print(f"  → Observación recibida automáticamente: {observacion}")
        creencia = actualizar_creencia(creencia, accion, observacion, P, O, estados)
        estado_real = estado_siguiente

    print("\nSimulación interactiva completada. Creencia final:", creencia)


def main():
    print("Seleccione modo de ejecución:")
    print(" 1) Modo DEMO")
    print(" 2) Modo INTERACTIVO\n")
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
