"""
029-E1-proceso_decision_markov.py
---------------------
Este script implementa un modelo simplificado de un Proceso de Decisión de Márkov (MDP):
- Define un conjunto de estados, acciones, probabilidades de transición y recompensas.
- Permite al usuario (o en modo demo) especificar un MDP y calcular el valor esperado de una política sencilla.
- Incluye dos modos de ejecución:
    1. MODO DEMO: escenario predefinido.
    2. MODO INTERACTIVO SIMPLIFICADO: datasets precargados + opción personalizada.
- Variables y funciones en español.

Un MDP se define por:
  - Estados (S): conjunto de situaciones posibles del sistema
  - Acciones (A): opciones disponibles en cada estado
  - Probabilidades de transición P(s,a,s'): probabilidad de ir de s a s' con acción a
  - Recompensas R(s,a,s'): beneficio/costo inmediato de cada transición
  - Factor de descuento γ: importancia de recompensas futuras (0 < γ < 1)

Autor: Alejandro Aguirre Díaz
"""

def valor_de_politica(estado_inicial, estados, acciones, P, R, politica, gamma=0.9, iteraciones=50):
    """
    Calcula el valor V(s) de ejecutar la política dada desde estado_inicial,
    iterando hasta aproximar la solución.
    :parametro estado_inicial: etiqueta del estado donde comenzamos
    :parametro estados: lista de etiquetas de estados
    :parametro acciones: lista de acciones posibles
    :parametro P: dict de la forma P[(s,a)][s2] = probabilidad de s→s2 al aplicar acción a
    :parametro R: dict de la forma R[(s,a,s2)] = recompensa inmediata por transición
    :parametro politica: dict de la forma politica[s] = acción a tomar en estado s
    :parametro gamma: factor de descuento (0 ≤ γ < 1)
    :parametro iteraciones: número de iteraciones a realizar
    :return: valor aproximado V(estado_inicial)
    """
    # Inicializa todos los valores de estado a 0.0
    V = {s: 0.0 for s in estados}
    
    # Itera para aproximar el valor de la política
    for i in range(1, iteraciones+1):
        # Copia los valores actuales para actualización
        V_nueva = V.copy()
        
        # Para cada estado, calcula su nuevo valor bajo la política
        for s in estados:
            # Obtiene la acción que la política prescribe para este estado
            a = politica.get(s, acciones[0])
            suma = 0.0
            
            # Suma sobre todos los posibles estados siguientes
            for s2, p in P.get((s,a), {}).items():
                # Obtiene la recompensa de la transición
                r = R.get((s,a,s2), 0.0)
                # Acumula: probabilidad × (recompensa + valor descontado del siguiente estado)
                suma += p * (r + gamma * V[s2])
            
            # Actualiza el valor del estado
            V_nueva[s] = suma
        
        # Actualiza V para la siguiente iteración
        V = V_nueva
    
    # Retorna el valor del estado inicial
    return V[estado_inicial]

def modo_demo():
    """
    Ejecuta un ejemplo predefinido de MDP con 3 estados y 2 acciones.
    Muestra cómo calcular el valor de una política específica.
    """
    print("\n--- MODO DEMO ---")
    # Define un MDP de ejemplo con 3 estados y 2 acciones
    estados = ['s0','s1','s2']
    acciones = ['a0','a1']
    
    # Probabilidades de transición P(s,a→s')
    # Cada entrada indica: desde estado s, con acción a, hacia estado s' con probabilidad p
    P = {
        ('s0','a0'): {'s0': 0.5, 's1': 0.5},
        ('s0','a1'): {'s1': 1.0},
        ('s1','a0'): {'s2': 1.0},
        ('s1','a1'): {'s0': 0.3, 's2': 0.7},
        ('s2','a0'): {'s2': 1.0},
        ('s2','a1'): {'s0': 1.0},
    }
    
    # Recompensas R(s,a,s')
    # Cada entrada indica: la recompensa recibida al hacer transición (s,a) → s'
    R = {
        ('s0','a0','s0'): 1,   ('s0','a0','s1'): 0,
        ('s0','a1','s1'): 5,
        ('s1','a0','s2'): 10,
        ('s1','a1','s0'): -1,  ('s1','a1','s2'): 2,
        ('s2','a0','s2'): 0,
        ('s2','a1','s0'): 0,
    }
    
    # Define una política específica (acción a tomar en cada estado)
    # Una política π(s) = a indica qué acción tomar en cada estado
    politica = {
        's0': 'a0',  # En estado s0, elegimos acción a0
        's1': 'a1',  # En estado s1, elegimos acción a1
        's2': 'a0'   # En estado s2, elegimos acción a0
    }
    
    # Factor de descuento (gamma): determina importancia de recompensas futuras
    # γ cercano a 1: valora mucho el futuro; cercano a 0: solo importa recompensa inmediata
    gamma = 0.9
    
    print("Estados:", estados)
    print("Acciones:", acciones)
    print("Política elegida:", politica)
    
    # Calcula el valor de la política desde el estado inicial
    valor = valor_de_politica(estados[0], estados, acciones, P, R, politica, gamma)
    print(f"\nValor estimado de iniciar en estado {estados[0]} con la política dada: V ≈ {valor:.4f}")

def modo_interactivo():
    """
    Modo interactivo simplificado con datasets precargados.
    Permite elegir entre:
      1) Dataset pequeño de ejemplo
      2) Dataset de mantenimiento de máquina
      3) Definir un MDP personalizado manualmente
    """
    print("\n--- MODO INTERACTIVO SIMPLIFICADO ---")
    print("Seleccione un dataset predefinido o elija personalizar uno:\n")
    print("1) Dataset 1 – MDP (3 estados / 2 acciones)")
    print("2) Dataset 2 – Mantenimiento de máquina (4 estados / 2 acciones)")
    print("3) Personalizar problema MDP")
    
    opcion = input("\nIntroduce el número de opción: ").strip()

    # ========== PASO 1: Selección del dataset ==========
    # Dataset 1: MDP pequeño (similar al demo)
    if opcion == '1':
        # Estados y acciones del sistema
        estados = ['s0','s1','s2']
        acciones = ['a0','a1']
        P = {
            ('s0','a0'): {'s0':0.5, 's1':0.5},
            ('s0','a1'): {'s1':1.0},
            ('s1','a0'): {'s2':1.0},
            ('s1','a1'): {'s0':0.3,'s2':0.7},
            ('s2','a0'): {'s2':1.0},
            ('s2','a1'): {'s0':1.0},
        }
        R = {
            ('s0','a0','s0'): 1,   ('s0','a0','s1'): 0,
            ('s0','a1','s1'): 5,
            ('s1','a0','s2'): 10,
            ('s1','a1','s0'):-1,  ('s1','a1','s2'):2,
            ('s2','a0','s2'):0,
            ('s2','a1','s0'):0,
        }
        print("\n[Dataset 1] MDP de 3 estados cargado exitosamente.")
    
    # Dataset 2: Mantenimiento de máquina
    elif opcion == '2':
        # Estados: Inactivo, Trabajando, Roto, Reparando
        estados = ['Idle','Working','Broken','Repairing']
        # Acciones: Operar la máquina o Mantenerla
        acciones = ['Operate','Maintain']
        # Probabilidades de transición del sistema de mantenimiento
        P = {
            ('Idle','Operate'): {'Working':1.0},              # Si está Idle y Operamos → Working (100%)
            ('Idle','Maintain'): {'Idle':1.0},                # Si está Idle y Mantenemos → sigue Idle
            ('Working','Operate'): {'Working':0.8, 'Broken':0.2},  # Working + Operar → 80% sigue Working, 20% se Rompe
            ('Working','Maintain'): {'Working':1.0},          # Working + Mantener → sigue Working (previene fallas)
            ('Broken','Operate'): {'Broken':1.0},             # Broken + Operar → sigue Broken (no se puede usar roto)
            ('Broken','Maintain'): {'Repairing':1.0},         # Broken + Mantener → pasa a Repairing
            ('Repairing','Operate'): {'Idle':1.0},            # Repairing + Operar → vuelve a Idle
            ('Repairing','Maintain'): {'Idle':1.0},           # Repairing + Mantener → vuelve a Idle
        }
        # Recompensas del sistema de mantenimiento
        R = {
            ('Idle','Operate','Working'): 5,          # Arrancar la máquina genera valor moderado
            ('Idle','Maintain','Idle'): 0,            # Mantener sin usar no genera valor
            ('Working','Operate','Working'): 10,      # Máquina trabajando genera máximo valor
            ('Working','Operate','Broken'): -50,      # Romperse genera gran pérdida
            ('Working','Maintain','Working'): 2,      # Mantenimiento preventivo genera poco valor pero evita fallas
            ('Broken','Operate','Broken'): -10,       # Intentar usar máquina rota genera pérdidas
            ('Broken','Maintain','Repairing'): -10,   # Iniciar reparación tiene costo
            ('Repairing','Operate','Idle'): 0,        # Terminar reparación no da recompensa inmediata
            ('Repairing','Maintain','Idle'): 0,       # Terminar reparación no da recompensa inmediata
        }
        print("\n[Dataset 2] Mantenimiento de máquina (4 estados, 2 acciones) cargado exitosamente.")
    
    # Opción 3: Personalizar (ingreso manual)
    elif opcion == '3':
        print("\n[Modo Personalizado]")
        # Paso 3.1: Definir estados del sistema
        n = int(input("¿Cuántos estados tiene el MDP? ").strip())
        estados = []
        for i in range(n):
            estados.append(input(f"  Etiqueta del estado {i+1}: ").strip())
        
        # Paso 3.2: Definir acciones disponibles
        m = int(input("¿Cuántas acciones posibles hay? ").strip())
        acciones = []
        for j in range(m):
            acciones.append(input(f"  Etiqueta de la acción {j+1}: ").strip())
        
        P = {}
        R = {}
        # Paso 3.3: Definir probabilidades de transición
        print("\nIntroduce probabilidades de transición P(s,a → s'):")
        for s in estados:
            for a in acciones:
                print(f"Estado {s}, acción {a}:")
                trans = {}
                total = 0.0
                # Para cada estado destino, pedir la probabilidad
                for s2 in estados:
                    p = float(input(f"  P({s} → {s2} | acción {a}) = ").strip())
                    trans[s2] = p
                    total += p
                # Verificar que las probabilidades sumen 1 (propiedad de distribución de probabilidad)
                if abs(total - 1.0) > 1e-6:
                    print("  Atención: probabilidades no suman 1. Se normalizarán automáticamente.")
                    for s2 in trans:
                        trans[s2] /= total
                P[(s,a)] = trans
        
        # Paso 3.4: Definir recompensas para cada transición posible
        print("\nIntroduce recompensas R(s,a,s'):")
        for (s,a), trans in P.items():
            for s2, p in trans.items():
                # Solo pedir recompensas para transiciones con probabilidad > 0
                if p > 0:
                    r = float(input(f"  R({s},{a} → {s2}) = ").strip())
                    R[(s,a,s2)] = r
        print("\n✓ Modo personalizado configurado correctamente.")
    
    # Opción no válida: usar Dataset 1 por defecto
    else:
        print("\nOpción no válida, se usará Dataset 1 por defecto.")
        estados = ['s0','s1','s2']
        acciones = ['a0','a1']
        P = {
            ('s0','a0'): {'s0':0.5, 's1':0.5},
            ('s0','a1'): {'s1':1.0},
            ('s1','a0'): {'s2':1.0},
            ('s1','a1'): {'s0':0.3,'s2':0.7},
            ('s2','a0'): {'s2':1.0},
            ('s2','a1'): {'s0':1.0},
        }
        R = {
            ('s0','a0','s0'): 1,   ('s0','a0','s1'): 0,
            ('s0','a1','s1'): 5,
            ('s1','a0','s2'): 10,
            ('s1','a1','s0'):-1,  ('s1','a1','s2'):2,
            ('s2','a0','s2'):0,
            ('s2','a1','s0'):0,
        }
        print("[Dataset 1] MDP pequeño de 3 estados (por defecto).")

    # ========== PASO 2: Configurar parámetros y calcular ==========
    # Solicita el factor de descuento
    gamma = float(input("\nIntroduce el factor de descuento γ (0-1): ").strip())
    
    # Crea una política simple: primera acción para todos los estados
    # Política uniforme: aplica la misma acción en cualquier estado
    politica = {s: acciones[0] for s in estados}
    print(f"\nPolítica utilizada (primera acción para todos los estados): {politica}")
    
    # ========== PASO 3: Calcular valor de la política ==========
    # Calcula el valor de la política desde el estado inicial
    print("\n⟳ Ejecutando cálculo de valor de política con el dataset seleccionado...")
    valor = valor_de_politica(estados[0], estados, acciones, P, R, politica, gamma)
    print(f"\n✓ Valor estimado de iniciar en estado {estados[0]}: V ≈ {valor:.4f}")
    print(f"  (Este es el valor esperado acumulado siguiendo la política con γ={gamma})")

def main():
    """
    Función principal que permite elegir entre modo demo o interactivo.
    """
    print("=" * 60)
    print(" PROCESO DE DECISIÓN DE MARKOV (MDP)")
    print("=" * 60)
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (ejemplo predefinido)")
    print("2) Modo INTERACTIVO (datasets precargados + personalizar)\n")
    opcion = input("Ingrese el número de opción: ").strip()
    
    # Ejecuta el modo seleccionado
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("⚠ Opción no válida. Se usará DEMO por defecto.")
        modo_demo()

if __name__ == "__main__":
    main()
