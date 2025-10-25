"""
033-E1-refuerzo_pasivo.py
--------------------------
Este script implementa una versión simplificada de lo que se conoce como **Aprendizaje por Refuerzo Pasivo**:
- El agente sigue una política fija (pre-definida) y **no toma decisiones de exploración** autónoma.
- El objetivo es estimar la función de utilidad (o valor) de esa política, es decir U^π(s) para cada estado s.
- Usa el método Monte Carlo: ejecuta episodios completos y promedia los retornos observados.
- Incluye dos modos:
    1. MODO DEMO: problema sencillo con política fija y ejecución simulada.
    2. MODO INTERACTIVO: el usuario selecciona un entorno pre-definido, se ejecuta la política fija varios episodios,
       y se muestra la estimación de valores por estado.
- Variables y funciones en español.

Conceptos clave:
- Refuerzo Pasivo: El agente NO elige acciones, solo sigue una política fija π(s)
- Monte Carlo First-Visit: Estima valores promediando retornos de la primera visita a cada estado
- Retorno G_t: Suma descontada de recompensas desde el tiempo t: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

Autor: Alejandro Aguirre Díaz
"""

import random

def simular_politica(estados, transiciones, recompensas, politica, gamma, episodios, max_pasos=50):
    """
    Simula la política fija π durante varios episodios para estimar utilidades usando Monte Carlo.
    :param estados: lista de estados posibles
    :param transiciones: dict (s, a) → dict s' → probabilidad
    :param recompensas: dict (s,a,s') → recompensa inmediata
    :param politica: dict s → acción fija
    :param gamma: factor de descuento (0 ≤ γ < 1)
    :param episodios: número de episodios a ejecutar
    :param max_pasos: número máximo de pasos por episodio
    :return: dict estado→estimación promedio de U^π(s)
    """
    # ========== FASE 1: Inicialización de estructuras de datos ==========
    # Inicializar acumuladores para calcular promedios
    # suma_retornos: suma total de retornos observados para cada estado
    # visitas: número de veces que cada estado ha sido visitado
    suma_retornos = {s: 0.0 for s in estados}
    visitas = {s: 0 for s in estados}

    # ========== FASE 2: Ejecución de episodios ==========
    # Ejecutar múltiples episodios para recolectar estadísticas
    for episodio in range(episodios):
        # --- 2.1: Iniciar nuevo episodio ---
        # Iniciar episodio en un estado aleatorio (no hay estado inicial fijo)
        estado_inicial = random.choice(estados)
        
        # --- 2.2: Generar trayectoria completa ---
        # Generar un episodio completo siguiendo la política π
        trayectoria = []  # Lista de tuplas (estado, accion, recompensa)
        s = estado_inicial  # Estado actual
        
        # Simular pasos del episodio hasta max_pasos o hasta estado sin salida
        for paso in range(max_pasos):
            # Obtener la acción según la política fija π(s)
            # La política dicta qué acción tomar en cada estado
            a = politica.get(s)
            if a is None:
                # Estado sin acción definida (posiblemente estado terminal)
                break
            
            # Obtener la distribución de transición P(s'|s,a)
            # Es decir, las probabilidades de ir a cada estado siguiente
            distrib = transiciones.get((s, a), {})
            if not distrib:
                # No hay transiciones disponibles desde este estado con esta acción
                break
            
            # Seleccionar el siguiente estado s' según las probabilidades de transición
            # Esto simula la estocasticidad del entorno
            estados_sig = list(distrib.keys())
            probs = list(distrib.values())
            s_prime = random.choices(estados_sig, weights=probs)[0]
            
            # Obtener la recompensa R(s,a,s') por esta transición
            # La recompensa puede depender del estado origen, acción y estado destino
            r = recompensas.get((s, a, s_prime), 0.0)
            
            # Guardar la transición en la trayectoria del episodio
            # Necesitamos esto para calcular retornos después
            trayectoria.append((s, a, r))
            
            # Avanzar al siguiente estado para continuar el episodio
            s = s_prime
        
        # --- 2.3: Calcular retornos y actualizar estadísticas ---
        # Calcular retornos desde cada estado visitado (método first-visit)
        # Retorno G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... (suma descontada)
        G = 0.0  # Retorno acumulado (empezamos desde el final con G=0)
        estados_visitados = set()  # Para implementar first-visit (solo primera aparición)
        
        # Recorrer la trayectoria en reversa para calcular retornos eficientemente
        # Vamos del final al inicio para aplicar: G_t = r_t + γ * G_{t+1}
        for i in range(len(trayectoria) - 1, -1, -1):
            estado, accion, recompensa = trayectoria[i]
            
            # Actualizar retorno usando la ecuación de Bellman hacia atrás
            # G_t = r_t + γ * G_{t+1}
            # En cada paso hacia atrás, incorporamos la recompensa actual
            # y descontamos el retorno futuro
            G = recompensa + gamma * G
            
            # Implementar first-visit Monte Carlo:
            # Solo contar la primera visita a cada estado en este episodio
            # Esto reduce varianza comparado con every-visit MC
            if estado not in estados_visitados:
                estados_visitados.add(estado)
                
                # Acumular el retorno observado para este estado
                # Esto se usará para calcular el promedio al final
                suma_retornos[estado] += G
                
                # Incrementar contador de visitas a este estado
                visitas[estado] += 1

    # ========== FASE 3: Calcular estimaciones finales ==========
    # Calcular estimación promedio de U^π(s) para cada estado
    # U^π(s) ≈ promedio de retornos observados desde s
    estimacion = {}
    for s in estados:
        if visitas[s] > 0:
            # Promedio de todos los retornos observados desde este estado
            # Esto converge al valor verdadero V^π(s) con suficientes muestras
            estimacion[s] = suma_retornos[s] / visitas[s]
        else:
            # Estado nunca visitado durante la simulación, asumir valor 0
            # (alternativa: mantener valor inicial o usar interpolación)
            estimacion[s] = 0.0
    
    return estimacion

def modo_demo():
    print("\nMODO DEMO: Aprendizaje por Refuerzo Pasivo")
    print("=" * 60)
    
    # ========== DEFINICIÓN DEL PROBLEMA ==========
    
    # Definir el espacio de estados del MDP
    # En este ejemplo: 3 estados discretos A, B, C
    estados = ['A','B','C']
    
    # Definir el conjunto de acciones disponibles
    # Acciones genéricas 'x' e 'y' (podrían ser cualquier nombre)
    acciones = ['x','y']
    
    # ========== MODELO DE TRANSICIÓN ==========
    # Definir el modelo de transición: P(s'|s,a)
    # Diccionario: (estado_actual, accion) -> {estado_siguiente: probabilidad}
    # Nota: No todas las acciones están disponibles en todos los estados
    transiciones = {
        ('A','x'): {'A':0.8,'B':0.2},  # Desde A con acción x: 80% queda en A, 20% va a B
        ('A','y'): {'B':1.0},           # Desde A con acción y: 100% va a B (determinista)
        ('B','x'): {'C':1.0},           # Desde B con acción x: 100% va a C
        ('B','y'): {'A':0.5,'C':0.5},   # Desde B con acción y: 50% vuelve a A, 50% va a C
        ('C','x'): {'C':1.0},           # Desde C con acción x: 100% se queda en C (bucle)
        ('C','y'): {'A':1.0}            # Desde C con acción y: 100% vuelve a A (reinicio)
    }
    
    # ========== FUNCIÓN DE RECOMPENSA ==========
    # Definir las recompensas: R(s,a,s')
    # Diccionario: (estado_origen, accion, estado_destino) -> recompensa_numérica
    # Recompensa recibida al realizar la transición s --a--> s'
    recompensas = {
        ('A','x','A'): 2,    # Permanecer en A da recompensa positiva (incentivo)
        ('A','x','B'): 0,    # Ir de A a B con x no da recompensa
        ('A','y','B'): 1,    # Ir de A a B con y da recompensa pequeña
        ('B','x','C'): 4,    # Ir de B a C da recompensa alta (muy deseable)
        ('B','y','A'): -1,   # Regresar a A desde B tiene penalización (indeseable)
        ('B','y','C'): 3,    # Ir de B a C con y da buena recompensa
        ('C','x','C'): 0,    # Permanecer en C no da recompensa ni penalización
        ('C','y','A'): 2     # Volver a A desde C da recompensa moderada
    }
    
    # ========== POLÍTICA FIJA ==========
    # Definir una política fija π(s) = a
    # Diccionario: estado -> acción_a_tomar
    # Esta política NO se optimiza, solo se evalúa (por eso es "pasivo")
    # El agente SIEMPRE sigue esta política, sin exploración
    politica = {
        'A': 'x',  # En estado A, siempre ejecutar acción x
        'B': 'y',  # En estado B, siempre ejecutar acción y
        'C': 'y'   # En estado C, siempre ejecutar acción y
    }
    
    # ========== PARÁMETROS DEL ALGORITMO ==========
    # Factor de descuento γ (gamma): controla la importancia de recompensas futuras
    # γ cercano a 0: solo importan recompensas inmediatas
    # γ cercano a 1: recompensas futuras son casi tan importantes como las inmediatas
    gamma = 0.9       # Factor de descuento (γ = 0.9 es valor típico)
    
    # Número de episodios a simular para estimar valores
    # Más episodios = mejor estimación pero más tiempo de cómputo
    episodios = 1000  # Número de episodios para estimar valores

    # ========== MOSTRAR CONFIGURACIÓN ==========
    print("\nCONFIGURACIÓN:")
    print(f"Estados: {estados}")
    print(f"Política fija π: {politica}")
    print(f"Factor de descuento γ: {gamma}")
    print(f"Número de episodios: {episodios}")
    
    # ========== EJECUTAR SIMULACIÓN ==========
    # Ejecutar simulación Monte Carlo para estimar U^π(s)
    # U^π(s) = valor esperado de seguir π desde estado s
    print("\nEjecutando simulación Monte Carlo...")
    estimacion = simular_politica(estados, transiciones, recompensas, politica, gamma, episodios)
    
    # ========== MOSTRAR RESULTADOS ==========
    print("\nRESULTADOS - Estimación de U^π(s) para cada estado:")
    print("-" * 60)
    for s in sorted(estimacion.keys()):
        print(f"  Estado {s}: U^π ≈ {estimacion[s]:.3f}")
    
    # Explicar el significado de los valores
    print("\nInterpretación: Estos valores representan el retorno esperado")
    print("que se obtiene al seguir la política π desde cada estado.")
    print("Valores más altos indican estados más 'valiosos' bajo esta política.")

def modo_interactivo():
    print("\nMODO INTERACTIVO: Aprendizaje por Refuerzo Pasivo")
    print("=" * 60)
    print("\nEscenarios predefinidos:")
    print("1) Entorno simple 3 estados (A, B, C)")
    print("2) Entorno GridWorld 2x2")
    
    # Solicitar al usuario que elija un escenario
    opcion = input("\nIntroduce el número de escenario: ").strip()
    
    # ========== ESCENARIO 2: GRIDWORLD 2x2 ==========
    if opcion == '2':
        # GridWorld 2x2: estados representan posiciones en una cuadrícula
        # Representación de la cuadrícula:
        #   (0,0) | (0,1)
        #   ------|------
        #   (1,0) | (1,1) <- META
        estados = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
        
        # ========== TRANSICIONES GRIDWORLD ==========
        # Acciones: arriba, abajo, izquierda, derecha
        # El agente se mueve determinísticamente en la dirección elegida
        # Nota: no se definen movimientos que salgan del grid
        transiciones = {
            # Desde (0,0) - esquina superior izquierda
            ('(0,0)','derecha'): {'(0,1)':1.0},  # Ir a la derecha
            ('(0,0)','abajo'): {'(1,0)':1.0},    # Ir hacia abajo
            
            # Desde (0,1) - esquina superior derecha
            ('(0,1)','izquierda'): {'(0,0)':1.0},  # Volver a la izquierda
            ('(0,1)','abajo'): {'(1,1)':1.0},      # Ir hacia abajo (a la META)
            
            # Desde (1,0) - esquina inferior izquierda
            ('(1,0)','arriba'): {'(0,0)':1.0},     # Ir hacia arriba
            ('(1,0)','derecha'): {'(1,1)':1.0},    # Ir a la derecha (a la META)
            
            # Desde (1,1) - META (esquina inferior derecha)
            ('(1,1)','arriba'): {'(0,1)':1.0},       # Salir de la meta hacia arriba
            ('(1,1)','izquierda'): {'(1,0)':1.0},    # Salir de la meta hacia izquierda
            ('(1,1)','abajo'): {'(1,1)':1.0}         # Quedarse en la meta (estado absorbente)
        }
        
        # ========== RECOMPENSAS GRIDWORLD ==========
        # Objetivo: alcanzar (1,1) da recompensa grande (+10)
        # Otros movimientos tienen costo (-1) para incentivar rutas cortas
        recompensas = {
            # Movimientos desde estados no-meta: costo de -1 por paso
            ('(0,0)','derecha','(0,1)'): -1,
            ('(0,0)','abajo','(1,0)'): -1,
            ('(0,1)','izquierda','(0,0)'): -1,
            
            # Transiciones que llegan a la META: recompensa alta +10
            ('(0,1)','abajo','(1,1)'): 10,    # Llegar a la meta desde arriba
            ('(1,0)','derecha','(1,1)'): 10,  # Llegar a la meta desde la izquierda
            
            # Movimientos desde la meta
            ('(1,0)','arriba','(0,0)'): -1,
            ('(1,1)','arriba','(0,1)'): -1,      # Salir de la meta (no deseable)
            ('(1,1)','izquierda','(1,0)'): -1,   # Salir de la meta (no deseable)
            ('(1,1)','abajo','(1,1)'): 0         # Permanecer en la meta (sin costo)
        }
        
        # ========== POLÍTICA GRIDWORLD ==========
        # Política simple: ir hacia la esquina inferior derecha (META)
        # Esta política intenta llegar a (1,1) por el camino más corto
        politica = {
            '(0,0)': 'derecha',  # Desde origen: ir a la derecha primero
            '(0,1)': 'abajo',    # Desde arriba-derecha: bajar a la meta
            '(1,0)': 'derecha',  # Desde abajo-izquierda: ir a la derecha a la meta
            '(1,1)': 'abajo'     # Ya en la meta: quedarse (bucle)
        }
        print("\nHas elegido GridWorld 2x2")
        
    # ========== ESCENARIO 1 (por defecto): ENTORNO SIMPLE ==========
    else:
        # Escenario por defecto: entorno simple de 3 estados (mismo que DEMO)
        estados = ['A','B','C']
        
        # Transiciones estocásticas (algunas acciones tienen resultados probabilísticos)
        transiciones = {
            ('A','x'): {'A':0.8,'B':0.2},  # Acción x desde A es estocástica
            ('A','y'): {'B':1.0},
            ('B','x'): {'C':1.0},
            ('B','y'): {'A':0.5,'C':0.5},  # Acción y desde B es estocástica
            ('C','x'): {'C':1.0},
            ('C','y'): {'A':1.0}
        }
        
        # Recompensas variadas para hacer el problema interesante
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
        
        # Política arbitraria para evaluar
        politica = {
            'A': 'x',
            'B': 'y',
            'C': 'y'
        }
        print("\nHas elegido el entorno simple de 3 estados")
    
    # ========== CONFIGURACIÓN Y PARÁMETROS ==========
    # Mostrar información del problema seleccionado
    print(f"\nEstados disponibles: {estados}")
    print(f"Política fija π: {politica}")
    
    # Solicitar parámetros del algoritmo al usuario
    # Factor de descuento: controla cuánto valoramos recompensas futuras
    gamma = float(input("\nIntroduce factor de descuento γ (entre 0 y 1, ej: 0.9): ").strip())
    
    # Número de episodios: más episodios = mejor estimación
    episodios = int(input("Introduce número de episodios (ej: 1000): ").strip())
    
    # ========== EJECUTAR SIMULACIÓN ==========
    # Ejecutar algoritmo Monte Carlo para estimar valores de la política
    print("\nEjecutando simulación Monte Carlo...")
    estimacion = simular_politica(estados, transiciones, recompensas, politica, gamma, episodios)
    
    # ========== MOSTRAR RESULTADOS ==========
    # Mostrar los valores estimados para cada estado
    print("\nRESULTADOS - Estimación de U^π(s) para cada estado:")
    print("-" * 60)
    for s in sorted(estimacion.keys()):
        print(f"  Estado {s}: U^π ≈ {estimacion[s]:.3f}")
    
    print("\nSimulación completada.")
    print("Nota: Los valores representan qué tan 'bueno' es cada estado")
    print("      bajo la política fija π que se está evaluando.")

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
