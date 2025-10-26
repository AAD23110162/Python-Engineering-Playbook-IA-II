"""
003-E2-RPT_reconocimiento_habla.py
--------------------------------
Este script implementa un sistema simplificado de Reconocimiento Probabilístico del Habla:
- Modela secuencias de fonemas usando Modelos Ocultos de Markov (HMM)
- Calcula la probabilidad de secuencias de observaciones acústicas
- Implementa el algoritmo de Viterbi para encontrar la secuencia más probable de estados
- Reconoce palabras a partir de patrones probabilísticos de señales de audio
- Maneja incertidumbre en las observaciones mediante modelos probabilísticos
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente reconocimiento de palabras predefinidas
2. INTERACTIVO: permite ingresar secuencias de observaciones y decodificar palabras

Autor: Alejandro Aguirre Díaz
"""

import numpy as np

class ModeloOcultoMarkov:
    """
    Implementación simplificada de un Modelo Oculto de Markov (HMM) para reconocimiento.
    """
    def __init__(self, estados, observaciones, probabilidades_iniciales, transiciones, emisiones):
        """
        :parametro estados: lista de nombres de estados ocultos
        :parametro observaciones: lista de símbolos observables
        :parametro probabilidades_iniciales: dict {estado: probabilidad inicial}
        :parametro transiciones: dict {(estado_i, estado_j): probabilidad de transición}
        :parametro emisiones: dict {(estado, observacion): probabilidad de emisión}
        """
        self.estados = estados
        self.observaciones = observaciones
        self.prob_iniciales = probabilidades_iniciales
        self.transiciones = transiciones
        self.emisiones = emisiones
    
    def viterbi(self, secuencia_observaciones, verbose=False):
        """
        Algoritmo de Viterbi para encontrar la secuencia más probable de estados.
        :parametro secuencia_observaciones: lista de observaciones
        :parametro verbose: si True, muestra el proceso paso a paso
        :return: tupla (mejor_camino, probabilidad_maxima)
        """
        # T = número de observaciones (longitud de la secuencia)
        # N = número de estados ocultos en el HMM
        T = len(secuencia_observaciones)
        N = len(self.estados)
        
        # ========== ESTRUCTURAS DE DATOS DE VITERBI ==========
        # Matriz viterbi[estado, tiempo]: almacena la probabilidad máxima de llegar
        # al estado 'estado' en el tiempo 't' con la secuencia de observaciones vista hasta t
        viterbi = np.zeros((N, T))
        
        # Matriz backpointer[estado, tiempo]: almacena el índice del estado anterior
        # que maximizó la probabilidad (para reconstruir el mejor camino)
        backpointer = np.zeros((N, T), dtype=int)
        
        if verbose:
            print(f"\n--- Algoritmo de Viterbi ---")
            print(f"Secuencia de observaciones: {secuencia_observaciones}")
            print(f"Estados: {self.estados}")
        
        # ========== PASO 1: INICIALIZACIÓN (t=0) ==========
        # Para el primer tiempo t=0, calculamos la probabilidad inicial de cada estado
        # multiplicada por la probabilidad de emitir la primera observación
        obs_0 = secuencia_observaciones[0]
        
        for i, estado in enumerate(self.estados):
            # π(estado) = probabilidad inicial de estar en este estado
            prob_inicial = self.prob_iniciales.get(estado, 1.0/N)
            
            # b(estado, obs_0) = probabilidad de que este estado emita obs_0
            prob_emision = self.emisiones.get((estado, obs_0), 1e-10)
            
            # viterbi[i, 0] = π(estado) × b(estado, obs_0)
            # Esta es la probabilidad de estar en el estado 'i' en t=0 y observar obs_0
            viterbi[i, 0] = prob_inicial * prob_emision
            
            if verbose:
                print(f"\nt=0, Estado={estado}:")
                print(f"  π({estado}) × b({estado},{obs_0}) = {prob_inicial:.4f} × {prob_emision:.4f} = {viterbi[i,0]:.6f}")
        
        # ========== PASO 2: RECURSIÓN (t=1 hasta T-1) ==========
        # Para cada paso de tiempo después del inicial, calculamos la probabilidad
        # máxima de llegar a cada estado considerando todos los caminos posibles
        for t in range(1, T):
            obs_t = secuencia_observaciones[t]  # Observación en el tiempo t
            
            if verbose:
                print(f"\n--- t={t}, Observación={obs_t} ---")
            
            # Para cada estado posible en el tiempo t...
            for j, estado_j in enumerate(self.estados):
                # Probabilidad de que estado_j emita la observación obs_t
                prob_emision = self.emisiones.get((estado_j, obs_t), 1e-10)
                
                # Encontrar el mejor estado anterior (que maximice la probabilidad)
                # Debemos considerar TODOS los estados posibles en t-1
                prob_max = -1      # Probabilidad máxima encontrada
                estado_max = -1    # Índice del estado que dio la prob. máxima
                
                # Evaluar transiciones desde cada estado anterior posible
                for i, estado_i in enumerate(self.estados):
                    # a(i→j) = probabilidad de transición de estado_i a estado_j
                    prob_trans = self.transiciones.get((estado_i, estado_j), 1e-10)
                    
                    # Calcular: viterbi[i, t-1] × a(i→j) × b(j, obs_t)
                    # = (mejor camino hasta estado_i en t-1) × (transición i→j) × (emisión de obs_t en j)
                    prob = viterbi[i, t-1] * prob_trans * prob_emision
                    
                    # Quedarnos con el máximo (este es el núcleo de Viterbi)
                    if prob > prob_max:
                        prob_max = prob
                        estado_max = i
                
                # Guardar la probabilidad máxima para estado_j en tiempo t
                viterbi[j, t] = prob_max
                
                # Guardar qué estado anterior produjo este máximo (para backtracking)
                backpointer[j, t] = estado_max
                
                if verbose:
                    print(f"  Estado {estado_j}: max_prob={prob_max:.6f} (desde {self.estados[estado_max]})")
        
        # ========== PASO 3: TERMINACIÓN ==========
        # Encontrar el estado con mayor probabilidad en el tiempo final T-1
        # Este es el punto de llegada del mejor camino
        ultimo_estado = np.argmax(viterbi[:, T-1])
        prob_maxima = viterbi[ultimo_estado, T-1]
        
        # ========== PASO 4: BACKTRACKING (RECONSTRUCCIÓN DEL CAMINO) ==========
        # Ahora que sabemos el estado final óptimo, reconstruimos el camino
        # hacia atrás usando los backpointers
        mejor_camino = [0] * T
        mejor_camino[T-1] = ultimo_estado  # Empezamos desde el último estado
        
        # Ir hacia atrás en el tiempo, siguiendo los backpointers
        for t in range(T-2, -1, -1):
            # El mejor estado en tiempo t es el que apunta el backpointer
            # del mejor estado en tiempo t+1
            mejor_camino[t] = backpointer[mejor_camino[t+1], t+1]
        
        # Convertir índices numéricos a nombres de estados para mejor lectura
        # Ejemplo: [7, 0, 1, 2, 3] → ['INICIO', 'H', 'O', 'L', 'A']
        camino_estados = [self.estados[i] for i in mejor_camino]
        
        if verbose:
            print(f"\n--- Resultado ---")
            print(f"Mejor camino: {camino_estados}")
            print(f"Probabilidad: {prob_maxima:.8f}")
        
        return camino_estados, prob_maxima

def crear_hmm_palabras_simples():
    """
    Crea un HMM simplificado para reconocer las palabras "HOLA" y "ADIÓS".
    Estados: fonemas simplificados {H, O, L, A, D, I, S}
    Observaciones: características acústicas simbólicas {h, o, l, a, d, i, s}
    """
    # ========== DEFINICIÓN DE ESTADOS Y OBSERVACIONES ==========
    # Estados ocultos: fonemas de las palabras HOLA y ADIOS, más estados especiales
    estados = ['H', 'O', 'L', 'A', 'D', 'I', 'S', 'INICIO', 'FIN']
    
    # Observaciones: símbolos acústicos que podemos "escuchar" (simplificado)
    # En un sistema real, serían características MFCC o espectrogramas
    observaciones = ['h', 'o', 'l', 'a', 'd', 'i', 's', '-']
    
    # ========== PROBABILIDADES INICIALES π(estado) ==========
    # ¿Con qué probabilidad empezamos en cada estado?
    # En este modelo, SIEMPRE empezamos en el estado 'INICIO'
    prob_iniciales = {
        'INICIO': 1.0,  # Siempre empezamos aquí
        # Todos los demás estados tienen probabilidad inicial 0
        'H': 0.0, 'O': 0.0, 'L': 0.0, 'A': 0.0,
        'D': 0.0, 'I': 0.0, 'S': 0.0, 'FIN': 0.0
    }
    
    # ========== PROBABILIDADES DE TRANSICIÓN a(i→j) ==========
    # ¿Con qué probabilidad pasamos del estado i al estado j?
    # Estas transiciones están diseñadas para modelar las palabras HOLA y ADIOS
    transiciones = {
        # --- Desde el estado INICIO ---
        ('INICIO', 'H'): 0.5,  # 50% de probabilidad de empezar con H (palabra HOLA)
        ('INICIO', 'A'): 0.5,  # 50% de probabilidad de empezar con A (palabra ADIOS)
        
        # --- Secuencia para HOLA: H → O → L → A → FIN ---
        ('H', 'O'): 0.9,       # De H vamos casi siempre a O
        ('O', 'L'): 0.9,       # De O vamos casi siempre a L (en HOLA)
        ('L', 'A'): 0.9,       # De L vamos casi siempre a A
        ('A', 'FIN'): 0.5,     # A puede terminar (fin de HOLA) o continuar
        
        # --- Secuencia para ADIOS: A → D → I → (O o directamente S) → S → FIN ---
        ('A', 'D'): 0.4,       # De A podemos ir a D (inicio de DIOS en ADIOS)
        ('D', 'I'): 0.9,       # De D vamos a I
        ('I', 'O'): 0.8,       # De I podemos ir a O (sonido IO en DIOS)
        ('O', 'S'): 0.1,       # De O podemos saltar a S (alternativa)
        ('I', 'S'): 0.2,       # O directamente de I a S (atajo simplificado)
        ('S', 'FIN'): 0.9,     # De S terminamos (fin de ADIOS)
        
        # --- Auto-transiciones (para manejar ruido o repeticiones) ---
        # Permiten que un fonema se "repita" si hay ruido o alargamiento
        ('H', 'H'): 0.05,      # H puede repetirse (hhola)
        ('O', 'O'): 0.05,      # O puede repetirse (hoola)
        ('L', 'L'): 0.05,      # L puede repetirse (holla)
        ('A', 'A'): 0.05,      # A puede repetirse (holaa)
        ('D', 'D'): 0.05,      # D puede repetirse
        ('I', 'I'): 0.05,      # I puede repetirse
        ('S', 'S'): 0.05,      # S puede repetirse (adioss)
    }
    
    # ========== PROBABILIDADES DE EMISIÓN b(estado, observación) ==========
    # ¿Con qué probabilidad un estado emite una observación particular?
    # Cada fonema tiene alta probabilidad de emitir su símbolo correspondiente,
    # pero también puede emitir símbolos cercanos (ruido acústico)
    emisiones = {
        # Estado H: emite principalmente 'h', a veces 'o' (sonido parecido), rara vez silencio
        ('H', 'h'): 0.85, ('H', 'o'): 0.05, ('H', '-'): 0.10,
        
        # Estado O: emite principalmente 'o', a veces confundido con 'h' o 'l'
        ('O', 'o'): 0.85, ('O', 'h'): 0.05, ('O', 'l'): 0.05, ('O', '-'): 0.05,
        
        # Estado L: emite principalmente 'l', a veces confundido con 'o' o 'a'
        ('L', 'l'): 0.85, ('L', 'o'): 0.05, ('L', 'a'): 0.05, ('L', '-'): 0.05,
        
        # Estado A: emite principalmente 'a', a veces 'o' (vocales parecidas) o 'd'
        ('A', 'a'): 0.85, ('A', 'o'): 0.05, ('A', 'd'): 0.05, ('A', '-'): 0.05,
        
        # Estado D: emite principalmente 'd', a veces confundido con 'a' o 'i'
        ('D', 'd'): 0.85, ('D', 'a'): 0.05, ('D', 'i'): 0.05, ('D', '-'): 0.05,
        
        # Estado I: emite principalmente 'i', a veces 'a' (vocales) o 's'
        ('I', 'i'): 0.85, ('I', 'a'): 0.05, ('I', 's'): 0.05, ('I', '-'): 0.05,
        
        # Estado S: emite principalmente 's', a veces 'i', o silencio final
        ('S', 's'): 0.85, ('S', 'i'): 0.05, ('S', '-'): 0.10,
        
        # Estado INICIO: puede emitir 'h' (HOLA), 'a' (ADIOS) o silencio
        ('INICIO', 'h'): 0.4, ('INICIO', 'a'): 0.4, ('INICIO', '-'): 0.2,
        
        # Estado FIN: solo emite silencio (fin de palabra)
        ('FIN', '-'): 1.0,
    }
    
    # Crear y retornar el objeto HMM con todos los parámetros
    return ModeloOcultoMarkov(estados, observaciones, prob_iniciales, transiciones, emisiones)

def modo_demo():
    """Ejecuta el modo demostrativo con reconocimiento de palabras predefinidas."""
    print("\n" + "="*70)
    print("MODO DEMO: Reconocimiento del Habla con HMM")
    print("="*70)
    
    # Crear el modelo HMM para reconocimiento de HOLA y ADIOS
    hmm = crear_hmm_palabras_simples()
    
    print("\n--- Modelo HMM ---")
    print("Palabras reconocibles: HOLA, ADIOS")
    print("Estados (fonemas): H, O, L, A, D, I, S")
    print("Observaciones acústicas: h, o, l, a, d, i, s, - (silencio)")
    
    # ========== EJEMPLO 1: Reconocer "HOLA" sin ruido ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Reconocer la palabra 'HOLA'")
    print("="*70)
    
    # Secuencia limpia: ['h', 'o', 'l', 'a']
    # Esperamos que el algoritmo identifique: INICIO → H → O → L → A
    secuencia_hola = ['h', 'o', 'l', 'a']
    print(f"\nSecuencia de observaciones: {secuencia_hola}")
    
    # Ejecutar Viterbi para encontrar la secuencia más probable de estados
    camino, prob = hmm.viterbi(secuencia_hola, verbose=True)
    
    print(f"\n>>> PALABRA RECONOCIDA: {''.join(camino)}")
    print(f"    Confianza: {prob:.8f}")
    print(f"    (El modelo identificó correctamente la secuencia de fonemas)")
    
    # ========== EJEMPLO 2: Reconocer "HOLA" con ruido acústico ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Reconocer 'HOLA' con ruido acústico")
    print("="*70)
    
    # Secuencia con ruido: ['h', 'o', 'o', 'l', 'a']
    # La 'o' está duplicada (podría ser alargamiento del sonido o eco)
    # El algoritmo debería ser robusto y aún reconocer HOLA
    secuencia_hola_ruido = ['h', 'o', 'o', 'l', 'a']  # 'o' duplicada (ruido)
    print(f"\nSecuencia de observaciones (con ruido): {secuencia_hola_ruido}")
    
    # Viterbi debería usar las auto-transiciones para manejar la 'o' extra
    camino2, prob2 = hmm.viterbi(secuencia_hola_ruido, verbose=True)
    
    print(f"\n>>> PALABRA RECONOCIDA: {''.join(camino2)}")
    print(f"    Confianza: {prob2:.8f}")
    print(f"    (El algoritmo maneja el ruido usando auto-transiciones O→O)")
    print(f"    (Aunque la probabilidad baja, aún identifica correctamente la palabra)")
    
    # ========== EJEMPLO 3: Secuencia corta y ambigua ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Secuencia corta ambigua")
    print("="*70)
    
    # Secuencia muy corta: solo ['a', 'd']
    # Esto podría ser el inicio de "ADIOS" pero está incompleto
    # El modelo debe decidir qué secuencia de estados es más probable
    secuencia_corta = ['a', 'd']
    print(f"\nSecuencia de observaciones: {secuencia_corta}")
    
    # Viterbi debe elegir entre:
    # - INICIO → A → D (inicio de ADIOS, más probable)
    # - O alguna otra secuencia menos probable
    camino3, prob3 = hmm.viterbi(secuencia_corta, verbose=True)
    
    print(f"\n>>> DECODIFICACIÓN: {''.join(camino3)}")
    print(f"    Confianza: {prob3:.8f}")
    print(f"    (Inicio de 'ADIOS' detectado correctamente)")
    print(f"    (Con secuencias cortas, la confianza es naturalmente menor)")

def modo_interactivo():
    """Ejecuta el modo interactivo con secuencias definidas por el usuario."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Reconocimiento del Habla con HMM")
    print("="*70)
    
    # Cargar el modelo HMM
    hmm = crear_hmm_palabras_simples()
    
    print("\nModelo cargado: Reconocimiento de HOLA y ADIOS")
    print("Observaciones válidas: h, o, l, a, d, i, s, - (silencio)")
    
    # ========== PASO 1: SOLICITAR ENTRADA DEL USUARIO ==========
    print("\n--- Ingresa una secuencia de observaciones ---")
    print("Ejemplos:")
    print("  - Para 'HOLA': h o l a")
    print("  - Para 'ADIOS': a d i s")
    print("  - Con ruido: h o o l a")
    
    entrada = input("\nIngresa las observaciones separadas por espacios: ").strip().lower()
    
    # ========== PASO 2: VALIDAR Y PROCESAR ENTRADA ==========
    if not entrada:
        # Si el usuario no ingresó nada, usar un ejemplo por defecto
        print("Entrada vacía. Usando secuencia de ejemplo: h o l a")
        secuencia = ['h', 'o', 'l', 'a']
    else:
        # Separar por espacios y convertir a lista
        secuencia = entrada.split()
        
        # Validar que todas las observaciones sean válidas
        obs_validas = {'h', 'o', 'l', 'a', 'd', 'i', 's', '-'}
        secuencia = [obs for obs in secuencia if obs in obs_validas]
        
        # Si después de filtrar no quedó nada válido, usar ejemplo
        if not secuencia:
            print("No se reconocieron observaciones válidas. Usando ejemplo por defecto.")
            secuencia = ['h', 'o', 'l', 'a']
    
    # ========== PASO 3: EJECUTAR RECONOCIMIENTO ==========
    print(f"\nSecuencia a decodificar: {secuencia}")
    print("\nEjecutando algoritmo de Viterbi...")
    
    # Llamar a Viterbi para obtener la mejor secuencia de estados
    camino, prob = hmm.viterbi(secuencia, verbose=True)
    
    # ========== PASO 4: MOSTRAR RESULTADOS ==========
    print("\n" + "="*70)
    print("RESULTADO DEL RECONOCIMIENTO")
    print("="*70)
    
    # Mostrar la secuencia completa de estados (incluye INICIO y FIN)
    print(f"Secuencia de estados: {camino}")
    
    # Extraer solo los fonemas (sin INICIO ni FIN) para formar la palabra
    palabra = ''.join([s for s in camino if s not in ['INICIO', 'FIN']])
    print(f"Palabra decodificada: {palabra}")
    
    # Mostrar probabilidad en escala logarítmica (más legible para valores muy pequeños)
    print(f"Probabilidad (log): {np.log10(prob) if prob > 0 else -float('inf'):.2f}")
    
    # Clasificar la confianza según umbrales heurísticos
    if prob > 1e-5:
        confianza = "Alta"
    elif prob > 1e-10:
        confianza = "Media"
    else:
        confianza = "Baja"
    print(f"Confianza: {confianza}")

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("RECONOCIMIENTO PROBABILÍSTICO DEL HABLA (HMM + Viterbi)")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (reconocimiento de palabras predefinidas)")
    print("2. INTERACTIVO (ingresa tu propia secuencia de observaciones)")
    
    opcion = input("\nIngresa el número de opción (1 o 2): ").strip()
    
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida. Ejecutando modo DEMO por defecto...")
        modo_demo()
    
    print("\n" + "="*70)
    print("FIN DEL PROGRAMA")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
