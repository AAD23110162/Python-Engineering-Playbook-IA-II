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
        T = len(secuencia_observaciones)
        N = len(self.estados)
        
        # Matrices de Viterbi y backpointer
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        
        if verbose:
            print(f"\n--- Algoritmo de Viterbi ---")
            print(f"Secuencia de observaciones: {secuencia_observaciones}")
            print(f"Estados: {self.estados}")
        
        # Inicialización (t=0)
        obs_0 = secuencia_observaciones[0]
        for i, estado in enumerate(self.estados):
            prob_inicial = self.prob_iniciales.get(estado, 1.0/N)
            prob_emision = self.emisiones.get((estado, obs_0), 1e-10)
            viterbi[i, 0] = prob_inicial * prob_emision
            
            if verbose:
                print(f"\nt=0, Estado={estado}:")
                print(f"  π({estado}) × b({estado},{obs_0}) = {prob_inicial:.4f} × {prob_emision:.4f} = {viterbi[i,0]:.6f}")
        
        # Recursión (t=1 hasta T-1)
        for t in range(1, T):
            obs_t = secuencia_observaciones[t]
            
            if verbose:
                print(f"\n--- t={t}, Observación={obs_t} ---")
            
            for j, estado_j in enumerate(self.estados):
                prob_emision = self.emisiones.get((estado_j, obs_t), 1e-10)
                
                # Encontrar el máximo entre todos los estados anteriores
                prob_max = -1
                estado_max = -1
                
                for i, estado_i in enumerate(self.estados):
                    prob_trans = self.transiciones.get((estado_i, estado_j), 1e-10)
                    prob = viterbi[i, t-1] * prob_trans * prob_emision
                    
                    if prob > prob_max:
                        prob_max = prob
                        estado_max = i
                
                viterbi[j, t] = prob_max
                backpointer[j, t] = estado_max
                
                if verbose:
                    print(f"  Estado {estado_j}: max_prob={prob_max:.6f} (desde {self.estados[estado_max]})")
        
        # Terminación: encontrar el estado final con mayor probabilidad
        ultimo_estado = np.argmax(viterbi[:, T-1])
        prob_maxima = viterbi[ultimo_estado, T-1]
        
        # Backtracking para obtener el mejor camino
        mejor_camino = [0] * T
        mejor_camino[T-1] = ultimo_estado
        
        for t in range(T-2, -1, -1):
            mejor_camino[t] = backpointer[mejor_camino[t+1], t+1]
        
        # Convertir índices a nombres de estados
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
    estados = ['H', 'O', 'L', 'A', 'D', 'I', 'S', 'INICIO', 'FIN']
    observaciones = ['h', 'o', 'l', 'a', 'd', 'i', 's', '-']
    
    # Probabilidades iniciales (desde INICIO)
    prob_iniciales = {
        'INICIO': 1.0,
        'H': 0.0, 'O': 0.0, 'L': 0.0, 'A': 0.0,
        'D': 0.0, 'I': 0.0, 'S': 0.0, 'FIN': 0.0
    }
    
    # Transiciones (diseñadas para secuencias HOLA y ADIOS)
    transiciones = {
        # Desde INICIO
        ('INICIO', 'H'): 0.5,  # Empieza HOLA
        ('INICIO', 'A'): 0.5,  # Empieza ADIOS
        
        # Secuencia HOLA
        ('H', 'O'): 0.9,
        ('O', 'L'): 0.9,
        ('L', 'A'): 0.9,
        ('A', 'FIN'): 0.5,  # Puede terminar aquí (HOLA) o seguir (ADIOS empieza con A)
        
        # Secuencia ADIOS (A-D-I-O-S)
        ('A', 'D'): 0.4,
        ('D', 'I'): 0.9,
        ('I', 'O'): 0.8,  # Reutiliza O
        ('O', 'S'): 0.1,  # Alternativa desde O
        ('I', 'S'): 0.2,  # Atajo (simplificación)
        ('S', 'FIN'): 0.9,
        
        # Auto-transiciones (ruido)
        ('H', 'H'): 0.05,
        ('O', 'O'): 0.05,
        ('L', 'L'): 0.05,
        ('A', 'A'): 0.05,
        ('D', 'D'): 0.05,
        ('I', 'I'): 0.05,
        ('S', 'S'): 0.05,
    }
    
    # Emisiones (estado → observación)
    # Cada fonema emite principalmente su símbolo correspondiente con ruido
    emisiones = {
        ('H', 'h'): 0.85, ('H', 'o'): 0.05, ('H', '-'): 0.10,
        ('O', 'o'): 0.85, ('O', 'h'): 0.05, ('O', 'l'): 0.05, ('O', '-'): 0.05,
        ('L', 'l'): 0.85, ('L', 'o'): 0.05, ('L', 'a'): 0.05, ('L', '-'): 0.05,
        ('A', 'a'): 0.85, ('A', 'o'): 0.05, ('A', 'd'): 0.05, ('A', '-'): 0.05,
        ('D', 'd'): 0.85, ('D', 'a'): 0.05, ('D', 'i'): 0.05, ('D', '-'): 0.05,
        ('I', 'i'): 0.85, ('I', 'a'): 0.05, ('I', 's'): 0.05, ('I', '-'): 0.05,
        ('S', 's'): 0.85, ('S', 'i'): 0.05, ('S', '-'): 0.10,
        ('INICIO', 'h'): 0.4, ('INICIO', 'a'): 0.4, ('INICIO', '-'): 0.2,
        ('FIN', '-'): 1.0,
    }
    
    return ModeloOcultoMarkov(estados, observaciones, prob_iniciales, transiciones, emisiones)

def modo_demo():
    """Ejecuta el modo demostrativo con reconocimiento de palabras predefinidas."""
    print("\n" + "="*70)
    print("MODO DEMO: Reconocimiento del Habla con HMM")
    print("="*70)
    
    hmm = crear_hmm_palabras_simples()
    
    print("\n--- Modelo HMM ---")
    print("Palabras reconocibles: HOLA, ADIOS")
    print("Estados (fonemas): H, O, L, A, D, I, S")
    print("Observaciones acústicas: h, o, l, a, d, i, s, - (silencio)")
    
    # Ejemplo 1: Reconocer "HOLA"
    print("\n" + "="*70)
    print("EJEMPLO 1: Reconocer la palabra 'HOLA'")
    print("="*70)
    
    secuencia_hola = ['h', 'o', 'l', 'a']
    print(f"\nSecuencia de observaciones: {secuencia_hola}")
    
    camino, prob = hmm.viterbi(secuencia_hola, verbose=True)
    
    print(f"\n>>> PALABRA RECONOCIDA: {''.join(camino)}")
    print(f"    Confianza: {prob:.8f}")
    
    # Ejemplo 2: Reconocer "HOLA" con ruido
    print("\n" + "="*70)
    print("EJEMPLO 2: Reconocer 'HOLA' con ruido acústico")
    print("="*70)
    
    secuencia_hola_ruido = ['h', 'o', 'o', 'l', 'a']  # 'o' duplicada (ruido)
    print(f"\nSecuencia de observaciones (con ruido): {secuencia_hola_ruido}")
    
    camino2, prob2 = hmm.viterbi(secuencia_hola_ruido, verbose=True)
    
    print(f"\n>>> PALABRA RECONOCIDA: {''.join(camino2)}")
    print(f"    Confianza: {prob2:.8f}")
    print(f"    (El algoritmo maneja el ruido y recupera la secuencia más probable)")
    
    # Ejemplo 3: Secuencia ambigua
    print("\n" + "="*70)
    print("EJEMPLO 3: Secuencia corta ambigua")
    print("="*70)
    
    secuencia_corta = ['a', 'd']
    print(f"\nSecuencia de observaciones: {secuencia_corta}")
    
    camino3, prob3 = hmm.viterbi(secuencia_corta, verbose=True)
    
    print(f"\n>>> DECODIFICACIÓN: {''.join(camino3)}")
    print(f"    Confianza: {prob3:.8f}")
    print(f"    (Inicio de 'ADIOS' detectado)")

def modo_interactivo():
    """Ejecuta el modo interactivo con secuencias definidas por el usuario."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Reconocimiento del Habla con HMM")
    print("="*70)
    
    hmm = crear_hmm_palabras_simples()
    
    print("\nModelo cargado: Reconocimiento de HOLA y ADIOS")
    print("Observaciones válidas: h, o, l, a, d, i, s, - (silencio)")
    
    print("\n--- Ingresa una secuencia de observaciones ---")
    print("Ejemplos:")
    print("  - Para 'HOLA': h o l a")
    print("  - Para 'ADIOS': a d i s")
    print("  - Con ruido: h o o l a")
    
    entrada = input("\nIngresa las observaciones separadas por espacios: ").strip().lower()
    
    if not entrada:
        print("Entrada vacía. Usando secuencia de ejemplo: h o l a")
        secuencia = ['h', 'o', 'l', 'a']
    else:
        secuencia = entrada.split()
        # Validar observaciones
        obs_validas = {'h', 'o', 'l', 'a', 'd', 'i', 's', '-'}
        secuencia = [obs for obs in secuencia if obs in obs_validas]
        
        if not secuencia:
            print("No se reconocieron observaciones válidas. Usando ejemplo por defecto.")
            secuencia = ['h', 'o', 'l', 'a']
    
    print(f"\nSecuencia a decodificar: {secuencia}")
    print("\nEjecutando algoritmo de Viterbi...")
    
    camino, prob = hmm.viterbi(secuencia, verbose=True)
    
    print("\n" + "="*70)
    print("RESULTADO DEL RECONOCIMIENTO")
    print("="*70)
    print(f"Secuencia de estados: {camino}")
    print(f"Palabra decodificada: {''.join([s for s in camino if s not in ['INICIO', 'FIN']])}")
    print(f"Probabilidad (log): {np.log10(prob) if prob > 0 else -float('inf'):.2f}")
    print(f"Confianza: {'Alta' if prob > 1e-5 else 'Media' if prob > 1e-10 else 'Baja'}")

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
