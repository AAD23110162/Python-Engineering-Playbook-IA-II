"""
029-E2-reconocimiento_del_habla.py
--------------------------------
Este script presenta Reconocimiento del Habla desde el enfoque probabilístico:
- Modela señales acústicas como secuencias de observaciones ruidosas.
- Aplica HMM/DBN para alineación y decodificación (Viterbi, forward-backward).
- Discute extracción de características (MFCC) a nivel conceptual.

El programa puede ejecutarse en dos modos:
1. DEMO: decodificación de palabras simples con HMM de juguete.
2. INTERACTIVO: ingreso de secuencias de observaciones simbólicas y decodificación.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple, Dict

# ============================================================================
# Modelo HMM para Reconocimiento de Habla
# ============================================================================

class HMMHabla:
    """
    Modelo Oculto de Markov para reconocimiento del habla.
    - Estados ocultos: fonemas o palabras
    - Observaciones: características acústicas (MFCC simulados como símbolos)
    """
    
    def __init__(self, estados: List[str], observaciones: List[str],
                 matriz_transicion: Dict[str, Dict[str, float]],
                 matriz_emision: Dict[str, Dict[str, float]],
                 probabilidad_inicial: Dict[str, float]):
        """
        Inicializa el HMM para habla.
        
        Args:
            estados: Lista de estados ocultos (fonemas/palabras)
            observaciones: Lista de símbolos observables (características acústicas)
            matriz_transicion: T[s1][s2] = P(s2 | s1)
            matriz_emision: E[s][o] = P(o | s)
            probabilidad_inicial: pi[s] = P(s_0 = s)
        """
        # Estados y observaciones del modelo
        self.estados = estados
        self.observaciones = observaciones
        
        # Parámetros del HMM
        self.T = matriz_transicion  # Probabilidades de transición entre estados
        self.E = matriz_emision      # Probabilidades de emisión de observaciones
        self.pi = probabilidad_inicial  # Distribución inicial de estados
    
    def viterbi(self, secuencia_obs: List[str]) -> Tuple[List[str], float]:
        """
        Algoritmo de Viterbi para encontrar la secuencia de estados más probable.
        
        Args:
            secuencia_obs: Secuencia de observaciones acústicas
            
        Returns:
            (mejor_camino, probabilidad_log): Secuencia de estados y su log-probabilidad
        """
        n = len(secuencia_obs)
        # Nota: operamos en log-espacio para estabilidad numérica (evita underflow)
        
        # Delta[t][s] = máxima probabilidad de llegar a estado s en tiempo t
        delta = [{} for _ in range(n)]
        # Psi[t][s] = mejor predecesor para estado s en tiempo t
        psi = [{} for _ in range(n)]
        
        # Inicialización (t=0)
        # delta_0(s) = pi(s) * E(s, obs_0)
        obs0 = secuencia_obs[0]
        for s in self.estados:
            # Usar max para evitar log(0)
            prob_inicial = max(self.pi[s], 1e-10)
            prob_emision = max(self.E[s].get(obs0, 1e-10), 1e-10)
            delta[0][s] = math.log(prob_inicial) + math.log(prob_emision)
            psi[0][s] = None
        
        # Recursión (t=1..n-1)
        # delta_t(s) = max_{s'} [delta_{t-1}(s') * T(s', s)] * E(s, obs_t)
        for t in range(1, n):
            obs_t = secuencia_obs[t]
            for s in self.estados:
                # Encontrar el mejor predecesor
                max_prob = float('-inf')
                mejor_prev = None
                
                for s_prev in self.estados:
                    # Probabilidad de llegar a s desde s_prev
                    prob_trans = max(self.T[s_prev].get(s, 1e-10), 1e-10)
                    prob = delta[t-1][s_prev] + math.log(prob_trans)
                    if prob > max_prob:
                        max_prob = prob
                        mejor_prev = s_prev
                
                # Actualizar delta y psi
                prob_emision = max(self.E[s].get(obs_t, 1e-10), 1e-10)
                delta[t][s] = max_prob + math.log(prob_emision)
                psi[t][s] = mejor_prev
        
        # Terminación: encontrar el mejor estado final
        max_prob_final = float('-inf')
        mejor_estado_final = None
        
        for s in self.estados:
            if delta[n-1][s] > max_prob_final:
                max_prob_final = delta[n-1][s]
                mejor_estado_final = s
        
        # Backtracking: reconstruir el camino óptimo
        camino = [mejor_estado_final]
        for t in range(n-1, 0, -1):
            camino.insert(0, psi[t][camino[0]])
        
        return camino, max_prob_final
    
    def forward(self, secuencia_obs: List[str]) -> Tuple[List[Dict[str, float]], float]:
        """
        Algoritmo forward para calcular la probabilidad de la secuencia.
        
        Args:
            secuencia_obs: Secuencia de observaciones
            
        Returns:
            (alpha, log_prob): Mensajes forward y log-probabilidad total
        """
        n = len(secuencia_obs)
        # Notas sobre estabilidad:
        # - Usamos factores de escala por tiempo para mantener alphas normalizados
        # - La log-verosimilitud se obtiene sumando los logs de esos factores
        
        # Alpha[t][s] = P(obs_0..obs_t, estado_t = s)
        alpha = [{} for _ in range(n)]
        factores_escala = []  # Para evitar underflow
        
        # Inicialización (t=0)
        obs0 = secuencia_obs[0]
        suma = 0.0
        for s in self.estados:
            alpha[0][s] = self.pi[s] * self.E[s].get(obs0, 1e-10)
            suma += alpha[0][s]
        
        # Escalar para evitar underflow
        factores_escala.append(suma)
        for s in self.estados:
            alpha[0][s] /= suma
        
        # Recursión (t=1..n-1)
        for t in range(1, n):
            obs_t = secuencia_obs[t]
            suma = 0.0
            
            for s in self.estados:
                # alpha_t(s) = sum_{s'} alpha_{t-1}(s') * T(s', s) * E(s, obs_t)
                alpha[t][s] = 0.0
                for s_prev in self.estados:
                    alpha[t][s] += alpha[t-1][s_prev] * self.T[s_prev].get(s, 1e-10)
                alpha[t][s] *= self.E[s].get(obs_t, 1e-10)
                suma += alpha[t][s]
            
            # Escalar
            factores_escala.append(suma)
            for s in self.estados:
                alpha[t][s] /= suma
        
        # Calcular log-probabilidad total
        log_prob = sum(math.log(c) for c in factores_escala)
        
        return alpha, log_prob

# ============================================================================
# Reconocedor de Palabras Simple
# ============================================================================

class ReconocedorPalabras:
    """
    Reconocedor de palabras usando múltiples HMMs (uno por palabra).
    """
    
    def __init__(self):
        """Inicializa el diccionario de modelos de palabras."""
        # Diccionario de HMMs: palabra -> modelo
        self.modelos = {}
    
    def agregar_palabra(self, palabra: str, modelo: HMMHabla):
        """
        Agrega un modelo HMM para una palabra.
        
        Args:
            palabra: Etiqueta de la palabra
            modelo: HMM correspondiente a esa palabra
        """
        self.modelos[palabra] = modelo
    
    def reconocer(self, secuencia_obs: List[str]) -> Tuple[str, float]:
        """
        Reconoce la palabra más probable dada la secuencia de observaciones.
        
        Args:
            secuencia_obs: Secuencia de características acústicas
            
        Returns:
            (palabra_reconocida, log_probabilidad)
        """
        # Recorremos cada HMM de palabra y nos quedamos con el de mayor log-verosimilitud
        mejor_palabra = None
        mejor_log_prob = float('-inf')
        
        # Evaluar cada modelo de palabra
        for palabra, modelo in self.modelos.items():
            # Calcular P(observaciones | palabra) usando forward
            _, log_prob = modelo.forward(secuencia_obs)
            
            # Seleccionar el modelo con mayor probabilidad
            if log_prob > mejor_log_prob:
                mejor_log_prob = log_prob
                mejor_palabra = palabra
        
        return mejor_palabra, mejor_log_prob

# ============================================================================
# Funciones de Utilidad
# ============================================================================

def simular_mfcc(texto: str, ruido: float = 0.2) -> List[str]:
    """
    Simula la extracción de características MFCC de una señal de audio.
    En este modelo simplificado, cada letra se mapea a un símbolo acústico
    con posible ruido.
    
    Args:
        texto: Texto a "pronunciar"
        ruido: Probabilidad de error en la observación
        
    Returns:
        Secuencia de símbolos acústicos observados
    """
    # Mapeo simplificado: letra -> símbolo acústico (sustituto de vectores MFCC)
    mapeo = {
        'h': 'H1', 'o': 'O1', 'l': 'L1', 'a': 'A1',
        's': 'S1', 'i': 'I1', 'n': 'N1'
    }
    
    # Símbolos de ruido (observaciones fuera del patrón ideal)
    simbolos_ruido = ['X1', 'X2', 'X3']
    
    observaciones = []
    for letra in texto.lower():
        if random.random() < ruido:
            # Observación ruidosa
            observaciones.append(random.choice(simbolos_ruido))
        else:
            # Observación correcta
            observaciones.append(mapeo.get(letra, 'X1'))
    
    return observaciones

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra reconocimiento de habla con HMMs simples."""
    print("MODO DEMO: Reconocimiento del Habla con HMM\n")
    
    # Definir vocabulario de observaciones acústicas
    obs_vocab = ['H1', 'O1', 'L1', 'A1', 'S1', 'I1', 'N1', 'X1', 'X2', 'X3']
    
    # ========================================
    # Modelo para la palabra "hola"
    # ========================================
    estados_hola = ['H', 'O', 'L', 'A']
    
    # Matriz de transición: secuencial con auto-loops
    T_hola = {
        'H': {'H': 0.3, 'O': 0.7},
        'O': {'O': 0.3, 'L': 0.7},
        'L': {'L': 0.3, 'A': 0.7},
        'A': {'A': 1.0}
    }
    
    # Matriz de emisión: cada estado emite principalmente su símbolo
    E_hola = {
        'H': {'H1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1},
        'O': {'O1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1},
        'L': {'L1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1},
        'A': {'A1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1}
    }
    
    # Distribución inicial
    pi_hola = {'H': 1.0, 'O': 0.0, 'L': 0.0, 'A': 0.0}
    
    modelo_hola = HMMHabla(estados_hola, obs_vocab, T_hola, E_hola, pi_hola)
    
    # ========================================
    # Modelo para la palabra "si"
    # ========================================
    estados_si = ['S', 'I']
    
    T_si = {
        'S': {'S': 0.3, 'I': 0.7},
        'I': {'I': 1.0}
    }
    
    E_si = {
        'S': {'S1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1},
        'I': {'I1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1}
    }
    
    pi_si = {'S': 1.0, 'I': 0.0}
    
    modelo_si = HMMHabla(estados_si, obs_vocab, T_si, E_si, pi_si)
    
    # ========================================
    # Modelo para la palabra "no"
    # ========================================
    estados_no = ['N', 'O']
    
    T_no = {
        'N': {'N': 0.3, 'O': 0.7},
        'O': {'O': 1.0}
    }
    
    E_no = {
        'N': {'N1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1},
        'O': {'O1': 0.7, 'X1': 0.1, 'X2': 0.1, 'X3': 0.1}
    }
    
    pi_no = {'N': 1.0, 'O': 0.0}
    
    modelo_no = HMMHabla(estados_no, obs_vocab, T_no, E_no, pi_no)
    
    # ========================================
    # Crear reconocedor y registrar los HMM por palabra
    # ========================================
    reconocedor = ReconocedorPalabras()
    reconocedor.agregar_palabra("hola", modelo_hola)
    reconocedor.agregar_palabra("si", modelo_si)
    reconocedor.agregar_palabra("no", modelo_no)
    
    # ========================================
    # Pruebas de reconocimiento
    # ========================================
    palabras_prueba = ["hola", "si", "no"]
    
    print("Reconocimiento de palabras con observaciones ruidosas:\n")
    
    for palabra_real in palabras_prueba:
        # Simular observaciones acústicas con ruido
        observaciones = simular_mfcc(palabra_real, ruido=0.15)
        
        # Reconocer
        palabra_reconocida, log_prob = reconocedor.reconocer(observaciones)
        
        print(f"Palabra real: '{palabra_real}'")
        print(f"  Observaciones MFCC: {observaciones}")
        print(f"  Palabra reconocida: '{palabra_reconocida}'")
        print(f"  Log-probabilidad: {log_prob:.4f}")
        
        # Usar Viterbi para encontrar la secuencia de estados más probable
        modelo_reconocido = reconocedor.modelos[palabra_reconocida]
        camino, prob_viterbi = modelo_reconocido.viterbi(observaciones)
        print(f"  Camino Viterbi: {' -> '.join(camino)}")
        print(f"  Log-prob Viterbi: {prob_viterbi:.4f}")
        print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario ingresar secuencias de observaciones y decodificarlas."""
    print("MODO INTERACTIVO: Reconocimiento del Habla\n")
    
    # Configurar un modelo HMM simple para demostración
    print("Configurando un HMM simple para la palabra 'hola'...")
    
    estados = ['H', 'O', 'L', 'A']
    obs_vocab = ['H1', 'O1', 'L1', 'A1', 'X1', 'X2']
    
    T = {
        'H': {'H': 0.3, 'O': 0.7},
        'O': {'O': 0.3, 'L': 0.7},
        'L': {'L': 0.3, 'A': 0.7},
        'A': {'A': 1.0}
    }
    
    E = {
        'H': {'H1': 0.7, 'X1': 0.1, 'X2': 0.2},
        'O': {'O1': 0.7, 'X1': 0.1, 'X2': 0.2},
        'L': {'L1': 0.7, 'X1': 0.1, 'X2': 0.2},
        'A': {'A1': 0.7, 'X1': 0.1, 'X2': 0.2}
    }
    
    pi = {'H': 1.0, 'O': 0.0, 'L': 0.0, 'A': 0.0}
    
    modelo = HMMHabla(estados, obs_vocab, T, E, pi)
    
    print(f"Estados: {estados}")
    print(f"Observaciones posibles: {obs_vocab}")
    print()
    
    # Solicitar secuencia de observaciones
    print("Ingrese una secuencia de observaciones separadas por espacios")
    print(f"(ejemplo: H1 O1 L1 A1):")
    entrada = input("> ").strip()
    
    if entrada:
        observaciones = entrada.split()
        
        # Validar observaciones: advertir si hay símbolos fuera del vocabulario
        for obs in observaciones:
            if obs not in obs_vocab:
                print(f"Advertencia: '{obs}' no está en el vocabulario de observaciones.")
        
        # Aplicar Viterbi
        print(f"\nDecodificando secuencia: {observaciones}")
        camino, log_prob = modelo.viterbi(observaciones)
        
        print(f"Secuencia de estados más probable: {' -> '.join(camino)}")
        print(f"Log-probabilidad: {log_prob:.4f}")
        
    # Aplicar forward para obtener la log-verosimilitud normalizada (escalada)
        alpha, log_prob_forward = modelo.forward(observaciones)
        print(f"\nLog-probabilidad (forward): {log_prob_forward:.4f}")
    else:
        print("No se ingresaron observaciones. Usando ejemplo por defecto.")
        observaciones = ['H1', 'O1', 'L1', 'A1']
        camino, log_prob = modelo.viterbi(observaciones)
        print(f"Secuencia de observaciones: {observaciones}")
        print(f"Secuencia de estados más probable: {' -> '.join(camino)}")
        print(f"Log-probabilidad: {log_prob:.4f}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("029-E2: Reconocimiento del Habla")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Reconocimiento de palabras con HMMs")
    print("2. INTERACTIVO: Decodificar secuencias personalizadas")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
