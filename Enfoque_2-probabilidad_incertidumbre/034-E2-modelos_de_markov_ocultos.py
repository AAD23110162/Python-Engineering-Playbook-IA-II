"""
034-E2-modelos_de_markov_ocultos.py
--------------------------------
Este script profundiza en Modelos de Markov Ocultos (HMM):
- Decodificación con Viterbi y suavizado con forward-backward.
- Entrenamiento de parámetros con Baum-Welch (EM) a nivel conceptual.
- Discute problemas de escalado y regularización.

El programa puede ejecutarse en dos modos:
1. DEMO: entrenamiento y decodificación en un HMM simple.
2. INTERACTIVO: permite definir estructura, inicializar parámetros y entrenar.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Dict, Tuple
from collections import defaultdict

# ============================================================================
# Modelo Oculto de Markov con Baum-Welch
# ============================================================================

class HMMBaumWelch:
    """
    Modelo Oculto de Markov con algoritmo Baum-Welch para entrenamiento.
    """
    
    def __init__(self, estados: List[str], observaciones: List[str]):
        """
        Inicializa el HMM.
        
        Args:
            estados: Lista de estados ocultos
            observaciones: Lista de símbolos observables
        """
        self.estados = estados
        self.observaciones = observaciones
        
        # Parámetros del HMM (se inicializarán aleatoriamente)
        self.T = {}  # Matriz de transición T[s1][s2] = P(s2 | s1)
        self.E = {}  # Matriz de emisión E[s][o] = P(o | s)
        self.pi = {}  # Distribución inicial pi[s] = P(s_0 = s)
        
        # Inicializar parámetros uniformes
        self._inicializar_uniforme()
    
    def _inicializar_uniforme(self):
        """Inicializa parámetros con distribuciones uniformes."""
        n_estados = len(self.estados)
        n_obs = len(self.observaciones)
        
        # Distribución inicial uniforme
        for s in self.estados:
            self.pi[s] = 1.0 / n_estados
        
        # Transiciones uniformes
        for s1 in self.estados:
            self.T[s1] = {}
            for s2 in self.estados:
                self.T[s1][s2] = 1.0 / n_estados
        
        # Emisiones uniformes
        for s in self.estados:
            self.E[s] = {}
            for o in self.observaciones:
                self.E[s][o] = 1.0 / n_obs
    
    def _forward(self, secuencia: List[str]) -> Tuple[List[Dict[str, float]], List[float]]:
        """
        Algoritmo forward con escalado.
        
        Returns:
            (alpha, factores_escala)
        """
        n = len(secuencia)
        alpha = [{} for _ in range(n)]
        factores_escala = []
        
        # Inicialización
        suma = 0.0
        for s in self.estados:
            alpha[0][s] = self.pi[s] * self.E[s].get(secuencia[0], 1e-10)
            suma += alpha[0][s]
        
        # Escalar
        factores_escala.append(suma if suma > 0 else 1.0)
        for s in self.estados:
            alpha[0][s] /= factores_escala[0]
        
        # Recursión
        for t in range(1, n):
            suma = 0.0
            for s in self.estados:
                alpha[t][s] = 0.0
                for s_prev in self.estados:
                    alpha[t][s] += alpha[t-1][s_prev] * self.T[s_prev].get(s, 1e-10)
                alpha[t][s] *= self.E[s].get(secuencia[t], 1e-10)
                suma += alpha[t][s]
            
            # Escalar
            factores_escala.append(suma if suma > 0 else 1.0)
            for s in self.estados:
                alpha[t][s] /= factores_escala[t]
        
        return alpha, factores_escala
    
    def _backward(self, secuencia: List[str], factores_escala: List[float]) -> List[Dict[str, float]]:
        """
        Algoritmo backward con escalado compatible con forward.
        
        Returns:
            beta
        """
        n = len(secuencia)
        beta = [{} for _ in range(n)]
        
        # Inicialización (t = n-1)
        for s in self.estados:
            beta[n-1][s] = 1.0 / factores_escala[n-1]
        
        # Recursión hacia atrás
        for t in range(n-2, -1, -1):
            for s in self.estados:
                beta[t][s] = 0.0
                for s_next in self.estados:
                    beta[t][s] += (self.T[s].get(s_next, 1e-10) * 
                                  self.E[s_next].get(secuencia[t+1], 1e-10) * 
                                  beta[t+1][s_next])
                beta[t][s] /= factores_escala[t]
        
        return beta
    
    def _calcular_gamma_xi(self, secuencia: List[str], 
                           alpha: List[Dict[str, float]], 
                           beta: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[Dict[Tuple[str, str], float]]]:
        """
        Calcula probabilidades γ (estado en tiempo t) y ξ (par de estados en t y t+1).
        
        Returns:
            (gamma, xi)
        """
        n = len(secuencia)
        
        # γ_t(s) = P(X_t = s | observaciones)
        gamma = []
        
        for t in range(n):
            gamma_t = {}
            suma = 0.0
            for s in self.estados:
                gamma_t[s] = alpha[t][s] * beta[t][s]
                suma += gamma_t[s]
            
            # Normalizar
            if suma > 0:
                for s in self.estados:
                    gamma_t[s] /= suma
            
            gamma.append(gamma_t)
        
        # ξ_t(s1, s2) = P(X_t = s1, X_{t+1} = s2 | observaciones)
        xi = []
        
        for t in range(n-1):
            xi_t = {}
            suma = 0.0
            
            for s1 in self.estados:
                for s2 in self.estados:
                    valor = (alpha[t][s1] * 
                            self.T[s1].get(s2, 1e-10) * 
                            self.E[s2].get(secuencia[t+1], 1e-10) * 
                            beta[t+1][s2])
                    xi_t[(s1, s2)] = valor
                    suma += valor
            
            # Normalizar
            if suma > 0:
                for key in xi_t:
                    xi_t[key] /= suma
            
            xi.append(xi_t)
        
        return gamma, xi
    
    def entrenar_baum_welch(self, secuencias: List[List[str]], 
                           max_iter: int = 50, 
                           tolerancia: float = 1e-4,
                           verbose: bool = True) -> int:
        """
        Entrena el HMM usando el algoritmo Baum-Welch (EM para HMMs).
        
        Args:
            secuencias: Lista de secuencias de observaciones
            max_iter: Número máximo de iteraciones
            tolerancia: Criterio de convergencia
            verbose: Si imprimir progreso
            
        Returns:
            Número de iteraciones realizadas
        """
        log_lik_anterior = float('-inf')
        
        for iteracion in range(max_iter):
            # ========== Paso E ==========
            # Acumuladores para estadísticas
            gamma_sum = defaultdict(float)  # Σ_t γ_t(s)
            gamma_sum_inicial = defaultdict(float)  # γ_0(s)
            xi_sum = defaultdict(float)  # Σ_t ξ_t(s1, s2)
            gamma_obs_sum = defaultdict(lambda: defaultdict(float))  # Σ_t γ_t(s) * I(o_t = o)
            
            log_lik_total = 0.0
            
            for secuencia in secuencias:
                # Forward-backward
                alpha, factores_escala = self._forward(secuencia)
                beta = self._backward(secuencia, factores_escala)
                gamma, xi = self._calcular_gamma_xi(secuencia, alpha, beta)
                
                # Acumular log-likelihood
                log_lik_total += sum(math.log(c) for c in factores_escala if c > 0)
                
                # Acumular estadísticas
                n = len(secuencia)
                
                # Gamma inicial
                for s in self.estados:
                    gamma_sum_inicial[s] += gamma[0][s]
                
                # Gamma total
                for t in range(n):
                    for s in self.estados:
                        gamma_sum[s] += gamma[t][s]
                        # Acumular emisiones
                        gamma_obs_sum[s][secuencia[t]] += gamma[t][s]
                
                # Xi
                for t in range(n-1):
                    for s1 in self.estados:
                        for s2 in self.estados:
                            xi_sum[(s1, s2)] += xi[t].get((s1, s2), 0.0)
            
            # Verificar convergencia
            if verbose and (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1}: Log-likelihood = {log_lik_total:.4f}")
            
            mejora = log_lik_total - log_lik_anterior
            if abs(mejora) < tolerancia and iteracion > 0:
                if verbose:
                    print(f"Convergencia alcanzada en iteración {iteracion + 1}")
                    print(f"Log-likelihood final = {log_lik_total:.4f}")
                return iteracion + 1
            
            log_lik_anterior = log_lik_total
            
            # ========== Paso M ==========
            # Actualizar distribución inicial
            suma_inicial = sum(gamma_sum_inicial.values())
            if suma_inicial > 0:
                for s in self.estados:
                    self.pi[s] = gamma_sum_inicial[s] / suma_inicial
            
            # Actualizar transiciones
            for s1 in self.estados:
                suma_desde_s1 = sum(xi_sum.get((s1, s2), 0.0) for s2 in self.estados)
                if suma_desde_s1 > 0:
                    for s2 in self.estados:
                        self.T[s1][s2] = xi_sum.get((s1, s2), 0.0) / suma_desde_s1
                else:
                    # Mantener uniforme si no hay transiciones
                    for s2 in self.estados:
                        self.T[s1][s2] = 1.0 / len(self.estados)
            
            # Actualizar emisiones
            for s in self.estados:
                if gamma_sum[s] > 0:
                    for o in self.observaciones:
                        self.E[s][o] = gamma_obs_sum[s].get(o, 0.0) / gamma_sum[s]
                else:
                    # Mantener uniforme si no hay emisiones
                    for o in self.observaciones:
                        self.E[s][o] = 1.0 / len(self.observaciones)
        
        if verbose:
            print(f"Máximo de iteraciones alcanzado ({max_iter})")
            print(f"Log-likelihood final = {log_lik_anterior:.4f}")
        
        return max_iter
    
    def viterbi(self, secuencia: List[str]) -> Tuple[List[str], float]:
        """
        Algoritmo de Viterbi para encontrar la secuencia de estados más probable.
        
        Returns:
            (camino_optimo, log_probabilidad)
        """
        n = len(secuencia)
        delta = [{} for _ in range(n)]
        psi = [{} for _ in range(n)]
        
        # Inicialización
        for s in self.estados:
            delta[0][s] = math.log(self.pi[s]) + math.log(self.E[s].get(secuencia[0], 1e-10))
            psi[0][s] = None
        
        # Recursión
        for t in range(1, n):
            for s in self.estados:
                max_prob = float('-inf')
                mejor_prev = None
                
                for s_prev in self.estados:
                    prob = delta[t-1][s_prev] + math.log(self.T[s_prev].get(s, 1e-10))
                    if prob > max_prob:
                        max_prob = prob
                        mejor_prev = s_prev
                
                delta[t][s] = max_prob + math.log(self.E[s].get(secuencia[t], 1e-10))
                psi[t][s] = mejor_prev
        
        # Terminación
        max_prob_final = float('-inf')
        mejor_estado_final = None
        
        for s in self.estados:
            if delta[n-1][s] > max_prob_final:
                max_prob_final = delta[n-1][s]
                mejor_estado_final = s
        
        # Backtracking
        camino = [mejor_estado_final]
        for t in range(n-1, 0, -1):
            camino.insert(0, psi[t][camino[0]])
        
        return camino, max_prob_final

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra entrenamiento y decodificación de HMM."""
    print("MODO DEMO: Modelos de Markov Ocultos con Baum-Welch\n")
    
    random.seed(42)
    
    # ========================================
    # Definir HMM verdadero (para generar datos)
    # ========================================
    print("=" * 60)
    print("Generando datos sintéticos")
    print("=" * 60)
    
    # Estados: clima (Sol, Lluvia)
    # Observaciones: actividad (Paseo, Compras, Limpiar)
    
    estados_reales = ['Sol', 'Lluvia']
    obs_reales = ['Paseo', 'Compras', 'Limpiar']
    
    # Parámetros reales
    T_real = {
        'Sol': {'Sol': 0.8, 'Lluvia': 0.2},
        'Lluvia': {'Sol': 0.3, 'Lluvia': 0.7}
    }
    
    E_real = {
        'Sol': {'Paseo': 0.6, 'Compras': 0.3, 'Limpiar': 0.1},
        'Lluvia': {'Paseo': 0.1, 'Compras': 0.4, 'Limpiar': 0.5}
    }
    
    pi_real = {'Sol': 0.6, 'Lluvia': 0.4}
    
    print("Parámetros reales del HMM:")
    print(f"  Estados: {estados_reales}")
    print(f"  Observaciones: {obs_reales}")
    print(f"  π = {pi_real}")
    print(f"  T = {T_real}")
    print(f"  E = {E_real}")
    print()
    
    # Generar secuencias
    def generar_secuencia(n):
        estados_seq = []
        obs_seq = []
        
        # Estado inicial
        r = random.random()
        estado = 'Sol' if r < pi_real['Sol'] else 'Lluvia'
        
        for _ in range(n):
            estados_seq.append(estado)
            
            # Emitir observación
            r = random.random()
            acum = 0.0
            for o, p in E_real[estado].items():
                acum += p
                if r <= acum:
                    obs_seq.append(o)
                    break
            
            # Transición
            r = random.random()
            acum = 0.0
            for s_next, p in T_real[estado].items():
                acum += p
                if r <= acum:
                    estado = s_next
                    break
        
        return estados_seq, obs_seq
    
    # Generar 10 secuencias de longitud 20
    secuencias_entrenamiento = []
    for i in range(10):
        estados_seq, obs_seq = generar_secuencia(20)
        secuencias_entrenamiento.append(obs_seq)
    
    print(f"Generadas {len(secuencias_entrenamiento)} secuencias de entrenamiento")
    print(f"Ejemplo: {secuencias_entrenamiento[0][:10]}...")
    print()
    
    # ========================================
    # Entrenar HMM con Baum-Welch
    # ========================================
    print("=" * 60)
    print("Entrenando HMM con Baum-Welch")
    print("=" * 60)
    
    modelo = HMMBaumWelch(estados_reales, obs_reales)
    iteraciones = modelo.entrenar_baum_welch(secuencias_entrenamiento, 
                                             max_iter=50, 
                                             tolerancia=1e-4, 
                                             verbose=True)
    print()
    
    # ========================================
    # Mostrar parámetros aprendidos
    # ========================================
    print("=" * 60)
    print("Parámetros aprendidos")
    print("=" * 60)
    
    print(f"Distribución inicial π:")
    for s, p in modelo.pi.items():
        print(f"  {s}: {p:.4f}")
    print()
    
    print(f"Matriz de transición T:")
    for s1 in modelo.estados:
        print(f"  {s1}:")
        for s2, p in modelo.T[s1].items():
            print(f"    -> {s2}: {p:.4f}")
    print()
    
    print(f"Matriz de emisión E:")
    for s in modelo.estados:
        print(f"  {s}:")
        for o, p in modelo.E[s].items():
            print(f"    {o}: {p:.4f}")
    print()
    
    # ========================================
    # Decodificación con Viterbi
    # ========================================
    print("=" * 60)
    print("Decodificación con Viterbi")
    print("=" * 60)
    
    # Generar nueva secuencia de prueba
    estados_test, obs_test = generar_secuencia(15)
    
    print(f"Secuencia de observaciones:")
    print(f"  {obs_test}")
    print(f"\nEstados reales:")
    print(f"  {estados_test}")
    
    # Decodificar
    camino_viterbi, log_prob = modelo.viterbi(obs_test)
    
    print(f"\nEstados decodificados (Viterbi):")
    print(f"  {camino_viterbi}")
    print(f"Log-probabilidad: {log_prob:.4f}")
    
    # Exactitud
    correctos = sum(1 for real, pred in zip(estados_test, camino_viterbi) if real == pred)
    exactitud = correctos / len(estados_test)
    print(f"Exactitud: {exactitud:.2%} ({correctos}/{len(estados_test)})")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario definir y entrenar un HMM."""
    print("MODO INTERACTIVO: Modelos de Markov Ocultos\n")
    
    print("Ingrese los estados ocultos separados por espacios:")
    estados = input("> ").strip().split()
    
    print("Ingrese las observaciones posibles separadas por espacios:")
    observaciones = input("> ").strip().split()
    
    if not estados or not observaciones:
        print("Debe definir al menos un estado y una observación.")
        return
    
    # Crear modelo
    modelo = HMMBaumWelch(estados, observaciones)
    print(f"\nModelo creado con {len(estados)} estados y {len(observaciones)} observaciones.")
    print("Parámetros inicializados uniformemente.")
    
    # Ingresar secuencias
    print("\nIngrese secuencias de observaciones para entrenamiento.")
    print("Una secuencia por línea, símbolos separados por espacios.")
    print("Escriba 'fin' cuando termine.\n")
    
    secuencias = []
    while True:
        entrada = input("> ").strip()
        if entrada.lower() == "fin":
            break
        
        secuencia = entrada.split()
        if secuencia:
            secuencias.append(secuencia)
    
    if not secuencias:
        print("No se ingresaron secuencias.")
        return
    
    print(f"\n{len(secuencias)} secuencias ingresadas.")
    
    # Entrenar
    max_iter = int(input("Número máximo de iteraciones (default=50): ") or "50")
    print("\nEntrenando...\n")
    
    modelo.entrenar_baum_welch(secuencias, max_iter=max_iter, verbose=True)
    
    # Mostrar parámetros
    print(f"\nParámetros aprendidos:")
    print(f"π = {modelo.pi}")
    print(f"T = {modelo.T}")
    print(f"E = {modelo.E}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("034-E2: Modelos de Markov Ocultos (Baum-Welch)")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Entrenamiento y decodificación")
    print("2. INTERACTIVO: Definir y entrenar HMM")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
