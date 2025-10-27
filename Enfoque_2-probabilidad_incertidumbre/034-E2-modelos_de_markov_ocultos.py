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
        
        # Distribución inicial uniforme: todos los estados tienen igual probabilidad de ser el inicial
        # π[s] = P(X_0 = s) = 1/n para todo s
        for s in self.estados:
            self.pi[s] = 1.0 / n_estados
        
        # Transiciones uniformes: desde cada estado, probabilidad igual de ir a cualquier otro
        # T[s1][s2] = P(X_t = s2 | X_{t-1} = s1) = 1/n para todo s2
        for s1 in self.estados:
            self.T[s1] = {}
            for s2 in self.estados:
                self.T[s1][s2] = 1.0 / n_estados
        
        # Emisiones uniformes: cada estado emite cada observación con igual probabilidad
        # E[s][o] = P(O_t = o | X_t = s) = 1/m para todo o
        for s in self.estados:
            self.E[s] = {}
            for o in self.observaciones:
                self.E[s][o] = 1.0 / n_obs
    
    def _forward(self, secuencia: List[str]) -> Tuple[List[Dict[str, float]], List[float]]:
        """
        Algoritmo forward con escalado para evitar underflow numérico.
        α_t(s) = P(O_1...O_t, X_t = s | λ), probabilidad de estar en s en t dado observaciones hasta t.
        
        Returns:
            (alpha, factores_escala)
        """
        n = len(secuencia)
        alpha = [{} for _ in range(n)]
        factores_escala = []
        
        # Inicialización (t=0): α_0(s) = π(s) * E[s][o_0]
        # Probabilidad de comenzar en s y emitir la primera observación
        suma = 0.0
        for s in self.estados:
            alpha[0][s] = self.pi[s] * self.E[s].get(secuencia[0], 1e-10)
            suma += alpha[0][s]
        
        # Escalar para evitar underflow: dividimos entre la suma para normalizar
        # Guardamos el factor para poder recuperar la probabilidad real después
        factores_escala.append(suma if suma > 0 else 1.0)
        for s in self.estados:
            alpha[0][s] /= factores_escala[0]
        
        # Recursión (t=1 hasta n-1): α_t(s) = [Σ_{s'} α_{t-1}(s') * T[s'][s]] * E[s][o_t]
        # Acumulamos probabilidades de todos los caminos que llegan a s en t
        for t in range(1, n):
            suma = 0.0
            for s in self.estados:
                alpha[t][s] = 0.0
                # Sumamos sobre todos los estados previos posibles
                for s_prev in self.estados:
                    # Prob. de estar en s_prev en t-1, luego transicionar a s
                    alpha[t][s] += alpha[t-1][s_prev] * self.T[s_prev].get(s, 1e-10)
                # Multiplicamos por la probabilidad de emitir la observación actual
                alpha[t][s] *= self.E[s].get(secuencia[t], 1e-10)
                suma += alpha[t][s]
            
            # Escalar para mantener valores numéricos manejables
            factores_escala.append(suma if suma > 0 else 1.0)
            for s in self.estados:
                alpha[t][s] /= factores_escala[t]
        
        return alpha, factores_escala
    
    def _backward(self, secuencia: List[str], factores_escala: List[float]) -> List[Dict[str, float]]:
        """
        Algoritmo backward con escalado compatible con forward.
        β_t(s) = P(O_{t+1}...O_n | X_t = s, λ), probabilidad de observaciones futuras dado s en t.
        
        Returns:
            beta
        """
        n = len(secuencia)
        beta = [{} for _ in range(n)]
        
        # Inicialización (t = n-1): β_{n-1}(s) = 1 para todo s
        # No hay observaciones futuras después del último paso
        for s in self.estados:
            beta[n-1][s] = 1.0 / factores_escala[n-1]
        
        # Recursión hacia atrás (t = n-2 hasta 0): β_t(s) = Σ_{s'} T[s][s'] * E[s'][o_{t+1}] * β_{t+1}(s')
        # Acumulamos probabilidades de todos los caminos futuros desde s
        for t in range(n-2, -1, -1):
            for s in self.estados:
                beta[t][s] = 0.0
                # Sumamos sobre todos los posibles estados siguientes
                for s_next in self.estados:
                    # Prob. de transicionar a s_next, emitir obs. en t+1, y seguir desde ahí
                    beta[t][s] += (self.T[s].get(s_next, 1e-10) * 
                                  self.E[s_next].get(secuencia[t+1], 1e-10) * 
                                  beta[t+1][s_next])
                # Aplicamos el mismo escalado que en forward para compatibilidad
                beta[t][s] /= factores_escala[t]
        
        return beta
    
    def _calcular_gamma_xi(self, secuencia: List[str], 
                           alpha: List[Dict[str, float]], 
                           beta: List[Dict[str, float]]) -> Tuple[List[Dict[str, float]], List[Dict[Tuple[str, str], float]]]:
        """
        Calcula probabilidades γ (estado en tiempo t) y ξ (par de estados en t y t+1).
        Estas son las estadísticas suficientes necesarias para el paso M de Baum-Welch.
        
        Returns:
            (gamma, xi)
        """
        n = len(secuencia)
        
        # γ_t(s) = P(X_t = s | observaciones completas, λ)
        # Probabilidad posterior de estar en estado s en tiempo t
        # Se calcula combinando información forward (pasado) y backward (futuro)
        gamma = []
        
        for t in range(n):
            gamma_t = {}
            suma = 0.0
            for s in self.estados:
                # α_t(s) captura el pasado, β_t(s) captura el futuro
                gamma_t[s] = alpha[t][s] * beta[t][s]
                suma += gamma_t[s]
            
            # Normalizar para asegurar que sumen 1 (distribución de probabilidad)
            if suma > 0:
                for s in self.estados:
                    gamma_t[s] /= suma
            
            gamma.append(gamma_t)
        
        # ξ_t(s1, s2) = P(X_t = s1, X_{t+1} = s2 | observaciones completas, λ)
        # Probabilidad posterior de estar en s1 en t y transicionar a s2 en t+1
        # Necesaria para reestimar la matriz de transición T
        xi = []
        
        for t in range(n-1):
            xi_t = {}
            suma = 0.0
            
            for s1 in self.estados:
                for s2 in self.estados:
                    # Combinamos 4 factores:
                    # 1. α_t(s1): probabilidad de llegar a s1 en t
                    # 2. T[s1][s2]: probabilidad de transicionar de s1 a s2
                    # 3. E[s2][o_{t+1}]: probabilidad de emitir obs. en t+1 desde s2
                    # 4. β_{t+1}(s2): probabilidad del futuro desde s2 en t+1
                    valor = (alpha[t][s1] * 
                            self.T[s1].get(s2, 1e-10) * 
                            self.E[s2].get(secuencia[t+1], 1e-10) * 
                            beta[t+1][s2])
                    xi_t[(s1, s2)] = valor
                    suma += valor
            
            # Normalizar para obtener distribución de probabilidad conjunta
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
            # ========== Paso E (Expectation) ==========
            # Calculamos las expectativas de las estadísticas suficientes
            # usando los parámetros actuales del modelo
            
            # Acumuladores para estadísticas suficientes:
            gamma_sum = defaultdict(float)  # Σ_t γ_t(s): tiempo esperado en estado s
            gamma_sum_inicial = defaultdict(float)  # γ_0(s): prob. de comenzar en s
            xi_sum = defaultdict(float)  # Σ_t ξ_t(s1, s2): transiciones esperadas s1→s2
            gamma_obs_sum = defaultdict(lambda: defaultdict(float))  # Σ_t γ_t(s) * I(o_t = o): emisiones esperadas
            
            log_lik_total = 0.0  # Log-verosimilitud para monitorear convergencia
            
            # Procesamos cada secuencia de entrenamiento independientemente
            for secuencia in secuencias:
                # Ejecutamos forward-backward para obtener α, β
                alpha, factores_escala = self._forward(secuencia)
                beta = self._backward(secuencia, factores_escala)
                # Calculamos γ y ξ a partir de α y β
                gamma, xi = self._calcular_gamma_xi(secuencia, alpha, beta)
                
                # Acumular log-verosimilitud: log P(O|λ) = Σ_t log(c_t)
                # Los factores de escala nos dan la probabilidad total
                log_lik_total += sum(math.log(c) for c in factores_escala if c > 0)
                
                # Acumular estadísticas suficientes de esta secuencia
                n = len(secuencia)
                
                # Acumular γ_0: probabilidad de cada estado en el tiempo inicial
                # Usaremos esto para reestimar π
                for s in self.estados:
                    gamma_sum_inicial[s] += gamma[0][s]
                
                # Acumular γ_t para todos los tiempos
                # Usaremos esto como denominador en las reestimaciones
                for t in range(n):
                    for s in self.estados:
                        gamma_sum[s] += gamma[t][s]
                        # Contar cuántas veces el estado s emitió esta observación
                        # Solo sumamos cuando o_t coincide con secuencia[t]
                        gamma_obs_sum[s][secuencia[t]] += gamma[t][s]
                
                # Acumular ξ_t: probabilidad de transiciones s1→s2
                # Usaremos esto para reestimar T
                for t in range(n-1):
                    for s1 in self.estados:
                        for s2 in self.estados:
                            xi_sum[(s1, s2)] += xi[t].get((s1, s2), 0.0)
            
            # Verificar convergencia monitoreando el cambio en log-verosimilitud
            if verbose and (iteracion + 1) % 10 == 0:
                print(f"Iteración {iteracion + 1}: Log-likelihood = {log_lik_total:.4f}")
            
            # Si la mejora es menor que la tolerancia, hemos convergido
            mejora = log_lik_total - log_lik_anterior
            if abs(mejora) < tolerancia and iteracion > 0:
                if verbose:
                    print(f"Convergencia alcanzada en iteración {iteracion + 1}")
                    print(f"Log-likelihood final = {log_lik_total:.4f}")
                return iteracion + 1
            
            log_lik_anterior = log_lik_total
            
            # ========== Paso M (Maximization) ==========
            # Reestimamos los parámetros del modelo usando las estadísticas calculadas en E
            
            # Actualizar distribución inicial: π(s) = γ_0(s) / Σ_s γ_0(s)
            # Frecuencia relativa de comenzar en cada estado
            suma_inicial = sum(gamma_sum_inicial.values())
            if suma_inicial > 0:
                for s in self.estados:
                    self.pi[s] = gamma_sum_inicial[s] / suma_inicial
            
            # Actualizar transiciones: T[s1][s2] = Σ_t ξ_t(s1,s2) / Σ_t γ_t(s1)
            # = (transiciones esperadas s1→s2) / (tiempo esperado en s1)
            for s1 in self.estados:
                suma_desde_s1 = sum(xi_sum.get((s1, s2), 0.0) for s2 in self.estados)
                if suma_desde_s1 > 0:
                    for s2 in self.estados:
                        self.T[s1][s2] = xi_sum.get((s1, s2), 0.0) / suma_desde_s1
                else:
                    # Caso degenerado: mantener uniforme si no hay transiciones observadas
                    for s2 in self.estados:
                        self.T[s1][s2] = 1.0 / len(self.estados)
            
            # Actualizar emisiones: E[s][o] = Σ_t γ_t(s)*I(o_t=o) / Σ_t γ_t(s)
            # = (veces que s emitió o) / (tiempo total en s)
            for s in self.estados:
                if gamma_sum[s] > 0:
                    for o in self.observaciones:
                        # Frecuencia relativa de emitir o cuando estamos en s
                        self.E[s][o] = gamma_obs_sum[s].get(o, 0.0) / gamma_sum[s]
                else:
                    # Caso degenerado: mantener uniforme si el estado nunca fue visitado
                    for o in self.observaciones:
                        self.E[s][o] = 1.0 / len(self.observaciones)
        
        if verbose:
            print(f"Máximo de iteraciones alcanzado ({max_iter})")
            print(f"Log-likelihood final = {log_lik_anterior:.4f}")
        
        return max_iter
    
    def viterbi(self, secuencia: List[str]) -> Tuple[List[str], float]:
        """
        Algoritmo de Viterbi para encontrar la secuencia de estados más probable.
        Usa programación dinámica para encontrar el camino óptimo eficientemente.
        
        Returns:
            (camino_optimo, log_probabilidad)
        """
        n = len(secuencia)
        # δ_t(s): log-probabilidad del mejor camino que termina en s en tiempo t
        delta = [{} for _ in range(n)]
        # ψ_t(s): estado predecesor óptimo para llegar a s en tiempo t
        psi = [{} for _ in range(n)]
        
        # Inicialización (t=0): δ_0(s) = log π(s) + log E[s][o_0]
        # Log-prob. de comenzar en s y emitir primera observación
        for s in self.estados:
            delta[0][s] = math.log(self.pi[s]) + math.log(self.E[s].get(secuencia[0], 1e-10))
            psi[0][s] = None  # No hay predecesor en t=0
        
        # Recursión (t=1 hasta n-1): δ_t(s) = max_{s'} [δ_{t-1}(s') + log T[s'][s]] + log E[s][o_t]
        # En cada paso, elegimos el mejor predecesor para llegar a s
        for t in range(1, n):
            for s in self.estados:
                max_prob = float('-inf')
                mejor_prev = None
                
                # Buscamos el mejor estado previo s' para llegar a s
                for s_prev in self.estados:
                    # Log-prob. del mejor camino hasta s_prev, más transición s'→s
                    prob = delta[t-1][s_prev] + math.log(self.T[s_prev].get(s, 1e-10))
                    if prob > max_prob:
                        max_prob = prob
                        mejor_prev = s_prev
                
                # Agregamos la emisión de la observación actual
                delta[t][s] = max_prob + math.log(self.E[s].get(secuencia[t], 1e-10))
                # Guardamos el predecesor óptimo para backtracking
                psi[t][s] = mejor_prev
        
        # Terminación: encontramos el estado con mayor log-probabilidad en t=n-1
        max_prob_final = float('-inf')
        mejor_estado_final = None
        
        for s in self.estados:
            if delta[n-1][s] > max_prob_final:
                max_prob_final = delta[n-1][s]
                mejor_estado_final = s
        
        # Backtracking: reconstruimos el camino óptimo siguiendo los ψ hacia atrás
        camino = [mejor_estado_final]
        for t in range(n-1, 0, -1):
            # Insertamos el predecesor óptimo al inicio del camino
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
    
    # Dominio del problema: clima oculto (Sol/Lluvia) y actividades observables
    # Estados ocultos: clima (Sol, Lluvia) - no podemos observar directamente el clima
    # Observaciones: actividad (Paseo, Compras, Limpiar) - vemos las acciones de la persona
    
    estados_reales = ['Sol', 'Lluvia']
    obs_reales = ['Paseo', 'Compras', 'Limpiar']
    
    # Parámetros reales del HMM "verdadero" (que queremos recuperar con Baum-Welch)
    # Transiciones: el clima tiende a persistir (alta prob. en diagonal)
    T_real = {
        'Sol': {'Sol': 0.8, 'Lluvia': 0.2},       # Si hay sol, 80% sigue sol
        'Lluvia': {'Sol': 0.3, 'Lluvia': 0.7}    # Si llueve, 70% sigue lluvia
    }
    
    # Emisiones: con sol se prefiere pasear, con lluvia se prefiere limpiar
    E_real = {
        'Sol': {'Paseo': 0.6, 'Compras': 0.3, 'Limpiar': 0.1},
        'Lluvia': {'Paseo': 0.1, 'Compras': 0.4, 'Limpiar': 0.5}
    }
    
    # Distribución inicial: más probable comenzar con sol
    pi_real = {'Sol': 0.6, 'Lluvia': 0.4}
    
    print("Parámetros reales del HMM:")
    print(f"  Estados: {estados_reales}")
    print(f"  Observaciones: {obs_reales}")
    print(f"  π = {pi_real}")
    print(f"  T = {T_real}")
    print(f"  E = {E_real}")
    print()
    
    # Función auxiliar para generar secuencias sintéticas usando el HMM real
    def generar_secuencia(n):
        """Simula n pasos del HMM con los parámetros reales."""
        estados_seq = []
        obs_seq = []
        
        # Muestrear estado inicial según π
        r = random.random()
        estado = 'Sol' if r < pi_real['Sol'] else 'Lluvia'
        
        for _ in range(n):
            estados_seq.append(estado)
            
            # Emitir observación según E[estado]
            # Muestreo categórico: acumulamos probabilidades
            r = random.random()
            acum = 0.0
            for o, p in E_real[estado].items():
                acum += p
                if r <= acum:
                    obs_seq.append(o)
                    break
            
            # Transicionar al siguiente estado según T[estado]
            r = random.random()
            acum = 0.0
            for s_next, p in T_real[estado].items():
                acum += p
                if r <= acum:
                    estado = s_next
                    break
        
        return estados_seq, obs_seq
    
    # Generar 10 secuencias de longitud 20 para entrenamiento
    # En aplicaciones reales, estas serían observaciones sin estados conocidos
    secuencias_entrenamiento = []
    for i in range(10):
        estados_seq, obs_seq = generar_secuencia(20)
        # Solo guardamos las observaciones (estados ocultos no se usan en entrenamiento)
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
    
    # Creamos un modelo con parámetros iniciales uniformes
    # Baum-Welch los ajustará para maximizar P(observaciones | modelo)
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
    
    # Generar nueva secuencia de prueba con estados conocidos
    estados_test, obs_test = generar_secuencia(15)
    
    print(f"Secuencia de observaciones:")
    print(f"  {obs_test}")
    print(f"\nEstados reales (verdad del terreno):")
    print(f"  {estados_test}")
    
    # Decodificar: encontrar la secuencia de estados más probable dado obs_test
    camino_viterbi, log_prob = modelo.viterbi(obs_test)
    
    print(f"\nEstados decodificados (Viterbi):")
    print(f"  {camino_viterbi}")
    print(f"Log-probabilidad del camino óptimo: {log_prob:.4f}")
    
    # Medir exactitud comparando con estados reales
    correctos = sum(1 for real, pred in zip(estados_test, camino_viterbi) if real == pred)
    exactitud = correctos / len(estados_test)
    print(f"Exactitud de la decodificación: {exactitud:.2%} ({correctos}/{len(estados_test)})")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario definir y entrenar un HMM."""
    print("MODO INTERACTIVO: Modelos de Markov Ocultos\n")
    
    # Definir la estructura del HMM
    print("Ingrese los estados ocultos separados por espacios:")
    print("  (Ejemplo: Soleado Lluvioso Nublado)")
    estados = input("> ").strip().split()
    
    print("Ingrese las observaciones posibles separadas por espacios:")
    print("  (Ejemplo: Caminar Comprar Limpiar)")
    observaciones = input("> ").strip().split()
    
    if not estados or not observaciones:
        print("Debe definir al menos un estado y una observación.")
        return
    
    # Crear modelo con parámetros uniformes iniciales
    modelo = HMMBaumWelch(estados, observaciones)
    # Ingresar secuencias de entrenamiento (solo observaciones, sin estados)
    print("\nIngrese secuencias de observaciones para entrenamiento.")
    print("Una secuencia por línea, símbolos separados por espacios.")
    print("  (Ejemplo: Caminar Comprar Caminar Limpiar)")
    print("Escriba 'fin' cuando termine.\n")
    
    secuencias = []
    while True:
        entrada = input("> ").strip()
        if entrada.lower() == "fin":
            break
        
        secuencia = entrada.split()
        # Validar que todos los símbolos sean observaciones válidas
        if secuencia:
            secuencias.append(secuencia)
    
    if not secuencias:
        print("No se ingresaron secuencias.")
        return
    
    print(f"\n{len(secuencias)} secuencias ingresadas.")
    
    # Configurar y ejecutar entrenamiento
    max_iter = int(input("Número máximo de iteraciones (default=50): ") or "50")
    print("\nEntrenando con Baum-Welch (EM para HMMs)...\n")
    
    # El algoritmo ajustará π, T, E para maximizar P(secuencias | modelo)
    modelo.entrenar_baum_welch(secuencias, max_iter=max_iter, verbose=True)
    
    # Mostrar parámetros aprendidos
    print(f"\nParámetros aprendidos:")
    print(f"Distribución inicial π = {modelo.pi}")
    print(f"Transiciones T = {modelo.T}")
    print(f"Emisiones E = {modelo.E}")

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
