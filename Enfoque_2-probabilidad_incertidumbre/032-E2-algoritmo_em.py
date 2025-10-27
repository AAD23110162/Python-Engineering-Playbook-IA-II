"""
032-E2-algoritmo_em.py
--------------------------------
Este script introduce el Algoritmo EM (Expectation-Maximization):
- Resuelve estimación de parámetros con variables latentes.
- Alterna pasos E (esperanza) y M (maximización) hasta convergencia.
- Aplica a mezclas gaussianas y HMM (Baum-Welch) a nivel conceptual.
- Discute criterios de parada y sensibilidad a inicialización.

El programa puede ejecutarse en dos modos:
1. DEMO: ajuste de una mezcla de gaussianas sintética.
2. INTERACTIVO: permite configurar número de componentes y criterios de convergencia.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple

# ============================================================================
# Mezcla de Gaussianas con EM
# ============================================================================

class MezclaGaussianas:
    """
    Modelo de Mezcla de Gaussianas (GMM) estimado con el algoritmo EM.
    """
    
    def __init__(self, k: int, dimension: int = 1):
        """
        Inicializa el modelo de mezcla.
        
        Args:
            k: Número de componentes (gaussianas) en la mezcla
            dimension: Dimensión de los datos (default=1)
        """
        # Configuración básica del modelo
        self.k = k  # Número de componentes
        self.d = dimension  # Dimensión de los datos
        
        # Parámetros de la mezcla
        self.pesos = [1.0 / k] * k  # π_j: peso de cada componente (suman 1)
        self.medias = []            # μ_j: media de cada componente
        self.varianzas = []         # σ²_j: varianza de cada componente
        
        # Log-likelihood para monitorear convergencia
        self.log_likelihood_historia = []
    
    def inicializar_aleatorio(self, datos: List[float]):
        """
        Inicializa los parámetros de forma aleatoria a partir de los datos.
        
        Args:
            datos: Datos observados
        """
        # Seleccionar medias aleatorias de los datos (semillas iniciales)
        self.medias = random.sample(datos, self.k)
        
        # Varianzas iniciales: varianza global de los datos
        media_global = sum(datos) / len(datos)
        varianza_global = sum((x - media_global) ** 2 for x in datos) / len(datos)
        self.varianzas = [varianza_global] * self.k
        
        # Pesos uniformes (cada componente igualmente probable al inicio)
        self.pesos = [1.0 / self.k] * self.k
    
    def _pdf_gaussiana(self, x: float, media: float, varianza: float) -> float:
        """Calcula la densidad de probabilidad gaussiana."""
        if varianza <= 0:
            varianza = 1e-6
        
        coef = 1.0 / math.sqrt(2 * math.pi * varianza)
        exp_term = -0.5 * ((x - media) ** 2) / varianza
        return coef * math.exp(exp_term)
    
    def _paso_E(self, datos: List[float]) -> List[List[float]]:
        """
        Paso E (Expectation): calcula responsabilidades.
        
        Args:
            datos: Datos observados
            
        Returns:
            responsabilidades[i][j] = P(componente j | dato i)
        """
        n = len(datos)
        responsabilidades = []
        
        # Para cada punto de datos, calcular responsabilidades
        for x in datos:
            # Calcular P(x | componente j) * P(componente j) para cada j
            probs = []
            for j in range(self.k):
                # Likelihood: P(x | μ_j, σ²_j)
                likelihood = self._pdf_gaussiana(x, self.medias[j], self.varianzas[j])
                # Multiplicar por prior: π_j
                prob = self.pesos[j] * likelihood
                probs.append(prob)
            
            # Normalizar para obtener responsabilidades (posterior)
            # γ_ij = P(componente j | dato i) vía regla de Bayes
            suma = sum(probs)
            if suma > 0:
                responsabilidades.append([p / suma for p in probs])
            else:
                # Evitar división por cero
                responsabilidades.append([1.0 / self.k] * self.k)
        
        return responsabilidades
    
    def _paso_M(self, datos: List[float], responsabilidades: List[List[float]]):
        """
        Paso M (Maximization): actualiza parámetros.
        
        Args:
            datos: Datos observados
            responsabilidades: Responsabilidades calculadas en paso E
        """
        n = len(datos)
        
        # Actualizar parámetros para cada componente
        for j in range(self.k):
            # Suma de responsabilidades para el componente j (masa total asignada)
            N_j = sum(resp[j] for resp in responsabilidades)
            
            if N_j < 1e-6:
                # Evitar división por cero si un componente no tiene masa
                continue
            
            # Actualizar peso: π_j = N_j / n (proporción de datos asignados)
            self.pesos[j] = N_j / n
            
            # Actualizar media: μ_j = Σ(γ_ij * x_i) / N_j (media ponderada)
            media_nueva = sum(resp[j] * x for resp, x in zip(responsabilidades, datos)) / N_j
            self.medias[j] = media_nueva
            
            # Actualizar varianza: σ²_j = Σ(γ_ij * (x_i - μ_j)²) / N_j (varianza ponderada)
            varianza_nueva = sum(resp[j] * (x - media_nueva) ** 2 
                                for resp, x in zip(responsabilidades, datos)) / N_j
            self.varianzas[j] = max(varianza_nueva, 1e-6)  # Evitar varianza cero
    
    def _calcular_log_likelihood(self, datos: List[float]) -> float:
        """
        Calcula el log-likelihood de los datos dado el modelo.
        
        Args:
            datos: Datos observados
            
        Returns:
            Log-likelihood total
        """
        log_likelihood = 0.0
        
        # Calcular P(x) para cada dato y sumar log(P(x))
        for x in datos:
            # P(x) = Σ_j π_j * P(x | μ_j, σ²_j) (mezcla de gaussianas)
            prob_x = 0.0
            for j in range(self.k):
                likelihood = self._pdf_gaussiana(x, self.medias[j], self.varianzas[j])
                prob_x += self.pesos[j] * likelihood
            
            # Sumar log(P(x)) para obtener log-verosimilitud
            log_likelihood += math.log(prob_x + 1e-10)
        
        return log_likelihood
    
    def ajustar(self, datos: List[float], max_iter: int = 100, 
                tolerancia: float = 1e-4, verbose: bool = True) -> int:
        """
        Ajusta el modelo a los datos usando el algoritmo EM.
        
        Args:
            datos: Datos observados
            max_iter: Número máximo de iteraciones
            tolerancia: Criterio de convergencia (cambio en log-likelihood)
            verbose: Si imprimir progreso
            
        Returns:
            Número de iteraciones realizadas
        """
        # Inicializar parámetros aleatoriamente desde los datos
        self.inicializar_aleatorio(datos)
        
        # Log-likelihood inicial (para monitorear convergencia)
        log_lik_anterior = self._calcular_log_likelihood(datos)
        self.log_likelihood_historia = [log_lik_anterior]
        
        if verbose:
            print(f"Iteración 0: Log-likelihood = {log_lik_anterior:.4f}")
        
        # Iterar EM hasta convergencia o máximo de iteraciones
        for iteracion in range(1, max_iter + 1):
            # Paso E: calcular responsabilidades (esperanza de variables latentes)
            responsabilidades = self._paso_E(datos)
            
            # Paso M: actualizar parámetros (maximizar log-likelihood esperada)
            self._paso_M(datos, responsabilidades)
            
            # Calcular nuevo log-likelihood
            log_lik_actual = self._calcular_log_likelihood(datos)
            self.log_likelihood_historia.append(log_lik_actual)
            
            if verbose and iteracion % 10 == 0:
                print(f"Iteración {iteracion}: Log-likelihood = {log_lik_actual:.4f}")
            
            # Verificar convergencia (cambio en log-likelihood menor que tolerancia)
            mejora = log_lik_actual - log_lik_anterior
            if abs(mejora) < tolerancia:
                if verbose:
                    print(f"Convergencia alcanzada en iteración {iteracion}")
                    print(f"Log-likelihood final = {log_lik_actual:.4f}")
                return iteracion
            
            log_lik_anterior = log_lik_actual
        
        if verbose:
            print(f"Máximo de iteraciones alcanzado ({max_iter})")
            print(f"Log-likelihood final = {log_lik_anterior:.4f}")
        
        return max_iter
    
    def predecir(self, x: float) -> Tuple[int, List[float]]:
        """
        Predice el componente más probable para un nuevo dato.
        
        Args:
            x: Dato a clasificar
            
        Returns:
            (componente_predicho, probabilidades)
        """
        probs = []
        # Calcular probabilidad posterior de cada componente dado x
        for j in range(self.k):
            likelihood = self._pdf_gaussiana(x, self.medias[j], self.varianzas[j])
            prob = self.pesos[j] * likelihood
            probs.append(prob)
        
        # Normalizar probabilidades
        suma = sum(probs)
        if suma > 0:
            probs = [p / suma for p in probs]
        else:
            probs = [1.0 / self.k] * self.k
        
        # Componente con mayor probabilidad (clasificación hard)
        componente = max(range(self.k), key=lambda j: probs[j])
        
        return componente, probs

# ============================================================================
# Funciones de Utilidad
# ============================================================================

def generar_mezcla_gaussianas(k: int, n_por_componente: int, 
                               params: List[Tuple[float, float]]) -> List[float]:
    """
    Genera datos sintéticos de una mezcla de gaussianas.
    
    Args:
        k: Número de componentes
        n_por_componente: Número de puntos por componente
        params: Lista de (media, desviación_estándar) para cada componente
        
    Returns:
        Lista de datos generados
    """
    datos = []
    # Generar datos de cada componente gaussiano
    for media, desv_est in params:
        for _ in range(n_por_componente):
            x = random.gauss(media, desv_est)
            datos.append(x)
    
    # Mezclar aleatoriamente (para simular mezcla real)
    random.shuffle(datos)
    return datos

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra el algoritmo EM con mezcla de gaussianas."""
    print("MODO DEMO: Algoritmo EM para Mezcla de Gaussianas\n")
    
    # Configurar semilla para reproducibilidad
    random.seed(42)
    
    # ========================================
    # Generar datos sintéticos
    # ========================================
    print("=" * 60)
    print("Generando datos sintéticos")
    print("=" * 60)
    
    # Parámetros reales de la mezcla (ground truth)
    k_real = 3
    params_reales = [
        (0.0, 0.5),   # Componente 1: μ=0, σ=0.5
        (5.0, 1.0),   # Componente 2: μ=5, σ=1.0
        (10.0, 0.8)   # Componente 3: μ=10, σ=0.8
    ]
    
    n_por_componente = 50
    # Generar datos sintéticos desde la mezcla real
    datos = generar_mezcla_gaussianas(k_real, n_por_componente, params_reales)
    
    print(f"Generados {len(datos)} puntos de una mezcla de {k_real} gaussianas")
    print("Parámetros reales:")
    for i, (media, desv_est) in enumerate(params_reales, 1):
        print(f"  Componente {i}: μ={media:.1f}, σ={desv_est:.1f}")
    print()
    
    # ========================================
    # Ajustar modelo con EM
    # ========================================
    print("=" * 60)
    print("Ajustando modelo con EM")
    print("=" * 60)
    
    # Crear modelo y ajustar con el algoritmo EM
    modelo = MezclaGaussianas(k=k_real)
    iteraciones = modelo.ajustar(datos, max_iter=50, tolerancia=1e-4, verbose=True)
    print()
    
    # ========================================
    # Mostrar parámetros estimados
    # ========================================
    print("=" * 60)
    print("Parámetros estimados")
    print("=" * 60)
    
    for j in range(modelo.k):
        print(f"Componente {j+1}:")
        print(f"  Peso: π = {modelo.pesos[j]:.4f}")
        print(f"  Media: μ = {modelo.medias[j]:.4f}")
        print(f"  Varianza: σ² = {modelo.varianzas[j]:.4f}")
        print(f"  Desviación estándar: σ = {math.sqrt(modelo.varianzas[j]):.4f}")
        print()
    
    # ========================================
    # Clasificar algunos puntos
    # ========================================
    print("=" * 60)
    print("Clasificación de puntos de prueba")
    print("=" * 60)
    
    # Puntos de prueba en distintas regiones
    puntos_prueba = [0.0, 2.5, 5.0, 7.5, 10.0]
    
    # Clasificar cada punto al componente más probable
    for x in puntos_prueba:
        componente, probs = modelo.predecir(x)
        print(f"Punto x={x:.1f}:")
        print(f"  Componente asignado: {componente + 1}")
        print(f"  Probabilidades: {', '.join(f'P(C{i+1})={p:.4f}' for i, p in enumerate(probs))}")
        print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario configurar y ejecutar EM."""
    print("MODO INTERACTIVO: Algoritmo EM\n")
    
    # Configurar número de componentes
    k = int(input("Ingrese el número de componentes (default=3): ") or "3")
    
    # Generar o ingresar datos
    print("\n¿Desea generar datos sintéticos? (s/n, default=s):")
    opcion = input("> ").strip().lower()
    
    if opcion == "n":
        # Ingresar datos manualmente
        print("\nIngrese los datos separados por espacios:")
        entrada = input("> ").strip()
        datos = [float(x) for x in entrada.split()]
    else:
        # Generar datos sintéticos
        n_total = int(input("\nIngrese el número total de puntos (default=150): ") or "150")
        n_por_comp = n_total // k
        
        print(f"\nGenerando {n_total} puntos con {k} componentes...")
        
        # Generar parámetros aleatorios para cada componente
        random.seed()
        params = []
        for i in range(k):
            media = random.uniform(-5, 15)
            desv_est = random.uniform(0.5, 2.0)
            params.append((media, desv_est))
            print(f"  Componente {i+1}: μ={media:.2f}, σ={desv_est:.2f}")
        
        # Generar mezcla sintética
        datos = generar_mezcla_gaussianas(k, n_por_comp, params)
    
    print(f"\nTotal de datos: {len(datos)}")
    print(f"Rango: [{min(datos):.2f}, {max(datos):.2f}]")
    print()
    
    # Configurar parámetros de EM
    max_iter = int(input("Ingrese el número máximo de iteraciones (default=100): ") or "100")
    tolerancia = float(input("Ingrese la tolerancia de convergencia (default=0.0001): ") or "0.0001")
    
    print("\nEjecutando EM...\n")
    
    # Ajustar modelo con el algoritmo EM
    modelo = MezclaGaussianas(k=k)
    iteraciones = modelo.ajustar(datos, max_iter=max_iter, 
                                 tolerancia=tolerancia, verbose=True)
    
    # Mostrar resultados finales
    print(f"\nParámetros estimados:")
    for j in range(modelo.k):
        print(f"Componente {j+1}:")
        print(f"  π={modelo.pesos[j]:.4f}, μ={modelo.medias[j]:.4f}, σ={math.sqrt(modelo.varianzas[j]):.4f}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("032-E2: Algoritmo EM (Expectation-Maximization)")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Ajuste de mezcla de gaussianas")
    print("2. INTERACTIVO: Configurar y ejecutar EM")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
