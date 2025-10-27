"""
030-E2-aprendizaje_bayesiano.py
--------------------------------
Este script introduce Aprendizaje Bayesiano:
- Actualiza distribuciones sobre parámetros (a priori → a posteriori) con datos.
- Presenta conjugación (Bernoulli-Beta, Poisson-Gamma, Normal-Normal) a nivel conceptual.
- Realiza predicción posterior predictiva de nuevos datos.
- Compara MAP, media posterior y punto máximo de verosimilitud (MLE).

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplos con modelos conjugados.
2. INTERACTIVO: configuración de priors y observaciones para actualizar creencias.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple

# ============================================================================
# Distribuciones Conjugadas
# ============================================================================

class BernoulliBeta:
    """
    Modelo conjugado Bernoulli-Beta para parámetros binarios.
    Prior: Beta(alpha, beta)
    Likelihood: Bernoulli(theta)
    Posterior: Beta(alpha + éxitos, beta + fracasos)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        # Inicialización de la prior Beta con pseudo-conteos
        """
        Inicializa la distribución prior Beta.
        
        Args:
            alpha: Parámetro alpha de la distribución Beta (pseudo-éxitos)
            beta: Parámetro beta de la distribución Beta (pseudo-fracasos)
        """
        # Prior Beta(alpha, beta)
        self.alpha = alpha  # Pseudo-conteo de éxitos
        self.beta = beta    # Pseudo-conteo de fracasos
    
    def actualizar(self, datos: List[int]):
        # Actualiza la posterior sumando éxitos y fracasos a los parámetros
        """
        Actualiza la distribución posterior con nuevos datos.
        
        Args:
            datos: Lista de observaciones binarias (0 o 1)
        """
        # Contar éxitos y fracasos en los datos
        exitos = sum(datos)
        fracasos = len(datos) - exitos
        
        # Actualización conjugada: posterior = Beta(alpha + éxitos, beta + fracasos)
        self.alpha += exitos
        self.beta += fracasos
    
    def media_posterior(self) -> float:
        # Calcula la media posterior (estimación de theta)
        """Calcula la media de la distribución posterior."""
        # Media de Beta(alpha, beta) = alpha / (alpha + beta)
        return self.alpha / (self.alpha + self.beta)
    
    def moda_posterior_MAP(self) -> float:
        # Calcula la moda (MAP) de la posterior, si está definida
        """Calcula la moda (MAP) de la distribución posterior."""
        # Moda de Beta(alpha, beta) = (alpha - 1) / (alpha + beta - 2)
        # Solo definida para alpha, beta > 1
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        else:
            return self.media_posterior()  # Usar media si no está definida
    
    def varianza_posterior(self) -> float:
        # Calcula la varianza posterior (incertidumbre sobre theta)
        """Calcula la varianza de la distribución posterior."""
        # Varianza de Beta(alpha, beta) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))
    
    def predecir_nuevo(self) -> float:
        # Probabilidad predictiva para una nueva observación (Bernoulli)
        """
        Calcula la probabilidad predictiva posterior de una nueva observación positiva.
        P(x_new = 1 | datos) = E[theta | datos] = alpha / (alpha + beta)
        """
        return self.media_posterior()
    
    def __str__(self) -> str:
        """Representación en string de la distribución."""
        return f"Beta(α={self.alpha:.2f}, β={self.beta:.2f})"


class PoissonGamma:
    """
    Modelo conjugado Poisson-Gamma para tasas de eventos.
    Prior: Gamma(alpha, beta)
    Likelihood: Poisson(lambda)
    Posterior: Gamma(alpha + suma_datos, beta + n)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        # Inicialización de la prior Gamma
        """
        Inicializa la distribución prior Gamma.
        
        Args:
            alpha: Parámetro de forma (shape)
            beta: Parámetro de tasa (rate)
        """
        # Prior Gamma(alpha, beta)
        self.alpha = alpha  # Forma (shape)
        self.beta = beta    # Tasa (rate) - equivalente a 1/escala
    
    def actualizar(self, datos: List[int]):
        # Actualiza la posterior sumando conteos y número de observaciones
        """
        Actualiza la distribución posterior con nuevos datos.
        
        Args:
            datos: Lista de conteos observados
        """
        # Suma de observaciones y número de observaciones
        suma_datos = sum(datos)
        n = len(datos)
        
        # Actualización conjugada: Gamma(alpha + suma, beta + n)
        self.alpha += suma_datos
        self.beta += n
    
    def media_posterior(self) -> float:
        # Media posterior (estimación de lambda)
        """Calcula la media de la distribución posterior."""
        # Media de Gamma(alpha, beta) = alpha / beta
        return self.alpha / self.beta
    
    def moda_posterior_MAP(self) -> float:
        # Moda (MAP) de la posterior, si está definida
        """Calcula la moda (MAP) de la distribución posterior."""
        # Moda de Gamma(alpha, beta) = (alpha - 1) / beta para alpha >= 1
        if self.alpha >= 1:
            return (self.alpha - 1) / self.beta
        else:
            return 0.0
    
    def varianza_posterior(self) -> float:
        # Varianza posterior (incertidumbre sobre lambda)
        """Calcula la varianza de la distribución posterior."""
        # Varianza de Gamma(alpha, beta) = alpha / beta^2
        return self.alpha / (self.beta * self.beta)
    
    def predecir_nuevo(self) -> float:
        # Tasa esperada para una nueva observación
        """
        Calcula la tasa esperada para una nueva observación.
        E[lambda | datos] = alpha / beta
        """
        return self.media_posterior()
    
    def __str__(self) -> str:
        """Representación en string de la distribución."""
        return f"Gamma(α={self.alpha:.2f}, β={self.beta:.2f})"


class NormalNormal:
    """
    Modelo conjugado Normal-Normal para la media (con varianza conocida).
    Prior: Normal(mu_0, sigma_0^2)
    Likelihood: Normal(mu, sigma^2) con sigma^2 conocida
    Posterior: Normal(mu_n, sigma_n^2)
    """
    
    def __init__(self, mu_0: float = 0.0, sigma_0: float = 1.0, sigma_likelihood: float = 1.0):
        # Inicialización de la prior Normal para la media
        """
        Inicializa la distribución prior Normal.
        
        Args:
            mu_0: Media prior
            sigma_0: Desviación estándar prior
            sigma_likelihood: Desviación estándar de la likelihood (conocida)
        """
        # Prior Normal(mu_0, sigma_0^2)
        self.mu = mu_0             # Media posterior
        self.sigma = sigma_0       # Desviación estándar posterior
        self.sigma_lik = sigma_likelihood  # Desviación estándar de los datos (conocida)
    
    def actualizar(self, datos: List[float]):
        # Actualiza la posterior combinando la prior y los datos (conjugación)
        """
        Actualiza la distribución posterior con nuevos datos.
        
        Args:
            datos: Lista de observaciones
        """
        if not datos:
            return
        
        n = len(datos)
        media_datos = sum(datos) / n
        
        # Precisiones (inverso de varianza)
        tau_0 = 1.0 / (self.sigma ** 2)         # Precisión prior
        tau_lik = 1.0 / (self.sigma_lik ** 2)   # Precisión likelihood
        
        # Actualización conjugada de la media
        # tau_n = tau_0 + n * tau_lik
        tau_n = tau_0 + n * tau_lik
        
        # mu_n = (tau_0 * mu_0 + n * tau_lik * media_datos) / tau_n
        self.mu = (tau_0 * self.mu + n * tau_lik * media_datos) / tau_n
        
        # sigma_n^2 = 1 / tau_n
        self.sigma = math.sqrt(1.0 / tau_n)
    
    def media_posterior(self) -> float:
        # Media posterior (estimación de la media verdadera)
        """Calcula la media de la distribución posterior."""
        return self.mu
    
    def moda_posterior_MAP(self) -> float:
        # Moda (MAP) de la posterior (igual a la media en Normal)
        """Calcula la moda (MAP) de la distribución posterior."""
        # Para una distribución Normal, media = moda
        return self.mu
    
    def varianza_posterior(self) -> float:
        # Varianza posterior (incertidumbre sobre la media)
        """Calcula la varianza de la distribución posterior."""
        return self.sigma ** 2
    
    def predecir_nuevo(self) -> Tuple[float, float]:
        # Predicción para una nueva observación: media y varianza predictiva
        """
        Calcula la distribución predictiva posterior para una nueva observación.
        Returns: (media_predictiva, varianza_predictiva)
        """
        # Media predictiva = media posterior
        media_pred = self.mu
        
        # Varianza predictiva = varianza posterior + varianza likelihood
        var_pred = self.sigma ** 2 + self.sigma_lik ** 2
        
        return media_pred, var_pred
    
    def __str__(self) -> str:
        """Representación en string de la distribución."""
        return f"Normal(μ={self.mu:.2f}, σ={self.sigma:.2f})"

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra aprendizaje bayesiano con modelos conjugados."""
    print("MODO DEMO: Aprendizaje Bayesiano\n")
    
    # ========================================
    # Ejemplo 1: Bernoulli-Beta (moneda sesgada)
    # ========================================
    print("=" * 60)
    print("Ejemplo 1: Inferencia de la probabilidad de una moneda")
    print("=" * 60)
    
    # Prior uniforme: Beta(1, 1) (no informativa)
    modelo_moneda = BernoulliBeta(alpha=1.0, beta=1.0)
    print(f"Prior: {modelo_moneda}")
    print(f"  Media prior: {modelo_moneda.media_posterior():.4f}")
    print()
    
    # Simular lanzamientos de una moneda con theta = 0.7
    random.seed(42)
    theta_real = 0.7
    # Simulación de lanzamientos de moneda (theta=0.7)
    datos_moneda = [1 if random.random() < theta_real else 0 for _ in range(10)]
    
    print(f"Observaciones (10 lanzamientos): {datos_moneda}")
    print(f"  Éxitos: {sum(datos_moneda)}, Fracasos: {len(datos_moneda) - sum(datos_moneda)}")
    print()
    
    # Actualizar con datos
    # Actualización de la posterior con los datos observados
    modelo_moneda.actualizar(datos_moneda)
    print(f"Posterior: {modelo_moneda}")
    print(f"  Media posterior: {modelo_moneda.media_posterior():.4f}")
    print(f"  MAP (moda): {modelo_moneda.moda_posterior_MAP():.4f}")
    print(f"  Varianza posterior: {modelo_moneda.varianza_posterior():.4f}")
    print(f"  Predicción nueva observación: {modelo_moneda.predecir_nuevo():.4f}")
    print(f"  Theta real: {theta_real}")
    print()
    
    # Más datos
    # Más datos para refinar la estimación
    datos_moneda2 = [1 if random.random() < theta_real else 0 for _ in range(90)]
    modelo_moneda.actualizar(datos_moneda2)
    print(f"Después de 100 observaciones totales:")
    print(f"  Posterior: {modelo_moneda}")
    print(f"  Media posterior: {modelo_moneda.media_posterior():.4f}")
    print(f"  MAP: {modelo_moneda.moda_posterior_MAP():.4f}")
    print(f"  Varianza posterior: {modelo_moneda.varianza_posterior():.4f}")
    print()
    
    # ========================================
    # Ejemplo 2: Poisson-Gamma (tasa de eventos)
    # ========================================
    print("=" * 60)
    print("Ejemplo 2: Inferencia de la tasa de llegadas (Poisson)")
    print("=" * 60)
    
    # Prior débil: Gamma(1, 1)
    # Prior débil: Gamma(1, 1) (no informativa)
    modelo_poisson = PoissonGamma(alpha=1.0, beta=1.0)
    print(f"Prior: {modelo_poisson}")
    print(f"  Media prior: {modelo_poisson.media_posterior():.4f}")
    print()
    
    # Simular conteos con lambda = 5.0
    lambda_real = 5.0
    # Simulación de conteos Poisson (lambda=5.0)
    datos_poisson = [sum(1 for _ in range(100) if random.random() < lambda_real / 100) 
                     for _ in range(20)]
    
    print(f"Observaciones (20 períodos): {datos_poisson}")
    print(f"  Media observada: {sum(datos_poisson) / len(datos_poisson):.2f}")
    print()
    
    # Actualizar con datos
    # Actualización de la posterior con los datos observados
    modelo_poisson.actualizar(datos_poisson)
    print(f"Posterior: {modelo_poisson}")
    print(f"  Media posterior: {modelo_poisson.media_posterior():.4f}")
    print(f"  MAP: {modelo_poisson.moda_posterior_MAP():.4f}")
    print(f"  Varianza posterior: {modelo_poisson.varianza_posterior():.4f}")
    print(f"  Lambda real: {lambda_real}")
    print()
    
    # ========================================
    # Ejemplo 3: Normal-Normal (media)
    # ========================================
    print("=" * 60)
    print("Ejemplo 3: Inferencia de la media (Normal)")
    print("=" * 60)
    
    # Prior: Normal(0, 2) con sigma_likelihood = 1
    # Prior para la media: Normal(0, 2) y varianza conocida
    modelo_normal = NormalNormal(mu_0=0.0, sigma_0=2.0, sigma_likelihood=1.0)
    print(f"Prior: {modelo_normal}")
    print(f"  Media prior: {modelo_normal.media_posterior():.4f}")
    print()
    
    # Simular datos con mu = 3.0, sigma = 1.0
    mu_real = 3.0
    # Simulación de datos normales (mu=3.0, sigma=1.0)
    datos_normal = [random.gauss(mu_real, 1.0) for _ in range(10)]
    
    print(f"Observaciones (10 datos): {[f'{x:.2f}' for x in datos_normal]}")
    print(f"  Media observada: {sum(datos_normal) / len(datos_normal):.4f}")
    print()
    
    # Actualizar con datos
    # Actualización de la posterior con los datos observados
    modelo_normal.actualizar(datos_normal)
    print(f"Posterior: {modelo_normal}")
    print(f"  Media posterior: {modelo_normal.media_posterior():.4f}")
    print(f"  MAP: {modelo_normal.moda_posterior_MAP():.4f}")
    print(f"  Varianza posterior: {modelo_normal.varianza_posterior():.4f}")
    
    media_pred, var_pred = modelo_normal.predecir_nuevo()
    print(f"  Predicción nueva observación: μ={media_pred:.4f}, σ²={var_pred:.4f}")
    print(f"  Mu real: {mu_real}")
    print()
    
    # Más datos
    # Más datos para refinar la estimación
    datos_normal2 = [random.gauss(mu_real, 1.0) for _ in range(90)]
    modelo_normal.actualizar(datos_normal2)
    print(f"Después de 100 observaciones totales:")
    print(f"  Posterior: {modelo_normal}")
    print(f"  Media posterior: {modelo_normal.media_posterior():.4f}")
    print(f"  Varianza posterior: {modelo_normal.varianza_posterior():.4f}")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario configurar priors y actualizar con datos."""
    print("MODO INTERACTIVO: Aprendizaje Bayesiano\n")
    
    print("Seleccione el tipo de modelo:")
    print("1. Bernoulli-Beta (datos binarios)")
    print("2. Poisson-Gamma (conteos)")
    print("3. Normal-Normal (valores continuos)")
    
    tipo = input("Ingrese el número del modelo (default=1): ").strip()
    
    # Selección de modelo por el usuario
    if tipo == "2":
        # Poisson-Gamma
        print("\nModelo Poisson-Gamma")
        alpha = float(input("Ingrese alpha prior (default=1.0): ") or "1.0")
        beta = float(input("Ingrese beta prior (default=1.0): ") or "1.0")
        # Inicialización del modelo Poisson-Gamma con prior personalizada
        modelo = PoissonGamma(alpha, beta)
        print(f"\nPrior: {modelo}")
        print(f"Media prior: {modelo.media_posterior():.4f}")
        print("\nIngrese observaciones de conteos separadas por espacios:")
        entrada = input("> ").strip()
        if entrada:
            # Conversión y actualización con los datos ingresados
            datos = [int(x) for x in entrada.split()]
            print(f"Datos: {datos}")
            modelo.actualizar(datos)
            print(f"\nPosterior: {modelo}")
            print(f"Media posterior: {modelo.media_posterior():.4f}")
            print(f"MAP: {modelo.moda_posterior_MAP():.4f}")
            print(f"Varianza posterior: {modelo.varianza_posterior():.4f}")
    elif tipo == "3":
        # Normal-Normal
        print("\nModelo Normal-Normal")
        mu_0 = float(input("Ingrese media prior (default=0.0): ") or "0.0")
        sigma_0 = float(input("Ingrese desviación estándar prior (default=1.0): ") or "1.0")
        sigma_lik = float(input("Ingrese desviación estándar likelihood (default=1.0): ") or "1.0")
        # Inicialización del modelo Normal-Normal con prior personalizada
        modelo = NormalNormal(mu_0, sigma_0, sigma_lik)
        print(f"\nPrior: {modelo}")
        print(f"Media prior: {modelo.media_posterior():.4f}")
        print("\nIngrese observaciones continuas separadas por espacios:")
        entrada = input("> ").strip()
        if entrada:
            # Conversión y actualización con los datos ingresados
            datos = [float(x) for x in entrada.split()]
            print(f"Datos: {datos}")
            modelo.actualizar(datos)
            print(f"\nPosterior: {modelo}")
            print(f"Media posterior: {modelo.media_posterior():.4f}")
            print(f"Varianza posterior: {modelo.varianza_posterior():.4f}")
            media_pred, var_pred = modelo.predecir_nuevo()
            print(f"Predicción: μ={media_pred:.4f}, σ²={var_pred:.4f}")
    else:
        # Bernoulli-Beta
        print("\nModelo Bernoulli-Beta")
        alpha = float(input("Ingrese alpha prior (default=1.0): ") or "1.0")
        beta = float(input("Ingrese beta prior (default=1.0): ") or "1.0")
        # Inicialización del modelo Bernoulli-Beta con prior personalizada
        modelo = BernoulliBeta(alpha, beta)
        print(f"\nPrior: {modelo}")
        print(f"Media prior: {modelo.media_posterior():.4f}")
        print("\nIngrese observaciones binarias (0 o 1) separadas por espacios:")
        entrada = input("> ").strip()
        if entrada:
            # Conversión y actualización con los datos ingresados
            datos = [int(x) for x in entrada.split()]
            print(f"Datos: {datos}")
            modelo.actualizar(datos)
            print(f"\nPosterior: {modelo}")
            print(f"Media posterior: {modelo.media_posterior():.4f}")
            print(f"MAP: {modelo.moda_posterior_MAP():.4f}")
            print(f"Varianza posterior: {modelo.varianza_posterior():.4f}")
            print(f"Predicción nueva observación: {modelo.predecir_nuevo():.4f}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("030-E2: Aprendizaje Bayesiano")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Ejemplos con modelos conjugados")
    print("2. INTERACTIVO: Configurar priors y actualizar con datos")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
