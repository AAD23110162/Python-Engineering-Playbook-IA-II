"""
011-E2-distribucion_probabilidad.py
--------------------------------
Este script implementa diferentes tipos de Distribuciones de Probabilidad:
- Implementa distribuciones discretas: Bernoulli, Binomial, Poisson, Geométrica
- Implementa distribuciones continuas: Uniforme, Normal (Gaussiana), Exponencial
- Calcula funciones de densidad de probabilidad (PDF) y funciones de distribución acumulativa (CDF)
- Genera muestras aleatorias de diferentes distribuciones
- Calcula estadísticas: media, varianza, desviación estándar, momentos
- Visualiza distribuciones y sus propiedades mediante gráficos de texto
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de diferentes distribuciones con parámetros predefinidos
2. INTERACTIVO: permite al usuario seleccionar distribuciones y configurar parámetros

Autor: Alejandro Aguirre Díaz
"""

import math
import random
import numpy as np

# Nota: Para mantener el proyecto autocontenible evitamos dependencias externas (scipy).
# Implementamos fórmulas básicas para PMF/PDF/CDF y muestreo con numpy/random.
# Aviso: En el encabezado se menciona la distribución Geométrica; en esta versión
# del archivo no se incluye para mantener el ejemplo compacto. Se puede añadir
# fácilmente siguiendo el mismo patrón (PMF, CDF, sample, media, varianza).

# ========== DISTRIBUCIONES DISCRETAS ==========

class Bernoulli:
	"""Distribución de Bernoulli con parámetro p = P(X=1)."""

	def __init__(self, p: float):
		# Validación básica: p debe estar en el intervalo [0,1]
		assert 0.0 <= p <= 1.0, "p debe estar en [0,1]"
		self.p = p

	def pmf(self, x: int) -> float:
		# PMF: P(X=1)=p, P(X=0)=1-p. Para otros valores, probabilidad 0.
		if x == 1:
			return self.p
		if x == 0:
			return 1.0 - self.p
		return 0.0

	def cdf(self, x: int) -> float:
		# CDF en Bernoulli es escalón: 0 si x<0, 1-p si 0<=x<1, 1 si x>=1
		if x < 0:
			return 0.0
		if x < 1:
			return 1.0 - self.p
		return 1.0

	def sample(self, n: int = 1) -> np.ndarray:
		# Muestreo comparando uniformes U(0,1) con p: True->1, False->0
		return (np.random.rand(n) < self.p).astype(int)

	def media(self) -> float:
		# E[X]=p
		return self.p

	def varianza(self) -> float:
		# Var(X)=p(1-p)
		return self.p * (1 - self.p)


class Binomial:
	"""Distribución Binomial con parámetros n (ensayos) y p (éxito)."""

	def __init__(self, n: int, p: float):
		# n: número de ensayos, p: prob. de éxito por ensayo
		assert n >= 0 and 0.0 <= p <= 1.0
		self.n = n
		self.p = p

	def pmf(self, k: int) -> float:
		# PMF: C(n,k) p^k (1-p)^(n-k)
		if k < 0 or k > self.n:
			return 0.0
		return math.comb(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

	def cdf(self, k: int) -> float:
		# CDF(k) = sum_{i=0..k} PMF(i)
		k = int(math.floor(k))
		return sum(self.pmf(i) for i in range(0, k + 1))

	def sample(self, n: int = 1) -> np.ndarray:
		# Uso del generador binomial de NumPy para n muestras
		return np.random.binomial(self.n, self.p, size=n)

	def media(self) -> float:
		# E[X]=np
		return self.n * self.p

	def varianza(self) -> float:
		# Var(X)=np(1-p)
		return self.n * self.p * (1 - self.p)


class Poisson:
	"""Distribución de Poisson con parámetro lambda_ (tasa)."""

	def __init__(self, lambda_: float):
		# lambda_ > 0 representa la tasa media de ocurrencias
		assert lambda_ > 0
		self.lambda_ = lambda_

	def pmf(self, k: int) -> float:
		# PMF: e^{-λ} λ^k / k! para k>=0
		if k < 0:
			return 0.0
		return math.exp(-self.lambda_) * (self.lambda_ ** k) / math.factorial(k)

	def cdf(self, k: int) -> float:
		# CDF(k) = sum_{i=0..k} PMF(i)
		k = int(math.floor(k))
		if k < 0:
			return 0.0
		return sum(self.pmf(i) for i in range(0, k + 1))

	def sample(self, n: int = 1) -> np.ndarray:
		# Genera n muestras Poisson con tasa λ
		return np.random.poisson(self.lambda_, size=n)

	def media(self) -> float:
		# E[X]=λ
		return self.lambda_

	def varianza(self) -> float:
		# Var(X)=λ
		return self.lambda_


# ========== DISTRIBUCIONES CONTINUAS ==========

class Uniforme:
	"""Distribución Uniforme continua en [a, b]."""

	def __init__(self, a: float, b: float):
		# Se exige a < b para un intervalo válido
		assert b > a, "Se requiere a < b"
		self.a = a
		self.b = b

	def pdf(self, x: float) -> float:
		# Densidad constante 1/(b-a) dentro del intervalo; 0 fuera
		return 1.0 / (self.b - self.a) if self.a <= x <= self.b else 0.0

	def cdf(self, x: float) -> float:
		# CDF lineal por tramos según la posición de x
		if x <= self.a:
			return 0.0
		if x >= self.b:
			return 1.0
		return (x - self.a) / (self.b - self.a)

	def sample(self, n: int = 1) -> np.ndarray:
		# Muestras uniformes continuas en [a,b]
		return np.random.uniform(self.a, self.b, size=n)

	def media(self) -> float:
		# E[X]=(a+b)/2
		return 0.5 * (self.a + self.b)

	def varianza(self) -> float:
		# Var(X)=(b-a)^2/12
		return ((self.b - self.a) ** 2) / 12.0


class Normal:
	"""Distribución Normal (Gaussiana) N(mu, sigma^2)."""

	def __init__(self, mu: float, sigma: float):
		# sigma es la desviación estándar (>0)
		assert sigma > 0
		self.mu = mu
		self.sigma = sigma

	def pdf(self, x: float) -> float:
		# Fórmula clásica usando la puntuación z
		z = (x - self.mu) / self.sigma
		return (1.0 / (self.sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)

	def cdf(self, x: float) -> float:
		# Usando la función error: CDF = 0.5 * [1 + erf( (x-mu)/(sigma*sqrt(2)) )]
		z = (x - self.mu) / (self.sigma * math.sqrt(2.0))
		return 0.5 * (1.0 + math.erf(z))

	def sample(self, n: int = 1) -> np.ndarray:
		# Generador normal de NumPy para n muestras
		return np.random.normal(self.mu, self.sigma, size=n)

	def media(self) -> float:
		# E[X]=mu
		return self.mu

	def varianza(self) -> float:
		# Var(X)=sigma^2
		return self.sigma ** 2


class Exponencial:
	"""Distribución Exponencial con tasa lambda_ (>0)."""

	def __init__(self, lambda_: float):
		# lambda_ es la tasa; el tiempo medio es 1/lambda_
		assert lambda_ > 0
		self.lambda_ = lambda_

	def pdf(self, x: float) -> float:
		# Densidad: λ e^{-λx} para x>=0 (0 para x<0)
		return self.lambda_ * math.exp(-self.lambda_ * x) if x >= 0 else 0.0

	def cdf(self, x: float) -> float:
		# CDF: 1 - e^{-λx} para x>=0
		return 1.0 - math.exp(-self.lambda_ * x) if x >= 0 else 0.0

	def sample(self, n: int = 1) -> np.ndarray:
		# NumPy usa 'scale' = 1/λ para la exponencial
		return np.random.exponential(1.0 / self.lambda_, size=n)

	def media(self) -> float:
		# E[X]=1/λ
		return 1.0 / self.lambda_

	def varianza(self) -> float:
		# Var(X)=1/λ^2
		return 1.0 / (self.lambda_ ** 2)


# ========== UTILIDADES ========== 

def histograma_texto(muestras: np.ndarray, bins: int = 10, ancho: int = 40):
	"""Dibuja un histograma simple en texto para visualizar la distribución de muestras."""
	if len(muestras) == 0:
		print("(sin datos)")
		return
	# Calculamos el rango de datos y construimos el histograma con NumPy
	min_v = float(np.min(muestras))
	max_v = float(np.max(muestras))
	if max_v == min_v:
		max_v = min_v + 1e-9
	hist, edges = np.histogram(muestras, bins=bins, range=(min_v, max_v))
	# Escalamos la longitud de barras al ancho deseado
	max_count = np.max(hist)
	for i in range(len(hist)):
		bar_len = int(ancho * (hist[i] / max_count)) if max_count > 0 else 0
		print(f"[{edges[i]:.2f}, {edges[i+1]:.2f}) " + "#" * bar_len)


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Distribuciones de Probabilidad")
	print("="*70)

	# Bernoulli
	print("\n--- Bernoulli p=0.3 ---")
	bern = Bernoulli(0.3)
	print(f"PMF(0)={bern.pmf(0):.2f}, PMF(1)={bern.pmf(1):.2f}")
	print(f"Media={bern.media():.2f}, Varianza={bern.varianza():.2f}")
	m = bern.sample(1000)
	# Un histograma de 2 bins refleja las frecuencias de 0s y 1s
	print("Muestras (10):", m[:10])
	histograma_texto(m, bins=2)

	# Binomial
	print("\n--- Binomial n=10, p=0.5 ---")
	bino = Binomial(10, 0.5)
	print(f"PMF(k=0..3)={[round(bino.pmf(k),3) for k in range(4)]}")
	print(f"CDF(5)={bino.cdf(5):.3f}")
	print(f"Media={bino.media():.2f}, Varianza={bino.varianza():.2f}")

	# Poisson
	print("\n--- Poisson lambda=3 ---")
	pois = Poisson(3.0)
	print(f"PMF(k=0..4)={[round(pois.pmf(k),3) for k in range(5)]}")
	print(f"CDF(4)={pois.cdf(4):.3f}")
	m = pois.sample(1000)
	# La forma del histograma se concentra alrededor de λ
	histograma_texto(m, bins=8)

	# Uniforme
	print("\n--- Uniforme [0,1] ---")
	uni = Uniforme(0.0, 1.0)
	print(f"PDF(0.2)={uni.pdf(0.2):.2f}, CDF(0.2)={uni.cdf(0.2):.2f}")
	print(f"Media={uni.media():.2f}, Varianza={uni.varianza():.3f}")
	m = uni.sample(500)
	# Debería verse relativamente plano
	histograma_texto(m, bins=10)

	# Normal
	print("\n--- Normal mu=0, sigma=1 ---")
	norm = Normal(0.0, 1.0)
	print(f"PDF(0)={norm.pdf(0.0):.3f}, CDF(0)={norm.cdf(0.0):.3f}")
	print(f"Media={norm.media():.2f}, Varianza={norm.varianza():.2f}")
	m = norm.sample(800)
	# El histograma debe aproximar una campana centrada en 0
	histograma_texto(m, bins=14)

	# Exponencial
	print("\n--- Exponencial lambda=2 ---")
	exp = Exponencial(2.0)
	print(f"PDF(0.5)={exp.pdf(0.5):.3f}, CDF(0.5)={exp.cdf(0.5):.3f}")
	print(f"Media={exp.media():.2f}, Varianza={exp.varianza():.2f}")
	m = exp.sample(800)
	# El histograma muestra decaimiento rápido a la derecha
	histograma_texto(m, bins=12)


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Distribuciones de Probabilidad")
	print("="*70)
	print("Selecciona una distribución:")
	print("1. Bernoulli")
	print("2. Binomial")
	print("3. Poisson")
	print("4. Uniforme (continua)")
	print("5. Normal (Gaussiana)")
	print("6. Exponencial")

	opcion = input("Opción (1-6): ").strip() or "1"

	try:
		if opcion == '1':
			p = float(input("p (0-1): ").strip() or "0.5")
			d = Bernoulli(p)
			n = int(input("Número de muestras: ").strip() or "1000")
			m = d.sample(n)
			# Comparamos estadísticos teóricos vs muestrales
			print(f"Media teórica: {d.media():.3f} | Var teórica: {d.varianza():.3f}")
			print(f"Media muestral: {np.mean(m):.3f} | Var muestral: {np.var(m):.3f}")
			histograma_texto(m, bins=2)

		elif opcion == '2':
			n_ = int(input("n (ensayos): ").strip() or "10")
			p = float(input("p (éxito): ").strip() or "0.5")
			d = Binomial(n_, p)
			k = int(input("k para CDF(k): ").strip() or "5")
			print(f"CDF({k}) = {d.cdf(k):.4f}")
			n = int(input("Número de muestras: ").strip() or "1000")
			m = d.sample(n)
			# Limitar bins a n+1 evita huecos en valores imposibles
			histograma_texto(m, bins=min(20, n_+1))

		elif opcion == '3':
			lamb = float(input("lambda (>0): ").strip() or "3.0")
			d = Poisson(lamb)
			n = int(input("Número de muestras: ").strip() or "1000")
			m = d.sample(n)
			histograma_texto(m, bins=10)

		elif opcion == '4':
			a = float(input("a: ").strip() or "0.0")
			b = float(input("b (>a): ").strip() or "1.0")
			d = Uniforme(a, b)
			n = int(input("Número de muestras: ").strip() or "1000")
			m = d.sample(n)
			print(f"Media teórica: {d.media():.3f}, Var teórica: {d.varianza():.3f}")
			histograma_texto(m, bins=12)

		elif opcion == '5':
			mu = float(input("mu: ").strip() or "0.0")
			sigma = float(input("sigma (>0): ").strip() or "1.0")
			d = Normal(mu, sigma)
			n = int(input("Número de muestras: ").strip() or "1000")
			m = d.sample(n)
			print(f"Media teórica: {d.media():.3f}, Var teórica: {d.varianza():.3f}")
			histograma_texto(m, bins=14)

		elif opcion == '6':
			lamb = float(input("lambda (>0): ").strip() or "1.0")
			d = Exponencial(lamb)
			n = int(input("Número de muestras: ").strip() or "1000")
			m = d.sample(n)
			print(f"Media teórica: {d.media():.3f}, Var teórica: {d.varianza():.3f}")
			histograma_texto(m, bins=12)
		else:
			print("Opción no válida.")
	except Exception as e:
		print("Error en la configuración:", e)


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("DISTRIBUCIONES DE PROBABILIDAD")
	print("="*70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		modo_demo()
	print("\n" + "="*70)
	print("FIN DEL PROGRAMA")
	print("="*70 + "\n")


if __name__ == "__main__":
	main()
