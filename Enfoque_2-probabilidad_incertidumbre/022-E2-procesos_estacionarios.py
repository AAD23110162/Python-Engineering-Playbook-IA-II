"""
022-E2-procesos_estacionarios.py
--------------------------------
Este script introduce procesos estocásticos estacionarios con un ejemplo AR(1):
- Simula un proceso AR(1): X_t = mu + phi*(X_{t-1}-mu) + w_t, w_t ~ N(0, sigma^2)
- Estima media, varianza y función de autocorrelación (ACF) empíricas.
- Ilustra condiciones de estacionariedad (|phi| < 1) y efecto del burn-in.

Modos de ejecución:
1. DEMO: corre simulaciones con parámetros predefinidos y reporta estadísticas.
2. INTERACTIVO: permite elegir phi, sigma, mu y tamaño de muestra.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List


class ProcesoAR1:
	"""Simulador y analizador sencillo de un proceso AR(1) estacionario."""

	def __init__(self, phi: float, sigma: float = 1.0, mu: float = 0.0):
		self.phi = phi
		self.sigma = sigma
		self.mu = mu

	def simular(self, n: int, burn_in: int = 100) -> List[float]:
		"""
		Genera una trayectoria del AR(1).
		- Si |phi| < 1, el proceso es estacionario; usar burn-in para aproximar el régimen estacionario.
		- Si |phi| ≥ 1, la varianza crece o diverge (no estacionario o límite).
		"""
		x = self.mu
		# Fase de burn-in para acercar al régimen estacionario
		for _ in range(max(0, burn_in)):
			w = random.gauss(0.0, self.sigma)
			x = self.mu + self.phi * (x - self.mu) + w

		# Generación de muestras a registrar
		serie = []
		for _ in range(n):
			w = random.gauss(0.0, self.sigma)
			x = self.mu + self.phi * (x - self.mu) + w
			serie.append(x)
		return serie


def media(serie: List[float]) -> float:
	return sum(serie) / len(serie) if serie else 0.0


def varianza(serie: List[float]) -> float:
	if not serie:
		return 0.0
	m = media(serie)
	return sum((x - m) ** 2 for x in serie) / len(serie)


def acf(serie: List[float], max_lag: int = 10) -> List[float]:
	"""
	Autocorrelación empírica hasta retardo max_lag (incluye lag 0 = 1.0).
	"""
	n = len(serie)
	if n == 0:
		return []
	m = media(serie)
	var = sum((x - m) ** 2 for x in serie)
	if var == 0:
		return [1.0] + [0.0] * max_lag
	res = [1.0]
	for k in range(1, max_lag + 1):
		num = 0.0
		for t in range(k, n):
			num += (serie[t] - m) * (serie[t - k] - m)
		res.append(num / var)
	return res


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Procesos Estacionarios (AR(1))")
	print("=" * 70)

	# Parámetros estacionarios típicos
	phi = 0.7
	sigma = 1.0
	mu = 0.0
	n = 2000
	burn_in = 200

	print(f"\nSimulando AR(1) con phi={phi}, sigma={sigma}, mu={mu}, n={n} (burn-in={burn_in})")
	ar1 = ProcesoAR1(phi=phi, sigma=sigma, mu=mu)
	serie = ar1.simular(n=n, burn_in=burn_in)

	m = media(serie)
	v = varianza(serie)
	r = acf(serie, max_lag=10)

	# Teórico para AR(1) estacionario: Var(X) = sigma^2 / (1 - phi^2), ACF(k) = phi^k
	var_teo = (sigma ** 2) / (1 - phi ** 2) if abs(phi) < 1 else float('inf')

	print("\nEstadísticas empíricas:")
	print(f"  Media ≈ {m:.3f}")
	print(f"  Varianza ≈ {v:.3f} (teórica: {var_teo:.3f})")
	print("  ACF (k=0..10):")
	print("   ", ", ".join(f"{val:.3f}" for val in r))

	print("\nNotas:")
	print("- Si |phi| < 1, la ACF decae aproximadamente como phi^k.")
	print("- Con burn-in suficiente, la media se acerca a mu y la varianza a la teórica.")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: AR(1) Estacionario")
	print("=" * 70)

	try:
		phi = float(input("phi (recomendado |phi|<1): ").strip() or "0.7")
		sigma = float(input("sigma (ruido): ").strip() or "1.0")
		mu = float(input("mu (media): ").strip() or "0.0")
		n = int(input("tamaño de muestra n: ").strip() or "2000")
		burn_in = int(input("burn-in: ").strip() or "200")
	except:
		phi, sigma, mu, n, burn_in = 0.7, 1.0, 0.0, 2000, 200
		print("Usando valores por defecto")

	ar1 = ProcesoAR1(phi=phi, sigma=sigma, mu=mu)
	serie = ar1.simular(n=n, burn_in=burn_in)

	m = media(serie)
	v = varianza(serie)
	r = acf(serie, max_lag=10)

	print(f"\nMedia ≈ {m:.4f}")
	print(f"Varianza ≈ {v:.4f}")
	print("ACF (k=0..10):")
	print(", ".join(f"{val:.3f}" for val in r))


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("PROCESOS ESTACIONARIOS")
	print("=" * 70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		# Por defecto, ejecutar la DEMO
		modo_demo()
	print("\n" + "=" * 70)
	print("FIN DEL PROGRAMA")
	print("=" * 70 + "\n")


if __name__ == "__main__":
	main()
