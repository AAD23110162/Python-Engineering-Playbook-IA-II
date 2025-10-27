"""
028-E2-red_bayes_dinamica_filtrado_de_particulas.py
----------------------------------------------------
Este script introduce Filtrado de Partículas en Redes Bayesianas Dinámicas:
- Representa creencias con conjuntos de partículas ponderadas.
- Implementa muestreo, actualización por evidencia y remuestreo (resampling).
- Maneja modelos no lineales y no gaussianos.
- Compara con Kalman/EKF/UKF en escenarios no lineales.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo de seguimiento con filtrado de partículas.
2. INTERACTIVO: permite ajustar número de partículas y dinámica/observación.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple


def gauss_pdf(x: float, mu: float, var: float) -> float:
	"""Calcula la densidad gaussiana evaluada en x con media mu y varianza var."""
	# Caso degenerado: varianza nula o negativa
	if var <= 0:
		return 1.0 if abs(x - mu) < 1e-12 else 0.0
	# Fórmula gaussiana: (1/√(2πσ²)) * exp(-0.5 * ((x-μ)²/σ²))
	coef = 1.0 / math.sqrt(2.0 * math.pi * var)
	expo = math.exp(-0.5 * ((x - mu) ** 2) / var)
	return coef * expo


class FiltroParticulas1D:
	def __init__(self, N: int, Q: float, R: float, u: float = 1.0, x_init: float = 0.0, spread: float = 1.0):
		# N: número de partículas (representa la distribución del estado)
		# Q: varianza del ruido de proceso (incertidumbre en la dinámica)
		# R: varianza del ruido de observación (incertidumbre en las mediciones)
		# u: control/deriva constante aplicada en cada paso
		# x_init: estimación inicial del estado
		# spread: dispersión inicial de las partículas alrededor de x_init
		self.N = N
		self.Q = Q
		self.R = R
		self.u = u
		# Inicializamos partículas alrededor de x_init con distribución gaussiana
		self.particulas = [random.gauss(x_init, spread) for _ in range(N)]
		# Pesos uniformes al inicio (todas las partículas igualmente probables)
		self.pesos = [1.0 / N] * N

	def predecir(self):
		"""Paso de PREDICCIÓN: propaga cada partícula según el modelo dinámico."""
		# Propagar con ruido de proceso: x_t = x_{t-1} + u + w_t, w_t ~ N(0, Q)
		for i in range(self.N):
			w = random.gauss(0.0, math.sqrt(self.Q))
			self.particulas[i] = self.particulas[i] + self.u + w

	def actualizar(self, z: float):
		"""Paso de ACTUALIZACIÓN: pondera cada partícula según su verosimilitud con z."""
		# Ponderación por verosimilitud de la observación: w_i ∝ P(z|x_i)
		pesos = []
		for x in self.particulas:
			# Calcula P(z|x) usando modelo de observación z = x + v, v ~ N(0, R)
			w = gauss_pdf(z, x, self.R)
			pesos.append(w)
		s = sum(pesos)
		if s == 0:
			# Evitar degeneración extrema: todas las partículas tienen peso cero
			# (esto ocurre si z es muy improbable para todas las partículas)
			self.pesos = [1.0 / self.N] * self.N
		else:
			# Normalizar pesos para que sumen 1
			self.pesos = [w / s for w in pesos]

	def remuestrear(self):
		"""Paso de REMUESTREO: reemplaza partículas según sus pesos (multinomial)."""
		# Remuestreo multinomial: duplica partículas con alto peso, descarta las de bajo peso
		N = self.N
		# Construye CDF acumulada para muestreo inverso
		acumulada = []
		c = 0.0
		for w in self.pesos:
			c += w
			acumulada.append(c)
		nuevas = []
		for _ in range(N):
			r = random.random()
			# Búsqueda binaria: encontrar primer índice con acumulada >= r
			lo, hi = 0, N - 1
			while lo < hi:
				mid = (lo + hi) // 2
				if acumulada[mid] < r:
					lo = mid + 1
				else:
					hi = mid
			nuevas.append(self.particulas[lo])
		# Reemplaza el conjunto de partículas y resetea pesos a uniformes
		self.particulas = nuevas
		self.pesos = [1.0 / N] * N

	def estimar(self) -> float:
		"""Calcula el estimado del estado como la media ponderada de las partículas."""
		# Estimador por media ponderada: E[x] = Σ w_i * x_i
		return sum(x * w for x, w in zip(self.particulas, self.pesos))


def simular_trayectoria(n: int, x0: float, u: float, Q: float, R: float) -> Tuple[List[float], List[float]]:
	"""Genera trayectoria real y observaciones ruidosas para el modelo 1D.
	Modelo verdadero:
	- x_t = x_{t-1} + u + w_t,   w_t ~ N(0, Q)  [dinámica con ruido de proceso]
	- z_t = x_t + v_t,           v_t ~ N(0, R)  [observación con ruido de medición]
	Devuelve:
	- xs: lista de estados reales x_t (t=1..n)
	- zs: lista de observaciones z_t (t=1..n)
	"""
	xs = [x0]
	zs = []
	x = x0
	for _ in range(n):
		# Evolucionar estado con ruido de proceso
		w = random.gauss(0.0, math.sqrt(Q))
		x = x + u + w
		xs.append(x)
		# Generar observación con ruido de medición
		v = random.gauss(0.0, math.sqrt(R))
		z = x + v
		zs.append(z)
	return xs[1:], zs


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Filtro de Partículas 1D")
	print("=" * 70)

	# Parámetros de la simulación
	n = 100      # número de pasos temporales
	N = 1000     # número de partículas (más partículas = mejor aproximación)
	u = 1.0      # deriva/control constante
	Q = 0.5      # varianza del ruido de proceso
	R = 2.0      # varianza del ruido de observación
	print(f"\nSimulando n={n}, N={N}, u={u}, Q={Q}, R={R}")

	# Genera trayectoria verdadera y observaciones ruidosas
	xs, zs = simular_trayectoria(n, 0.0, u, Q, R)
	
	# Inicializa filtro de partículas con spread inicial de 2.0
	pf = FiltroParticulas1D(N=N, Q=Q, R=R, u=u, x_init=0.0, spread=2.0)

	estimaciones = []
	for z in zs:
		# Ciclo SIR: Sample (predecir), Importance (actualizar), Resample (remuestrear)
		pf.predecir()      # propaga partículas
		pf.actualizar(z)   # pondera por verosimilitud
		pf.remuestrear()   # duplica partículas con alto peso
		estimaciones.append(pf.estimar())

	# RMSE: error cuadrático medio entre estimado y estado real
	rmse = math.sqrt(sum((e - x) ** 2 for e, x in zip(estimaciones, xs)) / len(xs))
	print(f"\nRMSE (filtro de partículas): {rmse:.4f}")
	print("Primeros 10 pasos (real, estimado, observación):")
	for i in range(min(10, n)):
		print(f"  t={i+1}: real={xs[i]:6.3f}, est={estimaciones[i]:6.3f}, obs={zs[i]:6.3f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: Filtro de Partículas 1D")
	print("=" * 70)
	try:
		# Permite ajustar parámetros del filtro y la simulación
		n = int(input("Pasos n: ").strip() or "100")
		N = int(input("Número de partículas N: ").strip() or "1000")
		u = float(input("Control/deriva u: ").strip() or "1.0")
		Q = float(input("Varianza del proceso Q: ").strip() or "0.5")
		R = float(input("Varianza de observación R: ").strip() or "2.0")
	except:
		n, N, u, Q, R = 100, 1000, 1.0, 0.5, 2.0
		print("Usando valores por defecto")

	xs, zs = simular_trayectoria(n, 0.0, u, Q, R)
	# Inicializa filtro con spread inicial para diversidad de partículas
	pf = FiltroParticulas1D(N=N, Q=Q, R=R, u=u, x_init=0.0, spread=2.0)

	estimaciones = []
	for z in zs:
		# Algoritmo SIR (Sequential Importance Resampling)
		pf.predecir()
		pf.actualizar(z)
		pf.remuestrear()
		estimaciones.append(pf.estimar())

	# Métrica de desempeño: RMSE del estimador por partículas
	rmse = math.sqrt(sum((e - x) ** 2 for e, x in zip(estimaciones, xs)) / len(xs))
	print(f"\nRMSE: {rmse:.4f}")
	for i in range(min(10, n)):
		print(f"  t={i+1}: real={xs[i]:6.3f}, est={estimaciones[i]:6.3f}, obs={zs[i]:6.3f}")


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("RED BAYESIANA DINÁMICA: FILTRADO DE PARTÍCULAS (1D)")
	print("=" * 70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		modo_demo()
	print("\n" + "=" * 70)
	print("FIN DEL PROGRAMA")
	print("=" * 70 + "\n")


if __name__ == "__main__":
	main()

