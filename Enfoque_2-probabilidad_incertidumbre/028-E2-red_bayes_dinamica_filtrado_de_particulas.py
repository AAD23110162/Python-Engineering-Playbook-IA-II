"""
028-E2-red_bayes_dinamica_filtrado_de_particulas.py
---------------------------------------------------
Este script implementa un Filtro de Partículas 1D en una Red Bayesiana Dinámica simple:
- Dinámica: x_t = x_{t-1} + u + w_t, w_t ~ N(0, Q)
- Observación: z_t = x_t + v_t, v_t ~ N(0, R)
- Propagación → ponderación → remuestreo (multinomial).

Modos de ejecución:
1. DEMO: seguimiento de un móvil 1D con N partículas y reporte de RMSE.
2. INTERACTIVO: permite cambiar N, Q, R, y pasos.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple


def gauss_pdf(x: float, mu: float, var: float) -> float:
	if var <= 0:
		return 1.0 if abs(x - mu) < 1e-12 else 0.0
	coef = 1.0 / math.sqrt(2.0 * math.pi * var)
	expo = math.exp(-0.5 * ((x - mu) ** 2) / var)
	return coef * expo


class FiltroParticulas1D:
	def __init__(self, N: int, Q: float, R: float, u: float = 1.0, x_init: float = 0.0, spread: float = 1.0):
		self.N = N
		self.Q = Q
		self.R = R
		self.u = u
		# Inicializamos partículas alrededor de x_init
		self.particulas = [random.gauss(x_init, spread) for _ in range(N)]
		self.pesos = [1.0 / N] * N

	def predecir(self):
		# Propagar con ruido de proceso
		for i in range(self.N):
			w = random.gauss(0.0, math.sqrt(self.Q))
			self.particulas[i] = self.particulas[i] + self.u + w

	def actualizar(self, z: float):
		# Ponderación por verosimilitud de la observación
		pesos = []
		for x in self.particulas:
			w = gauss_pdf(z, x, self.R)
			pesos.append(w)
		s = sum(pesos)
		if s == 0:
			# Evitar degeneración extrema
			self.pesos = [1.0 / self.N] * self.N
		else:
			self.pesos = [w / s for w in pesos]

	def remuestrear(self):
		# Remuestreo multinomial
		N = self.N
		acumulada = []
		c = 0.0
		for w in self.pesos:
			c += w
			acumulada.append(c)
		nuevas = []
		for _ in range(N):
			r = random.random()
			# encontrar primer índice con acumulada >= r
			lo, hi = 0, N - 1
			while lo < hi:
				mid = (lo + hi) // 2
				if acumulada[mid] < r:
					lo = mid + 1
				else:
					hi = mid
			nuevas.append(self.particulas[lo])
		self.particulas = nuevas
		self.pesos = [1.0 / N] * N

	def estimar(self) -> float:
		# Estimador por media ponderada
		return sum(x * w for x, w in zip(self.particulas, self.pesos))


def simular_trayectoria(n: int, x0: float, u: float, Q: float, R: float) -> Tuple[List[float], List[float]]:
	xs = [x0]
	zs = []
	x = x0
	for _ in range(n):
		w = random.gauss(0.0, math.sqrt(Q))
		x = x + u + w
		xs.append(x)
		v = random.gauss(0.0, math.sqrt(R))
		z = x + v
		zs.append(z)
	return xs[1:], zs


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Filtro de Partículas 1D")
	print("=" * 70)

	n = 100
	N = 1000
	u = 1.0
	Q = 0.5
	R = 2.0
	print(f"\nSimulando n={n}, N={N}, u={u}, Q={Q}, R={R}")

	xs, zs = simular_trayectoria(n, 0.0, u, Q, R)
	pf = FiltroParticulas1D(N=N, Q=Q, R=R, u=u, x_init=0.0, spread=2.0)

	estimaciones = []
	for z in zs:
		pf.predecir()
		pf.actualizar(z)
		pf.remuestrear()
		estimaciones.append(pf.estimar())

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
		n = int(input("Pasos n: ").strip() or "100")
		N = int(input("Número de partículas N: ").strip() or "1000")
		u = float(input("Control/deriva u: ").strip() or "1.0")
		Q = float(input("Varianza del proceso Q: ").strip() or "0.5")
		R = float(input("Varianza de observación R: ").strip() or "2.0")
	except:
		n, N, u, Q, R = 100, 1000, 1.0, 0.5, 2.0
		print("Usando valores por defecto")

	xs, zs = simular_trayectoria(n, 0.0, u, Q, R)
	pf = FiltroParticulas1D(N=N, Q=Q, R=R, u=u, x_init=0.0, spread=2.0)

	estimaciones = []
	for z in zs:
		pf.predecir()
		pf.actualizar(z)
		pf.remuestrear()
		estimaciones.append(pf.estimar())

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

"""
028-E2-red_bayes_dinamica_filtrado_de_particulas.py
--------------------------------
Este script introduce Filtrado de Partículas en Redes Bayesianas Dinámicas:
- Representa creencias con conjuntos de partículas ponderadas.
- Implementa muestreo, actualización por evidencia y remuestreo (resampling).
- Maneja modelos no lineales y no gaussianos.
- Compara con Kalman/EKF/UKF en escenarios no lineales.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo de seguimiento con filtrado de partículas.
2. INTERACTIVO: permite ajustar número de partículas y dinámica/observación.

Autor: Alejandro Aguirre Díaz
"""
