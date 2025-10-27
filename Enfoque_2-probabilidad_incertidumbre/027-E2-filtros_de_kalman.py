"""
027-E2-filtros_de_kalman.py
--------------------------------
Este script presenta Filtros de Kalman para modelos lineales-gaussianos:
- Define dinámica lineal y observaciones con ruido gaussiano.
- Implementa el ciclo de predicción-actualización de Kalman a nivel conceptual.
- Discute variantes: Kalman Extendido (EKF) y Unscented (UKF).
- Muestra ejemplos con seguimiento de posición/velocidad en 1D.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: trayectoria sintética con ruido y estimación de estado.
2. INTERACTIVO: permite ajustar matrices A, H, Q, R y estados iniciales.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple


class Kalman1D:
	"""Filtro de Kalman escalar para x_t con control constante u."""

	def __init__(self, Q: float, R: float, x0: float = 0.0, P0: float = 1.0, u: float = 0.0):
		# Q: varianza del ruido de proceso (incertidumbre en la dinámica)
		# R: varianza del ruido de medición (incertidumbre en la observación)
		# x0: estimación inicial del estado
		# P0: varianza inicial de la estimación (incertidumbre inicial)
		# u: control/deriva constante aplicada en cada paso (modelo x_t = x_{t-1} + u + w_t)
		self.Q = Q  # varianza del proceso
		self.R = R  # varianza de la observación
		self.x = x0  # estimado actual
		self.P = P0  # incertidumbre del estimado
		self.u = u  # control/deriva

	def predecir(self):
		# Paso de PREDICCIÓN (a priori): propaga estado e incertidumbre
		# Modelo 1D con A=1, B=1 y H=1: x_pred = x + u ; P_pred = P + Q
		self.x = self.x + self.u
		self.P = self.P + self.Q

	def actualizar(self, z: float):
		# Paso de ACTUALIZACIÓN (a posteriori): combina predicción con medición z
		# Ganancia de Kalman (peso óptimo para la medición frente a la predicción)
		# K = P_pred/(P_pred+R)
		K = self.P / (self.P + self.R)
		# Corrige el estado con la innovación (z - x_pred)
		# x = x_pred + K*(z - x_pred)
		self.x = self.x + K * (z - self.x)
		# Reduce la incertidumbre tras incorporar z: P = (1-K) P_pred
		self.P = (1 - K) * self.P


def simular_trayectoria(n: int, x0: float, u: float, Q: float, R: float) -> Tuple[List[float], List[float]]:
	"""Genera estados verdaderos y observaciones ruidosas para el modelo 1D.
	Modelo verdadero:
	- x_t = x_{t-1} + u + w_t,   w_t ~ N(0, Q)
	- z_t = x_t + v_t,           v_t ~ N(0, R)
	Devuelve:
	- xs: lista de estados reales x_t (t=1..n)
	- zs: lista de observaciones z_t (t=1..n)
	"""
	xs = [x0]
	zs = []
	x = x0
	for _ in range(n):
		# Evolucionar estado con ruido de proceso w
		w = random.gauss(0.0, math.sqrt(Q))
		x = x + u + w
		xs.append(x)
		# Generar observación con ruido de medición v
		v = random.gauss(0.0, math.sqrt(R))
		z = x + v
		zs.append(z)
	return xs[1:], zs


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Filtro de Kalman 1D")
	print("=" * 70)

	# Parámetros de la simulación (n pasos, deriva u y ruidos Q/R)
	n = 100
	x0 = 0.0
	u = 1.0
	Q = 0.1
	R = 1.0

	print(f"\nSimulando trayectoria: n={n}, u={u}, Q={Q}, R={R}")
	xs, zs = simular_trayectoria(n, x0, u, Q, R)

	# Inicializa el filtro con incertidumbre inicial P0 y misma deriva u
	kf = Kalman1D(Q=Q, R=R, x0=x0, P0=1.0, u=u)
	estimaciones = []
	for z in zs:
		# Ciclo predictivo: predice y luego actualiza con la medición z
		kf.predecir()
		kf.actualizar(z)
		estimaciones.append(kf.x)

	# RMSE: error cuadrático medio entre estimado y estado real
	rmse = math.sqrt(sum((e - x) ** 2 for e, x in zip(estimaciones, xs)) / len(xs))
	print(f"\nRMSE estimador vs. estado real: {rmse:.4f}")
	print("Primeras 10 parejas (real, estimado, observación):")
	for i in range(min(10, n)):
		print(f"  t={i+1}: real={xs[i]:6.3f}, est={estimaciones[i]:6.3f}, obs={zs[i]:6.3f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: Filtro de Kalman 1D")
	print("=" * 70)
	try:
		n = int(input("Pasos n: ").strip() or "100")
		u = float(input("Control/deriva u: ").strip() or "1.0")
		Q = float(input("Varianza del proceso Q: ").strip() or "0.1")
		R = float(input("Varianza de observación R: ").strip() or "1.0")
	except:
		n, u, Q, R = 100, 1.0, 0.1, 1.0
		print("Usando valores por defecto")

	xs, zs = simular_trayectoria(n, 0.0, u, Q, R)
	# Inicializa el filtro y ejecuta el ciclo predictivo por n pasos
	kf = Kalman1D(Q=Q, R=R, x0=0.0, P0=1.0, u=u)
	estimaciones = []
	for z in zs:
		kf.predecir()
		kf.actualizar(z)
		estimaciones.append(kf.x)

	# Métrica de desempeño: RMSE del estimador
	rmse = math.sqrt(sum((e - x) ** 2 for e, x in zip(estimaciones, xs)) / len(xs))
	print(f"\nRMSE: {rmse:.4f}")
	print("Primeros 10 pasos:")
	for i in range(min(10, n)):
		print(f"  t={i+1}: real={xs[i]:6.3f}, est={estimaciones[i]:6.3f}, obs={zs[i]:6.3f}")


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("FILTROS DE KALMAN (1D)")
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

