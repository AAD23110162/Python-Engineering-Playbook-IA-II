"""
040-E2-perceptron_adaline_madaline.py
--------------------------------
Este script introduce Perceptrón, ADALINE y una versión simple de MADALINE:
- Perceptrón: regla de aprendizaje y separabilidad lineal.
- ADALINE: mínimos cuadrados y actualización por gradiente.
- MADALINE simple: dos capas de unidades lineales con activación suave.

Modos:
1. DEMO: aprendizaje en datasets separables y XOR.
2. INTERACTIVO: elección de algoritmo y parámetros.

Autor: Alejandro Aguirre Díaz
"""

import math
import random
from typing import List, Tuple


# =============================================================================
# Utilidades
# =============================================================================

def agregar_bias(x: List[float]) -> List[float]:
	return x + [1.0]


def exactitud(y: List[int], yhat: List[int]) -> float:
	aciertos = sum(1 for a, b in zip(y, yhat) if a == b)
	return aciertos / len(y) if y else 0.0


def datos_separables(n: int = 80) -> Tuple[List[List[float]], List[int]]:
	X: List[List[float]] = []
	y: List[int] = []
	random.seed(0)
	for _ in range(n // 2):
		X.append([random.gauss(1.2, 0.4), random.gauss(1.0, 0.4)])
		y.append(1)
	for _ in range(n // 2):
		X.append([random.gauss(-1.2, 0.4), random.gauss(-1.0, 0.4)])
		y.append(-1)
	return X, y


def datos_xor() -> Tuple[List[List[float]], List[int]]:
	X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	y = [-1, 1, 1, -1]
	return X, y


# =============================================================================
# Perceptrón
# =============================================================================

class Perceptron:
	def __init__(self, tasa: float = 0.1, epocas: int = 50):
		self.tasa = tasa
		self.epocas = epocas
		self.w: List[float] = []  # incluye bias al final

	def ajustar(self, X: List[List[float]], y: List[int]):
		if not X:
			return
		n = len(X[0]) + 1
		random.seed(42)
		self.w = [random.uniform(-0.5, 0.5) for _ in range(n)]
		Xb = [agregar_bias(x) for x in X]
		for _ in range(self.epocas):
			errores = 0
			for xi, yi in zip(Xb, y):
				pred = 1 if sum(wj * xj for wj, xj in zip(self.w, xi)) >= 0 else -1
				if pred != yi:
					# Regla de aprendizaje perceptrón: w <- w + tasa*(y - pred)*x
					for j in range(n):
						self.w[j] += self.tasa * yi * xi[j]
					errores += 1
			if errores == 0:
				break

	def predecir(self, x: List[float]) -> int:
		xb = agregar_bias(x)
		return 1 if sum(wj * xj for wj, xj in zip(self.w, xb)) >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# ADALINE
# =============================================================================

class Adaline:
	def __init__(self, tasa: float = 0.01, epocas: int = 100):
		self.tasa = tasa
		self.epocas = epocas
		self.w: List[float] = []

	def ajustar(self, X: List[List[float]], y: List[int]):
		if not X:
			return
		n = len(X[0]) + 1
		random.seed(123)
		self.w = [random.uniform(-0.5, 0.5) for _ in range(n)]
		Xb = [agregar_bias(x) for x in X]
		for _ in range(self.epocas):
			for xi, yi in zip(Xb, y):
				a = sum(wj * xj for wj, xj in zip(self.w, xi))  # activación lineal
				error = yi - a
				for j in range(n):
					self.w[j] += self.tasa * error * xi[j]

	def predecir(self, x: List[float]) -> int:
		xb = agregar_bias(x)
		a = sum(wj * xj for wj, xj in zip(self.w, xb))
		return 1 if a >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# MADALINE simple (dos capas con activación suave tanh)
# =============================================================================

def tanh(z: float) -> float:
	return math.tanh(z)


def d_tanh(a: float) -> float:
	return 1.0 - a * a


class MadalineSimple:
	"""
	Dos capas: entrada -> oculta (tanh) -> salida (tanh). Entrenamiento por
	gradiente (backprop) sobre MSE con objetivos en {-1, +1}. No es la regla
	histórica MADALINE II, pero sirve como aproximación educativa.
	"""

	def __init__(self, n_entradas: int, n_oculta: int = 4, tasa: float = 0.05, epocas: int = 2000):
		self.n_entradas = n_entradas
		self.n_oculta = n_oculta
		self.tasa = tasa
		self.epocas = epocas
		random.seed(99)
		# Inicializaciones pequeñas
		self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(n_entradas + 1)] for _ in range(n_oculta)]  # +bias
		self.W2 = [random.uniform(-0.5, 0.5) for _ in range(n_oculta + 1)]  # +bias

	def _forward(self, x: List[float]):
		xb = agregar_bias(x)
		z1 = [sum(wj * xj for wj, xj in zip(w, xb)) for w in self.W1]
		a1 = [tanh(z) for z in z1]
		a1b = a1 + [1.0]
		z2 = sum(w * a for w, a in zip(self.W2, a1b))
		a2 = tanh(z2)
		return xb, z1, a1, a1b, z2, a2

	def ajustar(self, X: List[List[float]], y: List[int]):
		for _ in range(self.epocas):
			idx = list(range(len(X)))
			random.shuffle(idx)
			for i in idx:
				xb, z1, a1, a1b, z2, a2 = self._forward(X[i])
				t = y[i]
				# Backprop
				delta2 = (a2 - t) * d_tanh(a2)
				delta1 = [d_tanh(a1[j]) * self.W2[j] * delta2 for j in range(self.n_oculta)]

				# Actualización
				for j in range(self.n_oculta + 1):
					self.W2[j] -= self.tasa * delta2 * a1b[j]
				for j in range(self.n_oculta):
					for k in range(self.n_entradas + 1):
						self.W1[j][k] -= self.tasa * delta1[j] * xb[k]

	def predecir(self, x: List[float]) -> int:
		_, _, _, _, _, a2 = self._forward(x)
		return 1 if a2 >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# Modos
# =============================================================================

def modo_demo():
	print("MODO DEMO: Perceptrón, Adaline y Madaline Simple\n")

	# Separables
	Xs, ys = datos_separables()
	p = Perceptron(tasa=0.1, epocas=50)
	p.ajustar(Xs, ys)
	print(f"Perceptrón (separable) acc: {exactitud(ys, p.predecir_lote(Xs)):.2%}")
	a = Adaline(tasa=0.05, epocas=100)
	a.ajustar(Xs, ys)
	print(f"Adaline    (separable) acc: {exactitud(ys, a.predecir_lote(Xs)):.2%}")

	# XOR
	Xx, yx = datos_xor()
	p2 = Perceptron(tasa=0.1, epocas=100)
	p2.ajustar(Xx, yx)
	print(f"Perceptrón (XOR) acc: {exactitud(yx, p2.predecir_lote(Xx)):.2%}")
	a2 = Adaline(tasa=0.05, epocas=200)
	a2.ajustar(Xx, yx)
	print(f"Adaline    (XOR) acc: {exactitud(yx, a2.predecir_lote(Xx)):.2%}")

	m = MadalineSimple(n_entradas=2, n_oculta=4, tasa=0.05, epocas=3000)
	m.ajustar(Xx, yx)
	print(f"MadalineSimple (XOR) acc: {exactitud(yx, m.predecir_lote(Xx)):.2%}")
	for xi, yi, pi in zip(Xx, yx, m.predecir_lote(Xx)):
		print(f"  {xi} -> real={yi}, pred={pi}")


def modo_interactivo():
	print("MODO INTERACTIVO: Elija algoritmo\n1=Perceptrón  2=Adaline  3=MadalineSimple")
	op = input("Opción (1/2/3, default=1): ").strip() or "1"
	print("Datos: 1=separables  2=XOR")
	d = input("Opción (1/2, default=1): ").strip() or "1"
	X, y = datos_separables() if d == "1" else datos_xor()

	if op == "1":
		tasa = float(input("tasa (0.1): ") or "0.1")
		ep = int(input("épocas (50): ") or "50")
		modelo = Perceptron(tasa=tasa, epocas=ep)
	elif op == "2":
		tasa = float(input("tasa (0.05): ") or "0.05")
		ep = int(input("épocas (100): ") or "100")
		modelo = Adaline(tasa=tasa, epocas=ep)
	else:
		n_h = int(input("ocultas (4): ") or "4")
		tasa = float(input("tasa (0.05): ") or "0.05")
		ep = int(input("épocas (3000): ") or "3000")
		modelo = MadalineSimple(n_entradas=len(X[0]), n_oculta=n_h, tasa=tasa, epocas=ep)

	modelo.ajustar(X, y)
	acc = exactitud(y, modelo.predecir_lote(X))
	print(f"Exactitud: {acc:.2%}")


def main():
	print("=" * 60)
	print("040-E2: Perceptrón | Adaline | MadalineSimple")
	print("=" * 60)
	print("1. DEMO\n2. INTERACTIVO")
	op = input("Seleccione (1/2, default=1): ").strip()
	if op == "2":
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()
