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
	# Añade término de sesgo (bias) como una característica constante 1.0
	return x + [1.0]


def exactitud(y: List[int], yhat: List[int]) -> float:
	# Calcula la proporción de aciertos entre predicciones y etiquetas reales
	aciertos = sum(1 for a, b in zip(y, yhat) if a == b)
	return aciertos / len(y) if y else 0.0


def datos_separables(n: int = 80) -> Tuple[List[List[float]], List[int]]:
	X: List[List[float]] = []
	y: List[int] = []
	random.seed(0)
	# Generar nube positiva alrededor de (1.2, 1.0)
	for _ in range(n // 2):
		X.append([random.gauss(1.2, 0.4), random.gauss(1.0, 0.4)])
		y.append(1)
	# Generar nube negativa alrededor de (-1.2, -1.0)
	for _ in range(n // 2):
		X.append([random.gauss(-1.2, 0.4), random.gauss(-1.0, 0.4)])
		y.append(-1)
	return X, y


def datos_xor() -> Tuple[List[List[float]], List[int]]:
	# Conjunto XOR clásico en {-1, +1} para clases
	X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	y = [-1, 1, 1, -1]
	return X, y


# =============================================================================
# Perceptrón
# =============================================================================

class Perceptron:
	def __init__(self, tasa: float = 0.1, epocas: int = 50):
		# Hiperparámetros: tasa de aprendizaje y épocas de entrenamiento
		self.tasa = tasa
		self.epocas = epocas
		# Vector de pesos incluyendo el término de bias al final
		self.w: List[float] = []  # incluye bias al final

	def ajustar(self, X: List[List[float]], y: List[int]):
		if not X:
			return
		# Dimensión de pesos con bias: entradas + 1
		n = len(X[0]) + 1
		random.seed(42)
		# Inicialización aleatoria pequeña de pesos
		self.w = [random.uniform(-0.5, 0.5) for _ in range(n)]
		Xb = [agregar_bias(x) for x in X]
		for _ in range(self.epocas):
			errores = 0
			for xi, yi in zip(Xb, y):
				# Predicción por el signo de la activación lineal (umbral en 0)
				pred = 1 if sum(wj * xj for wj, xj in zip(self.w, xi)) >= 0 else -1
				if pred != yi:
					# Regla de aprendizaje perceptrón: w <- w + tasa * y * x
					for j in range(n):
						self.w[j] += self.tasa * yi * xi[j]
					errores += 1
			if errores == 0:
				break

	def predecir(self, x: List[float]) -> int:
		# Clasifica aplicando el umbral a la combinación lineal w·x + b
		xb = agregar_bias(x)
		return 1 if sum(wj * xj for wj, xj in zip(self.w, xb)) >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# ADALINE
# =============================================================================

class Adaline:
	def __init__(self, tasa: float = 0.01, epocas: int = 100):
		# Hiperparámetros y vector de pesos para ADALINE
		self.tasa = tasa
		self.epocas = epocas
		self.w: List[float] = []

	def ajustar(self, X: List[List[float]], y: List[int]):
		if not X:
			return
		# Dimensión con bias
		n = len(X[0]) + 1
		random.seed(123)
		# Inicialización de pesos pequeños
		self.w = [random.uniform(-0.5, 0.5) for _ in range(n)]
		Xb = [agregar_bias(x) for x in X]
		for _ in range(self.epocas):
			for xi, yi in zip(Xb, y):
				# Activación lineal (regresión) y error
				a = sum(wj * xj for wj, xj in zip(self.w, xi))  # activación lineal
				error = yi - a
				# Actualización por descenso de gradiente del MSE: w <- w + tasa * error * x
				for j in range(n):
					self.w[j] += self.tasa * error * xi[j]

	def predecir(self, x: List[float]) -> int:
		# Clasificación por el signo de la salida lineal aprendida
		xb = agregar_bias(x)
		a = sum(wj * xj for wj, xj in zip(self.w, xb))
		return 1 if a >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# MADALINE simple (dos capas con activación suave tanh)
# =============================================================================

def tanh(z: float) -> float:
	# Activación suave para MADALINE: salida continua en (-1,1)
	return math.tanh(z)


def d_tanh(a: float) -> float:
	# Derivada de tanh en términos de su salida a
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
		# W1: (n_oculta x (n_entradas+1)) incluyendo bias, W2: (n_oculta+1) incluyendo bias
		self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(n_entradas + 1)] for _ in range(n_oculta)]  # +bias
		self.W2 = [random.uniform(-0.5, 0.5) for _ in range(n_oculta + 1)]  # +bias

	def _forward(self, x: List[float]):
		# Paso hacia adelante: calcular activaciones de la capa oculta y salida
		xb = agregar_bias(x)
		# Capa oculta: z1 = W1·xb, a1 = tanh(z1)
		z1 = [sum(wj * xj for wj, xj in zip(w, xb)) for w in self.W1]
		a1 = [tanh(z) for z in z1]
		# Añadir bias a la activación de la oculta para la salida
		a1b = a1 + [1.0]
		# Capa salida: z2 = W2·a1b, a2 = tanh(z2)
		z2 = sum(w * a for w, a in zip(self.W2, a1b))
		a2 = tanh(z2)
		return xb, z1, a1, a1b, z2, a2

	def ajustar(self, X: List[List[float]], y: List[int]):
		for _ in range(self.epocas):
			idx = list(range(len(X)))
			random.shuffle(idx)
			for i in idx:
				# Forward para la muestra i
				xb, z1, a1, a1b, z2, a2 = self._forward(X[i])
				t = y[i]
				# Backprop: con MSE y tanh, dL/dz2 = (a2 - t) * d_tanh(a2)
				delta2 = (a2 - t) * d_tanh(a2)
				# Para la oculta: delta1_j = d_tanh(a1_j) * W2_j * delta2
				delta1 = [d_tanh(a1[j]) * self.W2[j] * delta2 for j in range(self.n_oculta)]

				# Actualización de pesos por descenso de gradiente
				# Salida
				for j in range(self.n_oculta + 1):
					self.W2[j] -= self.tasa * delta2 * a1b[j]
				# Oculta
				for j in range(self.n_oculta):
					for k in range(self.n_entradas + 1):
						self.W1[j][k] -= self.tasa * delta1[j] * xb[k]

	def predecir(self, x: List[float]) -> int:
		# Predicción por el signo de la salida tanh
		_, _, _, _, _, a2 = self._forward(x)
		return 1 if a2 >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# Modos
# =============================================================================

def modo_demo():
	print("MODO DEMO: Perceptrón, Adaline y Madaline Simple\n")

	# Caso 1: datos linealmente separables
	Xs, ys = datos_separables()
	p = Perceptron(tasa=0.1, epocas=50)
	p.ajustar(Xs, ys)
	print(f"Perceptrón (separable) acc: {exactitud(ys, p.predecir_lote(Xs)):.2%}")
	a = Adaline(tasa=0.05, epocas=100)
	a.ajustar(Xs, ys)
	print(f"Adaline    (separable) acc: {exactitud(ys, a.predecir_lote(Xs)):.2%}")

	# Caso 2: XOR (no separable linealmente)
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
		# Parámetros del Perceptrón
		tasa = float(input("tasa (0.1): ") or "0.1")
		ep = int(input("épocas (50): ") or "50")
		modelo = Perceptron(tasa=tasa, epocas=ep)
	elif op == "2":
		# Parámetros de Adaline
		tasa = float(input("tasa (0.05): ") or "0.05")
		ep = int(input("épocas (100): ") or "100")
		modelo = Adaline(tasa=tasa, epocas=ep)
	else:
		# Parámetros de MadalineSimple
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
