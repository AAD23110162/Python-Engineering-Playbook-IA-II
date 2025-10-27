"""
037-E2-aprendizaje_profundo.py
--------------------------------
Este script introduce conceptos de aprendizaje profundo con un MLP pequeño:
- Perceptrón multicapa (MLP) con una capa oculta.
- Entrenamiento por backpropagation y descenso de gradiente.
- Funciones de activación comunes (sigmoide, ReLU, tanh).
- Demostración con el problema XOR.

Modos de ejecución:
1. DEMO: entrena un MLP pequeño en XOR y muestra la pérdida.
2. INTERACTIVO: permite configurar tamaños y épocas.

Autor: Alejandro Aguirre Díaz
"""

import math
import random
from typing import List, Tuple, Callable


# =============================================================================
# Activaciones
# =============================================================================

def sigmoide(z: float) -> float:
	return 1.0 / (1.0 + math.exp(-z))


def d_sigmoide(a: float) -> float:
	# Derivada en términos de la salida a = sigmoide(z)
	return a * (1.0 - a)


def tanh(z: float) -> float:
	return math.tanh(z)


def d_tanh(a: float) -> float:
	return 1.0 - a * a


def relu(z: float) -> float:
	return z if z > 0 else 0.0


def d_relu(z: float) -> float:
	return 1.0 if z > 0 else 0.0


# =============================================================================
# MLP de una capa oculta (binario)
# =============================================================================

class MLP:
	"""
	MLP con una capa oculta y salida sigmoide para clasificación binaria.
	Tamaños: n_entradas -> n_oculta -> 1
	"""

	def __init__(
		self,
		n_entradas: int,
		n_oculta: int,
		activacion: str = "sigmoide",
		tasa_aprendizaje: float = 0.1,
		semilla: int = 123,
	):
		random.seed(semilla)
		self.n_entradas = n_entradas
		self.n_oculta = n_oculta
		self.tasa = tasa_aprendizaje

		# Inicialización de pesos (pequeños aleatorios)
		self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(n_entradas)] for _ in range(n_oculta)]
		self.b1 = [0.0 for _ in range(n_oculta)]
		self.W2 = [random.uniform(-0.5, 0.5) for _ in range(n_oculta)]  # salida 1
		self.b2 = 0.0

		if activacion == "tanh":
			self.f: Callable[[float], float] = tanh
			self.df = d_tanh
			self.df_input = False  # derivada recibe activación
		elif activacion == "relu":
			# Para ReLU es más conveniente pasar z a la derivada
			self.f = relu
			self.df = d_relu
			self.df_input = True
		else:
			self.f = sigmoide
			self.df = d_sigmoide
			self.df_input = False

	# ---------- Propagación hacia adelante ----------
	def _forward(self, x: List[float]) -> Tuple[List[float], List[float], float, float]:
		z1 = [sum(wj * xj for wj, xj in zip(w, x)) + b for w, b in zip(self.W1, self.b1)]
		a1 = [self.f(z) for z in z1]
		z2 = sum(w * a for w, a in zip(self.W2, a1)) + self.b2
		a2 = sigmoide(z2)  # salida binaria
		return z1, a1, z2, a2

	# ---------- Paso de entrenamiento (una muestra) ----------
	def _entrenar_muestra(self, x: List[float], y: float):
		z1, a1, z2, a2 = self._forward(x)

		# Pérdida logística (BCE): L = -[y log a2 + (1-y) log(1-a2)]
		# Gradiente salida: dL/da2 = (a2 - y)/(a2(1-a2)); con sigmoide + BCE, dL/dz2 = a2 - y
		delta2 = a2 - y  # escalar

		# Gradientes capa oculta
		if self.df_input:  # ReLU: derivada depende de z1
			dact = [self.df(z) for z in z1]
		else:  # sigmoide/tanh: derivada en función de a1
			dact = [self.df(a) for a in a1]
		delta1 = [d * self.W2[j] * delta2 for j, d in enumerate(dact)]

		# Actualización de pesos (descenso de gradiente)
		# Capa 2
		for j in range(self.n_oculta):
			self.W2[j] -= self.tasa * delta2 * a1[j]
		self.b2 -= self.tasa * delta2

		# Capa 1
		for i in range(self.n_oculta):
			for j in range(self.n_entradas):
				self.W1[i][j] -= self.tasa * delta1[i] * x[j]
			self.b1[i] -= self.tasa * delta1[i]

		# Devuelve pérdida para monitoreo
		eps = 1e-12
		loss = - (y * math.log(a2 + eps) + (1 - y) * math.log(1 - a2 + eps))
		return loss

	def ajustar(self, X: List[List[float]], y: List[int], epocas: int = 2000, barajar: bool = True) -> List[float]:
		historial: List[float] = []
		for _ in range(epocas):
			indices = list(range(len(X)))
			if barajar:
				random.shuffle(indices)
			perdida_media = 0.0
			for i in indices:
				perdida_media += self._entrenar_muestra(X[i], float(y[i]))
			perdida_media /= max(1, len(X))
			historial.append(perdida_media)
		return historial

	def predecir(self, x: List[float]) -> int:
		_, _, _, a2 = self._forward(x)
		return 1 if a2 >= 0.5 else 0

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# Datos y métricas
# =============================================================================

def datos_xor() -> Tuple[List[List[float]], List[int]]:
	X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	y = [0, 1, 1, 0]
	return X, y


def exactitud(y: List[int], yhat: List[int]) -> float:
	aciertos = sum(1 for a, b in zip(y, yhat) if a == b)
	return aciertos / len(y) if y else 0.0


# =============================================================================
# Modos
# =============================================================================

def modo_demo():
	print("MODO DEMO: MLP (XOR)\n")
	X, y = datos_xor()
	mlp = MLP(n_entradas=2, n_oculta=4, activacion="tanh", tasa_aprendizaje=0.1)
	hist = mlp.ajustar(X, y, epocas=3000)
	yhat = mlp.predecir_lote(X)
	print(f"Exactitud en XOR: {exactitud(y, yhat):.2%}")
	print(f"Pérdida final: {hist[-1]:.6f}")
	# Mostrar predicciones
	for xi, yi, pi in zip(X, y, yhat):
		print(f"Entrada {xi} -> real={yi}, pred={pi}")


def modo_interactivo():
	print("MODO INTERACTIVO: MLP\n")
	n_oculta = int(input("Tamaño de la capa oculta (default=4): ") or "4")
	activ = (input("Activación (sigmoide|tanh|relu, default=tanh): ") or "tanh").strip().lower()
	tasa = float(input("Tasa de aprendizaje (default=0.1): ") or "0.1")
	epocas = int(input("Épocas de entrenamiento (default=3000): ") or "3000")

	X, y = datos_xor()
	mlp = MLP(n_entradas=2, n_oculta=n_oculta, activacion=activ, tasa_aprendizaje=tasa)
	hist = mlp.ajustar(X, y, epocas=epocas)
	yhat = mlp.predecir_lote(X)
	print(f"\nExactitud en XOR: {exactitud(y, yhat):.2%}")
	print(f"Pérdida final: {hist[-1]:.6f}")


def main():
	print("=" * 60)
	print("037-E2: Aprendizaje profundo (MLP)")
	print("=" * 60)
	print("1. DEMO\n2. INTERACTIVO")
	op = input("Seleccione (1/2, default=1): ").strip()
	if op == "2":
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()
