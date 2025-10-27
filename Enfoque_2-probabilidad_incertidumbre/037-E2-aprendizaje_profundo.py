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
	# Función logística: comprime valores reales a (0,1)
	return 1.0 / (1.0 + math.exp(-z))


def d_sigmoide(a: float) -> float:
	# Derivada en términos de la salida a = sigmoide(z)
	# d/dz sigmoide(z) = a*(1-a), útil para backprop
	return a * (1.0 - a)


def tanh(z: float) -> float:
	# Tangente hiperbólica: rango (-1,1), centrada en 0
	return math.tanh(z)


def d_tanh(a: float) -> float:
	# Derivada: 1 - tanh(z)^2, expresada como 1 - a^2
	return 1.0 - a * a


def relu(z: float) -> float:
	# Rectified Linear Unit: útil para mitigar gradientes desaparecientes
	return z if z > 0 else 0.0


def d_relu(z: float) -> float:
	# Derivada de ReLU respecto de z
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
		# Fijar semilla para reproducibilidad de inicializaciones
		random.seed(semilla)
		self.n_entradas = n_entradas
		self.n_oculta = n_oculta
		self.tasa = tasa_aprendizaje  # tasa de aprendizaje (step de gradiente)

		# Inicialización de pesos (pequeños aleatorios) y sesgos
		# W1: (n_oculta x n_entradas), b1: (n_oculta)
		self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(n_entradas)] for _ in range(n_oculta)]
		self.b1 = [0.0 for _ in range(n_oculta)]
		# W2: (n_oculta) para una salida escalar, b2: escalar
		self.W2 = [random.uniform(-0.5, 0.5) for _ in range(n_oculta)]  # salida 1
		self.b2 = 0.0

		# Selección de función de activación de la capa oculta
		if activacion == "tanh":
			self.f: Callable[[float], float] = tanh
			self.df = d_tanh
			self.df_input = False  # derivada recibe activación (a), no z
		elif activacion == "relu":
			# Para ReLU es más conveniente pasar z a la derivada (no a)
			self.f = relu
			self.df = d_relu
			self.df_input = True
		else:
			self.f = sigmoide
			self.df = d_sigmoide
			self.df_input = False

	# ---------- Propagación hacia adelante ----------
	def _forward(self, x: List[float]) -> Tuple[List[float], List[float], float, float]:
		# Capa oculta: z1 = W1·x + b1 (por neurona)
		z1 = [sum(wj * xj for wj, xj in zip(w, x)) + b for w, b in zip(self.W1, self.b1)]
		# Activación por neurona oculta
		a1 = [self.f(z) for z in z1]
		# Capa de salida: z2 = W2·a1 + b2 (escalar)
		z2 = sum(w * a for w, a in zip(self.W2, a1)) + self.b2
		# Activación de salida: sigmoide para clasificación binaria
		a2 = sigmoide(z2)  # salida binaria
		return z1, a1, z2, a2

	# ---------- Paso de entrenamiento (una muestra) ----------
	def _entrenar_muestra(self, x: List[float], y: float):
		# Propagación hacia adelante para obtener activaciones
		z1, a1, z2, a2 = self._forward(x)

		# Pérdida logística (BCE): L = -[y log a2 + (1-y) log(1-a2)]
		# Con sigmoide + BCE, el gradiente simplifica a dL/dz2 = a2 - y
		delta2 = a2 - y  # escalar

		# Gradientes en la capa oculta
		# Para ReLU usamos df(z); para sigmoide/tanh usamos df(a)
		if self.df_input:  # ReLU: derivada depende de z1
			dact = [self.df(z) for z in z1]
		else:  # sigmoide/tanh: derivada en función de a1
			dact = [self.df(a) for a in a1]
		# Regla en cadena: delta1_j = df_j * W2_j * delta2
		delta1 = [d * self.W2[j] * delta2 for j, d in enumerate(dact)]

		# Actualización de pesos (descenso de gradiente)
		# Capa de salida: W2_j <- W2_j - tasa * delta2 * a1_j
		for j in range(self.n_oculta):
			self.W2[j] -= self.tasa * delta2 * a1[j]
		# Sesgo de salida
		self.b2 -= self.tasa * delta2

		# Capa oculta: W1_ij <- W1_ij - tasa * delta1_i * x_j, b1_i similar
		for i in range(self.n_oculta):
			for j in range(self.n_entradas):
				self.W1[i][j] -= self.tasa * delta1[i] * x[j]
			self.b1[i] -= self.tasa * delta1[i]

		# Devuelve pérdida para monitoreo de entrenamiento
		eps = 1e-12
		loss = - (y * math.log(a2 + eps) + (1 - y) * math.log(1 - a2 + eps))
		return loss

	def ajustar(self, X: List[List[float]], y: List[int], epocas: int = 2000, barajar: bool = True) -> List[float]:
		# Entrena durante 'epocas' pasadas completas; devuelve historial de pérdidas medias
		historial: List[float] = []
		for _ in range(epocas):
			# Opcionalmente barajar el orden de las muestras por época
			indices = list(range(len(X)))
			if barajar:
				random.shuffle(indices)
			perdida_media = 0.0
			for i in indices:
				# Entrenamiento estocástico: una muestra a la vez
				perdida_media += self._entrenar_muestra(X[i], float(y[i]))
			perdida_media /= max(1, len(X))
			historial.append(perdida_media)
		return historial

	def predecir(self, x: List[float]) -> int:
		# Calcula la probabilidad (a2) y aplica umbral 0.5
		_, _, _, a2 = self._forward(x)
		return 1 if a2 >= 0.5 else 0

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		return [self.predecir(x) for x in X]


# =============================================================================
# Datos y métricas
# =============================================================================

def datos_xor() -> Tuple[List[List[float]], List[int]]:
	# Conjunto clásico XOR: no separable linealmente
	X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	y = [0, 1, 1, 0]
	return X, y


def exactitud(y: List[int], yhat: List[int]) -> float:
	# Proporción de aciertos entre predicciones y etiquetas reales
	aciertos = sum(1 for a, b in zip(y, yhat) if a == b)
	return aciertos / len(y) if y else 0.0


# =============================================================================
# Modos
# =============================================================================

def modo_demo():
	print("MODO DEMO: MLP (XOR)\n")
	X, y = datos_xor()
	# MLP con 2 entradas, 4 neuronas ocultas y activación tanh
	mlp = MLP(n_entradas=2, n_oculta=4, activacion="tanh", tasa_aprendizaje=0.1)
	# Entrenamiento por 3000 épocas (suficiente para resolver XOR)
	hist = mlp.ajustar(X, y, epocas=3000)
	yhat = mlp.predecir_lote(X)
	print(f"Exactitud en XOR: {exactitud(y, yhat):.2%}")
	print(f"Pérdida final: {hist[-1]:.6f}")
	# Mostrar predicciones por muestra
	for xi, yi, pi in zip(X, y, yhat):
		print(f"Entrada {xi} -> real={yi}, pred={pi}")


def modo_interactivo():
	print("MODO INTERACTIVO: MLP\n")
	# Parámetros del modelo
	n_oculta = int(input("Tamaño de la capa oculta (default=4): ") or "4")
	activ = (input("Activación (sigmoide|tanh|relu, default=tanh): ") or "tanh").strip().lower()
	tasa = float(input("Tasa de aprendizaje (default=0.1): ") or "0.1")
	epocas = int(input("Épocas de entrenamiento (default=3000): ") or "3000")

	X, y = datos_xor()
	# Crear y entrenar el MLP con parámetros elegidos
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
