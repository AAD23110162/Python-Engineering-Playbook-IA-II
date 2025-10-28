"""
041-E2-separabilidad_lineal.py
--------------------------------
Este script trata la Separabilidad Lineal en clasificación binaria:
- Comprueba separabilidad entrenando un perceptrón y verificando convergencia.
- Muestra ejemplos con y sin separabilidad (Gaussiana vs XOR).

Modos:
1. DEMO: ejemplos 2D con y sin separabilidad.
2. INTERACTIVO: introducir datos y probar separabilidad.

Autor: Alejandro Aguirre Díaz
"""

import random
from typing import List, Tuple


def agregar_bias(x: List[float]) -> List[float]:
	# Agrega una característica constante 1.0 para modelar el término de sesgo (bias)
	return x + [1.0]


def perceptron_converge(X: List[List[float]], y: List[int], tasa: float = 0.1, max_epocas: int = 1000) -> Tuple[bool, List[float]]:
	"""
	Entrena un perceptrón y devuelve (converge, w). Converge si hay una época sin errores.
	y debe estar en {-1, +1}.
	"""
	# Si no hay datos, consideramos convergencia vacía
	if not X:
		return True, []
	# Dimensión del vector de pesos con bias
	n = len(X[0]) + 1
	# Inicialización de pesos en cero (incluye bias)
	w = [0.0 for _ in range(n)]
	# Añadimos bias a todas las muestras
	Xb = [agregar_bias(x) for x in X]
	# Entrenamiento por un máximo de 'max_epocas'
	for _ in range(max_epocas):
		errores = 0
		for xi, yi in zip(Xb, y):
			# Activación lineal
			a = sum(wj * xj for wj, xj in zip(w, xi))
			# Predicción por signo
			pred = 1 if a >= 0 else -1
			# Si hay error, aplicamos la regla del perceptrón
			if pred != yi:
				for j in range(n):
					w[j] += tasa * yi * xi[j]
				errores += 1
		# Si una época termina sin errores, el algoritmo ha convergido
		if errores == 0:
			return True, w
	return False, w


def datos_separables(n: int = 80) -> Tuple[List[List[float]], List[int]]:
	X: List[List[float]] = []
	y: List[int] = []
	random.seed(2)
	# Nube positiva centrada ~ (1.5, 1.2)
	for _ in range(n // 2):
		X.append([random.gauss(1.5, 0.4), random.gauss(1.2, 0.4)])
		y.append(1)
	# Nube negativa centrada ~ (-1.5, -1.2)
	for _ in range(n // 2):
		X.append([random.gauss(-1.5, 0.4), random.gauss(-1.2, 0.4)])
		y.append(-1)
	return X, y


def datos_xor() -> Tuple[List[List[float]], List[int]]:
	# Conjunto XOR: no es separable linealmente
	X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
	y = [-1, 1, 1, -1]
	return X, y


def modo_demo():
	print("MODO DEMO: Separabilidad lineal\n")
	# Caso 1: datos separables
	Xs, ys = datos_separables()
	conv_s, _ = perceptron_converge(Xs, ys, tasa=0.1, max_epocas=200)
	print(f"Datos separables -> ¿Converge el perceptrón? {conv_s}")

	# Caso 2: datos XOR (no separables linealmente)
	Xx, yx = datos_xor()
	conv_x, _ = perceptron_converge(Xx, yx, tasa=0.1, max_epocas=500)
	print(f"Datos XOR (no separables) -> ¿Converge el perceptrón? {conv_x}")


def modo_interactivo():
	print("MODO INTERACTIVO: Prueba de separabilidad\n")
	print("1 = datos separables, 2 = datos XOR, 3 = introducir datos manualmente")
	op = input("Opción (1/2/3, default=1): ").strip() or "1"
	if op == "1":
		X, y = datos_separables()
	elif op == "2":
		X, y = datos_xor()
	else:
		# Permite ingresar puntos manualmente para probar
		print("Ingrese puntos como x1,x2,label en líneas separadas; vacío para terminar.")
		X = []; y = []
		while True:
			linea = input().strip()
			if not linea:
				break
			partes = [p for p in linea.split(",") if p.strip()]
			if len(partes) != 3:
				print("Formato inválido. Ejemplo: 0.2,1.0,1")
				continue
			X.append([float(partes[0]), float(partes[1])])
			y.append(int(partes[2]))

	# Hiperparámetros del perceptrón
	tasa = float(input("tasa aprendizaje (0.1): ") or "0.1")
	maxep = int(input("máx épocas (300): ") or "300")
	# Entrenar y verificar convergencia
	conv, w = perceptron_converge(X, y, tasa=tasa, max_epocas=maxep)
	print(f"¿Converge? {conv}")
	if w:
		print("Vector de pesos (incluye bias):", [f"{wi:.3f}" for wi in w])


def main():
	print("=" * 60)
	print("041-E2: Separabilidad lineal")
	print("=" * 60)
	print("1. DEMO\n2. INTERACTIVO")
	op = input("Seleccione (1/2, default=1): ").strip()
	if op == "2":
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()
