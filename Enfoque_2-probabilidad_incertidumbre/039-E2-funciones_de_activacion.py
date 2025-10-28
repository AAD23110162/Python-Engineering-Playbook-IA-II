"""
039-E2-funciones_de_activacion.py
--------------------------------
Este script presenta funciones de activación comunes (sigmoide, tanh, ReLU,
Leaky ReLU, ELU y Softmax) y sus derivadas numéricas básicas.

Modos:
1. DEMO: imprime valores y derivadas en puntos representativos.
2. INTERACTIVO: calcula activaciones para entradas proporcionadas por el usuario.

Autor: Alejandro Aguirre Díaz
"""

import math
from typing import List


def sigmoide(z: float) -> float:
	# Función logística: salida en (0,1), útil para clasificación binaria
	return 1.0 / (1.0 + math.exp(-z))


def d_sigmoide(a: float) -> float:
	# Derivada de la sigmoide respecto de z, usando la salida a
	return a * (1.0 - a)


def tanh(z: float) -> float:
	# Tangente hiperbólica: salida en (-1,1), centrada en cero
	return math.tanh(z)


def d_tanh(a: float) -> float:
	# Derivada de tanh respecto de z, usando la salida a
	return 1.0 - a * a


def relu(z: float) -> float:
	# Rectified Linear Unit: salida 0 si z<0, z si z>=0
	return z if z > 0 else 0.0


def d_relu(z: float) -> float:
	# Derivada de ReLU: 1 si z>0, 0 si z<=0
	return 1.0 if z > 0 else 0.0


def leaky_relu(z: float, alpha: float = 0.01) -> float:
	# Leaky ReLU: igual a ReLU pero con pendiente pequeña para z<0
	return z if z > 0 else alpha * z


def d_leaky_relu(z: float, alpha: float = 0.01) -> float:
	# Derivada de Leaky ReLU: 1 si z>0, alpha si z<=0
	return 1.0 if z > 0 else alpha


def elu(z: float, alpha: float = 1.0) -> float:
	# Exponential Linear Unit: suave para z<0, lineal para z>=0
	return z if z >= 0 else alpha * (math.exp(z) - 1)


def d_elu(z: float, alpha: float = 1.0) -> float:
	# Derivada de ELU: 1 si z>=0, alpha*exp(z) si z<0
	return 1.0 if z >= 0 else alpha * math.exp(z)


def softmax(z: List[float]) -> List[float]:
	# Softmax: convierte un vector en probabilidades (suma 1)
	# Se resta el máximo para evitar overflow numérico
	m = max(z)
	exps = [math.exp(v - m) for v in z]
	s = sum(exps)
	return [e / s for e in exps]


def modo_demo():
	# Imprime valores y derivadas de activaciones en puntos representativos
	print("MODO DEMO: Funciones de activación\n")
	puntos = [-3.0, -1.0, 0.0, 1.0, 3.0]
	print("z\tsigmoide(a)\tdsigmoide\ttanh(a)\tdtanh\tReLU\tdReLU\tLeaky\tdLeaky\tELU\tdELU")
	for z in puntos:
		# Calcular activaciones y derivadas para cada punto
		a_sig = sigmoide(z)
		a_tanh = tanh(z)
		a_relu = relu(z)
		a_lrelu = leaky_relu(z)
		a_elu = elu(z)
		print(
			f"{z:+.1f}\t{a_sig:.4f}\t\t{d_sigmoide(a_sig):.4f}\t\t{a_tanh:.4f}\t{d_tanh(a_tanh):.4f}\t{a_relu:.2f}\t{d_relu(z):.1f}\t{a_lrelu:.2f}\t{d_leaky_relu(z):.2f}\t{a_elu:.2f}\t{d_elu(z):.2f}"
		)

	# Ejemplo de softmax sobre un vector de 3 valores
	print("\nSoftmax ejemplo: z = [2.0, 1.0, 0.1]")
	z = [2.0, 1.0, 0.1]
	p = softmax(z)
	print("Probabilidades:", [f"{pi:.4f}" for pi in p])
	print("Suma:", sum(p))


def modo_interactivo():
	# Permite al usuario ingresar valores y ver activaciones/derivadas
	print("MODO INTERACTIVO: Activaciones\n")
	print("Ingrese una lista de valores z separados por coma, ej: -2,-1,0,1,2")
	z_str = input("z = ").strip()
	if not z_str:
		print("Entrada vacía, saliendo.")
		return
	# Parsear lista de valores z
	z_vals = [float(v) for v in z_str.split(",") if v.strip()]
	print("\nResultados:")
	for z in z_vals:
		# Calcular activaciones y derivadas para cada valor ingresado
		a_sig = sigmoide(z)
		a_t = tanh(z)
		print(
			f"z={z:+.3f} -> sigmoide={a_sig:.4f}, dsig={d_sigmoide(a_sig):.4f}, tanh={a_t:.4f}, dtanh={d_tanh(a_t):.4f}, ReLU={relu(z):.3f}, dReLU={d_relu(z):.1f}, Leaky={leaky_relu(z):.3f}, ELU={elu(z):.3f}"
		)

	# Calcular softmax sobre todos los valores ingresados
	print("\nSoftmax de la lista completa:")
	p = softmax(z_vals)
	print("p =", [f"{pi:.4f}" for pi in p], "| suma=", f"{sum(p):.4f}")


def main():
	print("=" * 60)
	print("039-E2: Funciones de activación")
	print("=" * 60)
	print("1. DEMO\n2. INTERACTIVO")
	op = input("Seleccione (1/2, default=1): ").strip()
	if op == "2":
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()
