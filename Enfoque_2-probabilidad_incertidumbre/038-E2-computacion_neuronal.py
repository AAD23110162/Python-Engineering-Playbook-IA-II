"""
038-E2-computacion_neuronal.py
--------------------------------
Este script introduce la Computación Neuronal mediante neuronas umbral y
combinaciones que implementan funciones lógicas simples y XOR.

Modos de ejecución:
1. DEMO: tabla de verdad de neuronas (AND/OR/NOT/NAND) y red XOR.
2. INTERACTIVO: prueba una neurona con pesos y umbral personalizados.

Autor: Alejandro Aguirre Díaz
"""

from typing import List, Tuple


def escalon(z: float) -> int:
	"""Función de activación escalón (Heaviside)."""
	# Devuelve 1 si el potencial supera el umbral (>= 0), en caso contrario 0
	return 1 if z >= 0 else 0


class NeuronaUmbral:
	"""
	Neurona de umbral: salida = 1 si w·x - t >= 0, en otro caso 0.
	w: pesos, t: umbral.
	"""

	def __init__(self, pesos: List[float], umbral: float):
		# Guardamos los pesos de las conexiones de entrada
		self.pesos = pesos
		# t (umbral) desplaza la frontera de decisión
		self.umbral = umbral

	def activar(self, x: List[float]) -> int:
		# Potencial de activación: z = w·x - t
		z = sum(w * xi for w, xi in zip(self.pesos, x)) - self.umbral
		# Aplicamos función escalón para obtener salida binaria {0,1}
		return escalon(z)


# -----------------------------------------------------------------------------
# Neuronas lógicas básicas con parámetros fijos conocidos
# -----------------------------------------------------------------------------

def neurona_AND() -> NeuronaUmbral:
	# w=[1,1], umbral=1.5 -> solo 1 si ambas entradas son 1
	# Interpretación: requiere que la suma de entradas active por encima del umbral
	return NeuronaUmbral([1.0, 1.0], umbral=1.5)


def neurona_OR() -> NeuronaUmbral:
	# w=[1,1], umbral=0.5 -> 1 si alguna entrada es 1
	# Con umbral bajo, basta con que una sola entrada sea 1 para activar
	return NeuronaUmbral([1.0, 1.0], umbral=0.5)


def neurona_NOT() -> NeuronaUmbral:
	# NOT(x) con una sola entrada: salida 1 cuando x=0
	# w=[-1], umbral=-0.5 -> z = -x + 0.5 >= 0 cuando x=0
	# Equivale a invertir la señal de entrada con un sesgo
	return NeuronaUmbral([-1.0], umbral=-0.5)


def neurona_NAND() -> NeuronaUmbral:
	# NAND es negación de AND: w=[-1,-1], umbral=-1.5
	# Es universal: permite construir cualquier compuerta lógica
	return NeuronaUmbral([-1.0, -1.0], umbral=-1.5)


# -----------------------------------------------------------------------------
# Red XOR con dos capas de neuronas umbral
# -----------------------------------------------------------------------------

class RedXOR:
	"""
	Implementa XOR con 2 neuronas ocultas y 1 neurona de salida:
	  h1 = NAND(x1, x2)
	  h2 = OR(x1, x2)
	  y  = AND(h1, h2)
	"""

	def __init__(self):
		# Capa oculta: dos neuronas lógicas (NAND y OR)
		self.h1 = neurona_NAND()
		self.h2 = neurona_OR()
		# Capa de salida: combina ambas con AND
		self.out = neurona_AND()

	def predecir(self, x: List[int]) -> int:
		# Activaciones ocultas intermedias
		h1 = self.h1.activar(x)
		h2 = self.h2.activar(x)
		# Salida final combinando h1 y h2
		return self.out.activar([h1, h2])


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def tabla_verdad_doble(f, nombre: str):
	# Imprime la tabla de verdad para funciones binarias con entradas {0,1}
	print(f"Tabla de verdad: {nombre}")
	for x1 in (0, 1):
		for x2 in (0, 1):
			# Evaluar la función sobre cada combinación de entradas
			y = f([x1, x2])
			print(f"  {nombre}({x1}, {x2}) -> {y}")
	print()


def modo_demo():
	print("MODO DEMO: Neuronas umbral y XOR\n")
	# Crear neuronas lógicas básicas preconfiguradas
	and_n = neurona_AND()
	or_n = neurona_OR()
	not_n = neurona_NOT()
	nand_n = neurona_NAND()

	# Mostrar tablas de verdad para compuertas clásicas
	tabla_verdad_doble(lambda x: and_n.activar(x), "AND")
	tabla_verdad_doble(lambda x: or_n.activar(x), "OR")
	# NOT es unario
	print("Tabla de verdad: NOT")
	for x in (0, 1):
		print(f"  NOT({x}) -> {not_n.activar([x])}")
	print()
	tabla_verdad_doble(lambda x: nand_n.activar(x), "NAND")

	# Red XOR armada con NAND, OR y AND
	red = RedXOR()
	tabla_verdad_doble(lambda x: red.predecir(x), "XOR")


def modo_interactivo():
	print("MODO INTERACTIVO: Neurona umbral personalizada\n")
	print("Ingrese pesos separados por coma, ejemplo: 1, -0.5, 2")
	w_str = input("Pesos: ").strip()
	# Parsear lista de pesos desde texto
	pesos = [float(p) for p in w_str.split(",") if p.strip()]
	umbral = float(input("Umbral (t): ").strip())
	neu = NeuronaUmbral(pesos, umbral)

	print("Ingrese vectores de entrada separados por coma, o vacío para terminar.")
	while True:
		x_str = input("x = ").strip()
		if not x_str:
			break
		# Convertir la línea a un vector de entradas numéricas
		x = [float(v) for v in x_str.split(",") if v.strip()]
		y = neu.activar(x)
		print(f"Salida: {y}\n")


def main():
	print("=" * 60)
	print("038-E2: Computación neuronal (neuronas umbral)")
	print("=" * 60)
	print("1. DEMO\n2. INTERACTIVO")
	op = input("Seleccione (1/2, default=1): ").strip()
	if op == "2":
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()
