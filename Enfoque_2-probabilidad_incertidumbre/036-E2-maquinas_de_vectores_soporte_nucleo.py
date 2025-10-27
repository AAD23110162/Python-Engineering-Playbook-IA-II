"""
036-E2-maquinas_de_vectores_soporte_nucleo.py
----------------------------------------------
Este script presenta Máquinas de Vectores de Soporte (SVM) con núcleos:
- Clasificación con márgenes máximos y vectores soporte (enfoque perceptrón con kernel).
- Núcleos lineal, polinomial y RBF implementados a mano.
- Discute C, gamma y regularización a nivel conceptual.
- Relación con separabilidad no lineal mediante el "truco del kernel".
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: separación de clases con distintos núcleos en datos sintéticos.
2. INTERACTIVO: elección de hiperparámetros y evaluación de desempeño.

Autor: Alejandro Aguirre Díaz
"""

import math
import random
from typing import List, Tuple, Callable

# =============================================================================
# Núcleos (kernels)
# =============================================================================

def kernel_lineal(x: List[float], y: List[float]) -> float:
	"""Producto punto lineal."""
	# Calcula el producto escalar x·y = sum(x_i * y_i)
	return sum(a * b for a, b in zip(x, y))


def kernel_polinomial(x: List[float], y: List[float], grado: int = 2, c: float = 1.0) -> float:
	"""Kernel polinomial: (x·y + c)^grado."""
	# Primero calcula el producto punto lineal
	prod_punto = kernel_lineal(x, y)
	# Luego aplica la transformación polinomial (prod + c)^grado
	return (prod_punto + c) ** grado


def kernel_rbf(x: List[float], y: List[float], gamma: float = 1.0) -> float:
	"""Kernel RBF (gaussiano): exp(-gamma ||x-y||^2)."""
	# Calcula la distancia euclidiana al cuadrado entre x e y
	dist2 = sum((a - b) ** 2 for a, b in zip(x, y))
	# Aplica la función gaussiana exp(-gamma * ||x-y||^2)
	return math.exp(-gamma * dist2)


# =============================================================================
# Clasificador tipo SVM: Perceptrón con Kernel (sin librerías externas)
# =============================================================================

class KernelPerceptron:
	"""
	Clasificador perceptrón en espacio de características implícito vía kernel.
	- No resuelve el problema cuadrático de SVM; es una aproximación didáctica.
	- Aprende coeficientes alfa sobre cada ejemplo de entrenamiento.
	"""

	def __init__(
		self,
		kernel: str = "lineal",
		grado: int = 2,
		gamma: float = 1.0,
		c_pol: float = 1.0,
	):
		# Guardar hiperparámetros del kernel
		self.kernel_nombre = kernel
		self.grado = grado
		self.gamma = gamma
		self.c_pol = c_pol
		
		# Inicializar estructuras de datos para el modelo
		self.alfa: List[int] = []  # Conteos de actualizaciones por ejemplo
		self.X: List[List[float]] = []  # Conjunto de entrenamiento
		self.y: List[int] = []     # Etiquetas en {-1, +1}

		# Construir función kernel parametrizada según el tipo elegido
		if kernel == "lineal":
			# Kernel lineal: K(x,y) = x·y
			self.K: Callable[[List[float], List[float]], float] = kernel_lineal
		elif kernel == "polinomial":
			# Kernel polinomial con parámetros fijos
			self.K = lambda a, b: kernel_polinomial(a, b, grado=self.grado, c=self.c_pol)
		elif kernel == "rbf":
			# Kernel RBF (gaussiano) con gamma fijo
			self.K = lambda a, b: kernel_rbf(a, b, gamma=self.gamma)
		else:
			raise ValueError("Kernel desconocido. Use: lineal | polinomial | rbf")

	def ajustar(self, X: List[List[float]], y: List[int], epocas: int = 10):
		"""
		Ajusta el modelo mediante actualizaciones del perceptrón en espacio kernel.
		Args:
			X: Lista de vectores de características.
			y: Etiquetas en {-1, +1}.
			epocas: Número de pasadas sobre los datos.
		"""
		if not X:
			return
		
		# Guardar copia del conjunto de entrenamiento (necesario para predicción kernelizada)
		self.X = [xi[:] for xi in X]
		self.y = y[:]
		# Inicializar coeficientes alfa en cero para cada ejemplo
		self.alfa = [0 for _ in X]

		# Entrenamiento tipo perceptrón con kernel
		for _ in range(epocas):
			errores = 0
			# Iterar sobre cada ejemplo de entrenamiento
			for i, xi in enumerate(self.X):
				# Calcular el margen de decisión usando la función kernel
				margen = self._decision_function(xi)
				# Clasificar según el signo del margen
				pred = 1 if margen >= 0 else -1
				
				# Si hay error de clasificación, actualizar alfa_i
				if pred != self.y[i]:
					# Actualización: alfa_i += 1 (equivale a sumar y_i * x_i en espacio feature)
					self.alfa[i] += 1
					errores += 1
			
			# Si no hay errores, el modelo ha convergido
			if errores == 0:
				break

	def _decision_function(self, x: List[float]) -> float:
		"""f(x) = sum_i alfa_i * y_i * K(x_i, x)."""
		# Calcula el margen de decisión sumando las contribuciones de todos los ejemplos
		# Cada ejemplo aporta: alfa_i * y_i * K(x_i, x)
		# Solo ejemplos con alfa_i > 0 contribuyen (vectores soporte implícitos)
		return sum(ai * yi * self.K(xi, x) for ai, xi, yi in zip(self.alfa, self.X, self.y))

	def predecir(self, x: List[float]) -> int:
		"""Devuelve la etiqueta predicha en {-1, +1}."""
		# Clasificar según el signo de la función de decisión
		return 1 if self._decision_function(x) >= 0 else -1

	def predecir_lote(self, X: List[List[float]]) -> List[int]:
		"""Predice un lote de ejemplos."""
		# Aplica predicción a cada ejemplo en la lista
		return [self.predecir(x) for x in X]


# =============================================================================
# Utilidades: generación de datos sintéticos
# =============================================================================

def generar_datos_lineales(n: int = 60) -> Tuple[List[List[float]], List[int]]:
	"""Genera dos nubes 2D linealmente separables."""
	X: List[List[float]] = []
	y: List[int] = []
	random.seed(42)
	
	# Generar primera nube (clase +1) centrada en (1.5, 1.5)
	for _ in range(n // 2):
		x1 = random.gauss(1.5, 0.5)
		x2 = random.gauss(1.5, 0.5)
		X.append([x1, x2]); y.append(1)
	
	# Generar segunda nube (clase -1) centrada en (-1.5, -1.5)
	for _ in range(n // 2):
		x1 = random.gauss(-1.5, 0.5)
		x2 = random.gauss(-1.5, 0.5)
		X.append([x1, x2]); y.append(-1)
	
	return X, y


def generar_datos_circunferencias(n: int = 80) -> Tuple[List[List[float]], List[int]]:
	"""Genera datos 2D no linealmente separables: círculo interno vs externo."""
	X: List[List[float]] = []
	y: List[int] = []
	random.seed(7)
	
	# Círculo interno (radio ~1, clase +1)
	for _ in range(n // 2):
		ang = random.random() * 2 * math.pi  # Ángulo aleatorio [0, 2π]
		r = abs(random.gauss(1.0, 0.1))      # Radio cerca de 1
		# Convertir coordenadas polares a cartesianas
		X.append([r * math.cos(ang), r * math.sin(ang)])
		y.append(1)
	
	# Círculo externo (radio ~2, clase -1)
	for _ in range(n // 2):
		ang = random.random() * 2 * math.pi  # Ángulo aleatorio [0, 2π]
		r = abs(random.gauss(2.0, 0.1))      # Radio cerca de 2
		# Convertir coordenadas polares a cartesianas
		X.append([r * math.cos(ang), r * math.sin(ang)])
		y.append(-1)
	
	return X, y


def exactitud(y_real: List[int], y_pred: List[int]) -> float:
	# Cuenta las predicciones correctas
	aciertos = sum(1 for a, b in zip(y_real, y_pred) if a == b)
	# Retorna la proporción de aciertos
	return aciertos / len(y_real) if y_real else 0.0


# =============================================================================
# Modo DEMO
# =============================================================================

def modo_demo():
	print("MODO DEMO: Perceptrón con Kernel (SVM didáctico)\n")

	# ========== Caso 1: datos linealmente separables ==========
	# Generar dos nubes gaussianas bien separadas
	X_lin, y_lin = generar_datos_lineales(n=80)

	# Entrenar con kernel lineal (debería funcionar bien)
	modelo_lin = KernelPerceptron(kernel="lineal")
	modelo_lin.ajustar(X_lin, y_lin, epocas=20)
	y_pred_lin = modelo_lin.predecir_lote(X_lin)
	print("Caso lineal:")
	print(f"  Exactitud (kernel lineal): {exactitud(y_lin, y_pred_lin):.2%}")

	# ========== Caso 2: datos no lineales (círculos) ==========
	# Generar círculos concéntricos (NO separables linealmente)
	X_circ, y_circ = generar_datos_circunferencias(n=120)

	# Probar kernel lineal (no debería funcionar bien)
	modelo_lin2 = KernelPerceptron(kernel="lineal")
	modelo_lin2.ajustar(X_circ, y_circ, epocas=30)
	y_pred_lin2 = modelo_lin2.predecir_lote(X_circ)
	print("\nCaso no lineal (círculos):")
	print(f"  Exactitud (kernel lineal): {exactitud(y_circ, y_pred_lin2):.2%}")

	# Probar kernel polinomial grado 3 (mejor que lineal)
	modelo_poly = KernelPerceptron(kernel="polinomial", grado=3, c_pol=1.0)
	modelo_poly.ajustar(X_circ, y_circ, epocas=30)
	y_pred_poly = modelo_poly.predecir_lote(X_circ)
	print(f"  Exactitud (kernel polinomial grado 3): {exactitud(y_circ, y_pred_poly):.2%}")

	# Probar kernel RBF con gamma=2.0 (debería funcionar muy bien)
	modelo_rbf = KernelPerceptron(kernel="rbf", gamma=2.0)
	modelo_rbf.ajustar(X_circ, y_circ, epocas=30)
	y_pred_rbf = modelo_rbf.predecir_lote(X_circ)
	print(f"  Exactitud (kernel RBF gamma=2.0): {exactitud(y_circ, y_pred_rbf):.2%}")


# =============================================================================
# Modo INTERACTIVO
# =============================================================================

def modo_interactivo():
	print("MODO INTERACTIVO: Perceptrón con Kernel (SVM didáctico)\n")
	
	# ========== Selección del kernel ==========
	print("Seleccione kernel: 1=lineal, 2=polinomial, 3=rbf")
	opcion = input("Opción (1/2/3, default=3): ").strip() or "3"
	
	if opcion == "1":
		# Kernel lineal: K(x,y) = x·y
		kernel = "lineal"
		grado = 2
		gamma = 1.0
	elif opcion == "2":
		# Kernel polinomial: permite configurar el grado
		kernel = "polinomial"
		grado = int(input("Grado polinomial (default=3): ") or "3")
		gamma = 1.0
	else:
		# Kernel RBF: permite configurar gamma (controla el ancho de la gaussiana)
		kernel = "rbf"
		grado = 2
		gamma = float(input("Gamma RBF (default=1.0): ") or "1.0")

	# ========== Selección del conjunto de datos ==========
	print("\nSeleccione datos: 1=lineales, 2=círculos (no lineales)")
	datos_opt = input("Opción (1/2, default=2): ").strip() or "2"
	if datos_opt == "1":
		# Datos separables linealmente
		X, y = generar_datos_lineales(n=80)
	else:
		# Datos círculos concéntricos (no separables linealmente)
		X, y = generar_datos_circunferencias(n=120)

	# ========== Configuración de entrenamiento ==========
	epocas = int(input("Épocas de entrenamiento (default=20): ") or "20")

	# ========== Entrenamiento y evaluación ==========
	# Crear el modelo con los parámetros elegidos
	modelo = KernelPerceptron(kernel=kernel, grado=grado, gamma=gamma)
	# Entrenar el modelo
	modelo.ajustar(X, y, epocas=epocas)
	# Predecir sobre todo el conjunto
	y_pred = modelo.predecir_lote(X)
	# Calcular y mostrar la exactitud
	acc = exactitud(y, y_pred)
	print(f"\nExactitud en el conjunto completo: {acc:.2%}")


# =============================================================================
# Main
# =============================================================================

def main():
	print("=" * 60)
	print("036-E2: SVM con Núcleos (Perceptrón Kernel)")
	print("=" * 60)
	print("Seleccione el modo de ejecución:")
	print("1. DEMO: Datos sintéticos y núcleos")
	print("2. INTERACTIVO: Probar hiperparámetros")
	print("=" * 60)

	opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
	if opcion == "2":
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()
