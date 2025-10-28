"""
045-E2-hamming_hopfield_hebb_boltzmann.py
--------------------------------
Este script presenta redes clásicas: Hamming, Hopfield, Hebb y Boltzmann:
- Hamming/Hopfield: memorias asociativas y estados de energía.
- Hebb: regla de aprendizaje hebbiano.
- Boltzmann: redes estocásticas con distribución en equilibrio.
- Discute convergencia y mínimos locales.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: recuperación de patrones en una red de Hopfield.
2. INTERACTIVO: definición de patrones y observación de dinámica.

Autor: Alejandro Aguirre Díaz
"""


# Importar librerías necesarias
import numpy as np  # Para operaciones numéricas y manejo de arrays
import random  # Para generación de números aleatorios

# -----------------------------
# Red de Hamming
# -----------------------------
class RedHamming:
	def __init__(self, patrones):
		"""
		Inicializa la red de Hamming con patrones binarios.
		"""
		# Almacena los patrones como array de numpy
		self.patrones = np.array(patrones)

	def reconocer(self, entrada):
		"""
		Devuelve el patrón almacenado más cercano (mayor coincidencia).
		"""
		# Calcula la cantidad de coincidencias entre la entrada y cada patrón
		distancias = [np.sum(entrada == p) for p in self.patrones]
		# Selecciona el patrón con mayor coincidencia
		idx = np.argmax(distancias)
		return self.patrones[idx]

# -----------------------------
# Red de Hopfield
# -----------------------------
class RedHopfield:
	def __init__(self, patrones):
		"""
		Inicializa la red de Hopfield y calcula la matriz de pesos.
		"""
		self.n = len(patrones[0])  # Número de neuronas
		self.patrones = np.array(patrones)
		self.pesos = np.zeros((self.n, self.n))  # Matriz de pesos
		# Aprendizaje hebbiano: suma de productos externos de cada patrón
		for p in self.patrones:
			self.pesos += np.outer(p, p)
		# Eliminar auto-conexiones (diagonal)
		np.fill_diagonal(self.pesos, 0)
		# Normalizar por el número de neuronas
		self.pesos /= self.n

	def recuperar(self, entrada, pasos=10):
		"""
		Recupera un patrón a partir de una entrada ruidosa.
		"""
		estado = np.array(entrada)  # Copia del estado inicial
		for _ in range(pasos):
			for i in range(self.n):
				# Calcula la suma ponderada de las entradas
				suma = np.dot(self.pesos[i], estado)
				# Actualiza el estado según el signo de la suma
				estado[i] = 1 if suma >= 0 else -1
		return estado

# -----------------------------
# Red de Hebb
# -----------------------------
class RedHebb:
	def __init__(self, n):
		"""
		Inicializa la red Hebbiana con n entradas.
		"""
		self.n = n  # Número de entradas
		self.pesos = np.zeros(n)  # Vector de pesos

	def entrenar(self, entradas, salidas):
		"""
		Aplica la regla de Hebb para ajustar los pesos.
		"""
		# Para cada par entrada-salida, ajusta los pesos
		for x, y in zip(entradas, salidas):
			self.pesos += x * y

	def predecir(self, entrada):
		"""
		Predice la salida para una entrada dada.
		"""
		# Calcula el producto punto y aplica la función signo
		return np.sign(np.dot(self.pesos, entrada))

# -----------------------------
# Máquina de Boltzmann
# -----------------------------
class MaquinaBoltzmann:
	def __init__(self, n):
		"""
		Inicializa la máquina de Boltzmann con n neuronas.
		"""
		self.n = n  # Número de neuronas
		# Inicializa la matriz de pesos con valores pequeños aleatorios
		self.pesos = np.random.randn(n, n) * 0.1
		np.fill_diagonal(self.pesos, 0)  # Sin auto-conexión

	def energia(self, estado):
		"""
		Calcula la energía del estado actual.
		"""
		# Fórmula de energía de Hopfield/Boltzmann
		return -0.5 * np.dot(estado, np.dot(self.pesos, estado))

	def muestrear(self, estado, temperatura=1.0, pasos=100):
		"""
		Realiza muestreo estocástico para encontrar estados de baja energía.
		"""
		estado = np.array(estado)  # Copia del estado inicial
		for _ in range(pasos):
			# Selecciona una neurona aleatoria
			i = random.randint(0, self.n - 1)
			# Calcula el cambio de energía si se invierte la neurona
			delta_E = 2 * estado[i] * np.dot(self.pesos[i], estado)
			# Decide si acepta el cambio según la probabilidad de Metropolis
			if delta_E < 0 or random.random() < np.exp(-delta_E / temperatura):
				estado[i] *= -1
		return estado

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: Recuperación de patrones en Hopfield ---")
	# Definir patrones binarios (+1/-1)
	patrones = [np.array([1, -1, 1, -1]), np.array([-1, 1, -1, 1])]
	# Crear red de Hopfield y mostrar recuperación
	hopfield = RedHopfield(patrones)
	entrada = np.array([1, -1, -1, -1])  # Entrada ruidosa
	print(f"Entrada ruidosa: {entrada}")
	recuperado = hopfield.recuperar(entrada)
	print(f"Patrón recuperado: {recuperado}")
	# Red de Hamming: reconocimiento por coincidencia
	hamming = RedHamming(patrones)
	reconocida = hamming.reconocer(entrada)
	print(f"Patrón reconocido por Hamming: {reconocida}")
	# Red Hebbiana: entrenamiento y predicción
	hebb = RedHebb(4)
	entradas = [np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])]
	salidas = [1, -1]
	hebb.entrenar(entradas, salidas)
	pred = hebb.predecir(np.array([1, 0, 1, 0]))
	print(f"Predicción Hebbiana: {pred}")
	# Máquina de Boltzmann: muestreo estocástico
	boltz = MaquinaBoltzmann(4)
	estado_inicial = np.array([1, 1, -1, -1])
	estado_final = boltz.muestrear(estado_inicial, temperatura=1.0, pasos=100)
	print(f"Estado Boltzmann final: {estado_final}")

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Define tus propios patrones ---")
	try:
		# Solicitar parámetros al usuario
		n = int(input("Número de neuronas (ej. 4): "))
		num_patrones = int(input("Número de patrones: "))
		patrones = []
		for i in range(num_patrones):
			patron = input(f"Patrón {i+1} (separado por espacios, +1/-1): ")
			patron = np.array([int(x) for x in patron.strip().split()])
			patrones.append(patron)
		entrada = input("Entrada ruidosa para recuperar (separado por espacios, +1/-1): ")
		entrada = np.array([int(x) for x in entrada.strip().split()])
	except Exception as e:
		print("Error en la entrada, usando valores por defecto.")
		patrones = [np.array([1, -1, 1, -1]), np.array([-1, 1, -1, 1])]
		entrada = np.array([1, -1, -1, -1])
	# Recuperación y reconocimiento con Hopfield y Hamming
	hopfield = RedHopfield(patrones)
	recuperado = hopfield.recuperar(entrada)
	print(f"Patrón recuperado por Hopfield: {recuperado}")
	hamming = RedHamming(patrones)
	reconocida = hamming.reconocer(entrada)
	print(f"Patrón reconocido por Hamming: {reconocida}")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	print("\nScript 045-E2-hamming_hopfield_hebb_boltzmann.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (recuperación de patrones en Hopfield)")
	print("2. INTERACTIVO (define tus propios patrones)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
