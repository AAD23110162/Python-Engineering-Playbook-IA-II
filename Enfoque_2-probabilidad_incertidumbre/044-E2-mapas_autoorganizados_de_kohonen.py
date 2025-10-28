"""
044-E2-mapas_autoorganizados_de_kohonen.py
Este script introduce los Mapas Autoorganizados de Kohonen (SOM):

El programa puede ejecutarse en dos modos:
1. DEMO: entrenamiento de un SOM en datos 2D sintéticos.
2. INTERACTIVO: configuración de tamaño de mapa y parámetros de entrenamiento.

Autor: Alejandro Aguirre Díaz
"""

# Importar librerías necesarias para cálculos y visualización
import numpy as np  # Para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Para graficar
import random  # Para generación de datos aleatorios

# Clase para el Mapa Autoorganizado de Kohonen (SOM)
class MapaKohonen:
	def __init__(self, filas, columnas, dimension_entrada, tasa_aprendizaje=0.5, radio_inicial=2.0, decaimiento=0.99):
		"""
		Inicializa el mapa SOM con pesos aleatorios.
		filas, columnas: tamaño del mapa
		dimension_entrada: dimensión de los datos de entrada
		tasa_aprendizaje: velocidad de ajuste de los pesos
		radio_inicial: radio de vecindad inicial
		decaimiento: factor de reducción por época
		"""
		# Número de filas y columnas del mapa
		self.filas = filas
		self.columnas = columnas
		# Dimensión de los datos de entrada
		self.dimension_entrada = dimension_entrada
		# Tasa de aprendizaje inicial
		self.tasa_aprendizaje = tasa_aprendizaje
		# Radio de vecindad inicial
		self.radio = radio_inicial
		# Factor de decaimiento por época
		self.decaimiento = decaimiento
		# Inicializar pesos aleatorios para cada neurona del mapa
		self.pesos = np.random.rand(filas, columnas, dimension_entrada)

	def encontrar_ganador(self, entrada):
		"""
		Encuentra la neurona ganadora (BMU) más cercana al vector de entrada.
		"""
		# Calcular la distancia euclídea entre la entrada y cada neurona del mapa
		distancias = np.linalg.norm(self.pesos - entrada, axis=2)
		# Encontrar el índice de la neurona con menor distancia (BMU)
		indice = np.unravel_index(np.argmin(distancias), distancias.shape)
		return indice

	def actualizar_pesos(self, entrada, ganador):
		"""
		Actualiza los pesos de la neurona ganadora y su vecindad.
		"""
		# Recorrer todas las neuronas del mapa
		for i in range(self.filas):
			for j in range(self.columnas):
				# Calcular distancia en el mapa entre la neurona y el ganador
				distancia = np.sqrt((i - ganador[0])**2 + (j - ganador[1])**2)
				# Solo actualizar si está dentro del radio de vecindad
				if distancia <= self.radio:
					# Calcular el factor de influencia según la distancia
					influencia = np.exp(-distancia**2 / (2 * (self.radio**2)))
					# Actualizar los pesos acercándolos al vector de entrada
					self.pesos[i, j] += self.tasa_aprendizaje * influencia * (entrada - self.pesos[i, j])

	def entrenar(self, datos, epocas=100):
		"""
		Entrena el SOM con los datos proporcionados.
		"""
		for epoca in range(epocas):
			# Mezclar datos en cada época para evitar orden fijo
			random.shuffle(datos)
			for entrada in datos:
				# Encontrar la neurona ganadora para cada entrada
				ganador = self.encontrar_ganador(entrada)
				# Actualizar los pesos de la vecindad
				self.actualizar_pesos(entrada, ganador)
			# Decaer tasa de aprendizaje y radio de vecindad
			self.tasa_aprendizaje *= self.decaimiento
			self.radio *= self.decaimiento
			# Mostrar progreso cada cierto número de épocas
			if epoca % max(1, epocas // 10) == 0:
				print(f"Época {epoca+1}/{epocas} - tasa: {self.tasa_aprendizaje:.3f}, radio: {self.radio:.3f}")

	def mapear(self, datos):
		"""
		Mapea cada dato a la posición de su neurona ganadora en el mapa.
		"""
		posiciones = []  # Lista para guardar la posición de la neurona ganadora de cada dato
		for entrada in datos:
			ganador = self.encontrar_ganador(entrada)
			posiciones.append(ganador)
		return posiciones

	def mostrar_mapa(self):
		"""
		Visualiza los pesos del mapa SOM con etiquetas para la leyenda.
		"""
		if self.dimension_entrada == 2:
			plt.figure(figsize=(6,6))
			primer = True
			# Graficar cada neurona del mapa como un punto azul
			for i in range(self.filas):
				for j in range(self.columnas):
					if primer:
						plt.scatter(self.pesos[i, j, 0], self.pesos[i, j, 1], c='blue', marker='o', label='Pesos SOM')
						primer = False
					else:
						plt.scatter(self.pesos[i, j, 0], self.pesos[i, j, 1], c='blue', marker='o')
			plt.title('Mapa SOM (pesos finales)')
			plt.xlabel('X')
			plt.ylabel('Y')
			plt.grid(True)
			plt.legend()
			plt.show()
		else:
			print("Visualización solo disponible para datos 2D.")

# Función para generar datos sintéticos 2D
def generar_datos_sinteticos(n=200):
	"""
	Genera datos 2D en forma de anillos y grupos para demo.
	"""
	datos = []  # Lista para almacenar los datos generados
	# Generar puntos en forma de anillo
	for _ in range(n//2):
		ang = random.uniform(0, 2*np.pi)  # Ángulo aleatorio
		r = random.uniform(0.7, 1.0)  # Radio aleatorio
		x = r * np.cos(ang)
		y = r * np.sin(ang)
		datos.append(np.array([x, y]))
	# Generar puntos agrupados cerca del origen
	for _ in range(n//2):
		x = random.gauss(0, 0.3)  # Coordenada x con distribución normal
		y = random.gauss(0, 0.3)  # Coordenada y con distribución normal
		datos.append(np.array([x, y]))
	return datos

# Modo DEMO
def modo_demo():
	print("\n--- MODO DEMO: Entrenamiento SOM en datos sintéticos 2D ---")
	datos = generar_datos_sinteticos(200)  # Generar datos sintéticos para la demostración
	som = MapaKohonen(filas=10, columnas=10, dimension_entrada=2, tasa_aprendizaje=0.5, radio_inicial=3.0, decaimiento=0.98)  # Crear el SOM con parámetros estándar
	som.entrenar(datos, epocas=50)  # Entrenar el SOM
	posiciones = som.mapear(datos)  # Mapear cada dato a su neurona ganadora
	# Visualizar los datos originales y las neuronas ganadoras
	plt.figure(figsize=(7,7))
	xs = [d[0] for d in datos]
	ys = [d[1] for d in datos]
	plt.scatter(xs, ys, c='gray', alpha=0.5, label='Datos')  # Graficar los datos originales
	# Graficar las posiciones de las neuronas ganadoras
	primer = True
	for pos in posiciones:
		peso = som.pesos[pos[0], pos[1]]
		if primer:
			plt.scatter(peso[0], peso[1], c='red', marker='x', label='Neurona ganadora')
			primer = False
		else:
			plt.scatter(peso[0], peso[1], c='red', marker='x')
	som.mostrar_mapa()  # Mostrar el mapa de pesos finales
	plt.title('Datos y neuronas ganadoras (SOM)')
	plt.legend()
	plt.show()

# Modo INTERACTIVO
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Configura tu propio SOM ---")
	try:
		# Solicitar parámetros al usuario
		filas = int(input("Filas del mapa (ej. 10): "))
		columnas = int(input("Columnas del mapa (ej. 10): "))
		dimension = int(input("Dimensión de los datos (2 recomendado): "))
		tasa = float(input("Tasa de aprendizaje inicial (ej. 0.5): "))
		radio = float(input("Radio inicial de vecindad (ej. 3.0): "))
		decaimiento = float(input("Decaimiento por época (ej. 0.98): "))
		epocas = int(input("Número de épocas de entrenamiento (ej. 50): "))
	except Exception as e:
		# Si hay error en la entrada, usar valores por defecto
		print("Error en la entrada, usando valores por defecto.")
		filas, columnas, dimension, tasa, radio, decaimiento, epocas = 10, 10, 2, 0.5, 3.0, 0.98, 50
	# Para datos 2D, generar datos sintéticos
	if dimension == 2:
		datos = generar_datos_sinteticos(200)
	else:
		print("Genera tus propios datos para dimensiones >2.")
		return
	# Crear y entrenar el SOM con los parámetros elegidos
	som = MapaKohonen(filas, columnas, dimension, tasa, radio, decaimiento)
	som.entrenar(datos, epocas)
	posiciones = som.mapear(datos)
	# Visualizar los datos y las neuronas ganadoras
	plt.figure(figsize=(7,7))
	xs = [d[0] for d in datos]
	ys = [d[1] for d in datos]
	plt.scatter(xs, ys, c='gray', alpha=0.5, label='Datos')
	primer = True
	for pos in posiciones:
		peso = som.pesos[pos[0], pos[1]]
		if primer:
			plt.scatter(peso[0], peso[1], c='red', marker='x', label='Neurona ganadora')
			primer = False
		else:
			plt.scatter(peso[0], peso[1], c='red', marker='x')
	som.mostrar_mapa()
	plt.title('Datos y neuronas ganadoras (SOM)')
	plt.legend()
	plt.show()

# Menú principal
if __name__ == "__main__":
	# Menú principal para seleccionar el modo de ejecución
	print("\nScript 044-E2-mapas_autoorganizados_de_kohonen.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (entrenamiento SOM en datos sintéticos)")
	print("2. INTERACTIVO (configura tu propio SOM)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
