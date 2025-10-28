"""
049-E2-recuperacion_de_datos.py
--------------------------------
Este script presenta Recuperación de Datos (Information Retrieval):
- Modelos booleano, vectorial y probabilístico a nivel conceptual.
- Representación TF-IDF y similitud de coseno.
- Métricas de evaluación: precisión, exhaustividad, MAP.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: recuperación en un pequeño índice de documentos.
2. INTERACTIVO: consultas y ranking sobre un conjunto de textos cargado por el usuario.

Autor: Alejandro Aguirre Díaz
"""

# Importar librerías necesarias
import re      # Expresiones regulares para tokenización
import math    # Operaciones matemáticas
from collections import Counter, defaultdict  # Contadores y diccionarios por defecto

# -----------------------------
# Tokenización simple
# -----------------------------
def tokenizar(texto):
	# Extrae palabras en minúsculas usando regex
	return re.findall(r"[a-záéíóúñü]+", texto.lower())

# -----------------------------
# Índice TF-IDF
# -----------------------------
class IndiceTFIDF:
	def __init__(self):
		self.doc_tokens = []   # Lista de listas de tokens por documento
		self.df = Counter()    # Frecuencia de documentos por término
		self.idf = defaultdict(float)  # IDF por término
		self.tfidf_doc = []    # Lista de vectores TF-IDF por documento
		self.N = 0             # Número total de documentos

	def construir(self, documentos):
		"""
		Construye el índice TF-IDF a partir de una lista de documentos (strings).
		"""
		# Tokenizar todos los documentos
		self.doc_tokens = [tokenizar(d) for d in documentos]
		self.N = len(self.doc_tokens)
		# Calcular DF: número de documentos que contienen cada término
		for tokens in self.doc_tokens:
			for t in set(tokens):
				self.df[t] += 1
		# Calcular IDF: log(N / df) suavizado
		for t, df_t in self.df.items():
			self.idf[t] = math.log((self.N + 1) / (df_t + 1)) + 1.0  # idf-smooth
		# Calcular vector TF-IDF por documento
		self.tfidf_doc = []
		for tokens in self.doc_tokens:
			tf = Counter(tokens)  # Frecuencia de términos en el documento
			vec = Counter()
			for t, f in tf.items():
				vec[t] = (f / len(tokens)) * self.idf[t]
			self.tfidf_doc.append(vec)

	def _coseno(self, v1, v2):
		# Calcula la similitud de coseno entre dos vectores
		# Numerador: producto punto
		numer = sum(v1[t] * v2.get(t, 0.0) for t in v1)
		# Denominadores: normas de los vectores
		n1 = math.sqrt(sum(x*x for x in v1.values()))
		n2 = math.sqrt(sum(x*x for x in v2.values()))
		if n1 == 0 or n2 == 0:
			return 0.0
		return numer / (n1 * n2)

	def buscar(self, consulta, k=5):
		"""
		Devuelve top-k documentos más similares a la consulta.
		"""
		# Tokenizar la consulta
		tokens = tokenizar(consulta)
		tf = Counter(tokens)  # Frecuencia de términos en la consulta
		vq = Counter()
		for t, f in tf.items():
			vq[t] = (f / max(1, len(tokens))) * self.idf.get(t, 0.0)
		# Puntuar todos los documentos usando similitud de coseno
		puntajes = []
		for idx, vd in enumerate(self.tfidf_doc):
			puntajes.append((idx, self._coseno(vq, vd)))
		# Ordenar por puntaje descendente
		puntajes.sort(key=lambda x: x[1], reverse=True)
		return puntajes[:k]

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: Búsqueda TF-IDF ---")
	# Documentos de ejemplo
	documentos = [
		"El gato come pescado y duerme en el sofá",
		"El perro persigue al gato en el jardín",
		"La gata juega con una pelota",
		"Perro y gato pueden ser amigos",
	]
	indice = IndiceTFIDF()
	indice.construir(documentos)
	consulta = "gato y perro"
	resultados = indice.buscar(consulta, k=3)
	print("Consulta:", consulta)
	# Mostrar resultados de la búsqueda
	for idx, score in resultados:
		print(f"[{idx}] score={score:.3f} - {documentos[idx]}")

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Índice y consultas ---")
	try:
		# Solicitar documentos al usuario
		n = int(input("Número de documentos: "))
		documentos = []
		for i in range(n):
			documentos.append(input(f"Doc {i+1}: "))
	except Exception:
		print("Usando colección por defecto.")
		documentos = [
			"recuperacion de informacion con tf idf",
			"modelos probabilisticos y medicion de similitud",
			"busqueda en documentos y ranking",
		]
	indice = IndiceTFIDF()
	indice.construir(documentos)
	while True:
		# Solicitar consulta al usuario
		q = input("Consulta (o 'q' para salir): ").strip().lower()
		if q == 'q':
			break
		resultados = indice.buscar(q, k=5)
		# Mostrar resultados de la búsqueda
		for idx, score in resultados:
			print(f"[{idx}] score={score:.3f} - {documentos[idx]}")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	# Menú principal para seleccionar el modo de ejecución
	print("\nScript 049-E2-recuperacion_de_datos.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (búsqueda TF-IDF)")
	print("2. INTERACTIVO (índice y consultas)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
