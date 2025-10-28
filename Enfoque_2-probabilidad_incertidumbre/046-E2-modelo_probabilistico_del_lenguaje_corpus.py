"""
046-E2-modelo_probabilistico_del_lenguaje_corpus.py
--------------------------------
Este script aborda el Modelo Probabilístico del Lenguaje desde el Corpus:
- Preparación de corpus: tokenización, normalización y particiones.
- Estimación de n-gramas y conteos con suavizado a nivel conceptual.
- Medición de perplejidad y evaluación intrínseca.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: construcción de un modelo de lenguaje simple sobre un corpus de ejemplo.
2. INTERACTIVO: carga de corpus propio y exploración de estadísticas.

Autor: Alejandro Aguirre Díaz
"""
import re
from collections import Counter, defaultdict
import math
import random

# -----------------------------
# Utilidades de texto
# -----------------------------
def tokenizar(texto):
	"""
	Tokeniza un texto en palabras minúsculas usando una expresión regular simple.
	"""
	return re.findall(r"[a-záéíóúñü]+", texto.lower())

def preparar_corpus(oraciones):
	"""
	Añade marcadores de inicio/fin para bigramas y tokeniza.
	"""
	corpus_tokens = []
	for oracion in oraciones:
		tokens = tokenizar(oracion)
		if not tokens:
			continue
		# Agregar tokens de inicio/fin para bigramas
		corpus_tokens.append(["<s>"] + tokens + ["</s>"])
	return corpus_tokens

# -----------------------------
# Modelo de Lenguaje n-gramas (n=1 o n=2)
# -----------------------------
class ModeloLenguaje:
	def __init__(self, n=2, suavizado_add1=True):
		"""
		Modelo de lenguaje simple con n-gramas (unigramas/bigramas).
		suavizado_add1: si True, aplica Laplace (add-one) en bigramas.
		"""
		assert n in (1, 2), "Solo se soporta n=1 o n=2 en este ejemplo"
		self.n = n
		self.suavizado_add1 = suavizado_add1
		self.unigramas = Counter()
		self.bigramas = Counter()
		self.vocab = set()
		self.total_unigramas = 0

	def entrenar(self, oraciones):
		"""
		Entrena el modelo con una lista de oraciones (listas de tokens con <s>, </s>).
		"""
		for tokens in oraciones:
			# Contar unigramas
			self.unigramas.update(tokens)
			# Contar bigramas si corresponde
			if self.n == 2:
				for i in range(len(tokens) - 1):
					self.bigramas[(tokens[i], tokens[i+1])] += 1
		self.vocab = set(self.unigramas.keys())
		self.total_unigramas = sum(self.unigramas.values())

	def prob_unigrama(self, w):
		"""
		P(w) = count(w) / total
		Si la palabra no existe, retorna una probabilidad pequeña (fuera de vocabulario).
		"""
		if w not in self.vocab:
			# Prob. para OOV: asignar masa muy pequeña
			return 1.0 / (self.total_unigramas + len(self.vocab) + 1)
		return self.unigramas[w] / self.total_unigramas

	def prob_bigrama(self, w_prev, w):
		"""
		P(w|w_prev) = count(w_prev, w) / count(w_prev)
		Con suavizado add-one opcional.
		"""
		count_prev = self.unigramas[w_prev]
		count_big = self.bigramas[(w_prev, w)]
		V = len(self.vocab)
		if self.suavizado_add1:
			return (count_big + 1) / (count_prev + V)
		# Sin suavizado
		if count_prev == 0:
			return 1.0 / V
		return count_big / count_prev

	def prob_secuencia(self, tokens):
		"""
		Devuelve la probabilidad (log) de una secuencia completa incluyendo <s> y </s>.
		"""
		if self.n == 1:
			logp = 0.0
			for w in tokens:
				logp += math.log(self.prob_unigrama(w) + 1e-12)
			return logp
		else:
			logp = 0.0
			for i in range(len(tokens) - 1):
				logp += math.log(self.prob_bigrama(tokens[i], tokens[i+1]) + 1e-12)
			return logp

	def generar(self, max_palabras=20):
		"""
		Genera una oración muestreando palabra a palabra.
		"""
		if self.n == 1:
			# Muestreo por distribución de unigramas (sin dependencia de contexto)
			vocab_sin_marcas = [w for w in self.vocab if w not in {"<s>", "</s>"}]
			probs = [self.prob_unigrama(w) for w in vocab_sin_marcas]
			s = []
			for _ in range(max_palabras):
				w = random.choices(vocab_sin_marcas, weights=probs, k=1)[0]
				s.append(w)
			return " ".join(s)
		else:
			# Comenzar en <s> y muestrear siguiente hasta </s>
			actual = "<s>"
			salida = []
			for _ in range(max_palabras):
				candidatos = list(self.vocab)
				# Evitar elegir <s> como siguiente normal
				candidatos = [w for w in candidatos if w != "<s>"]
				pesos = [self.prob_bigrama(actual, w) for w in candidatos]
				if sum(pesos) == 0:
					break
				w = random.choices(candidatos, weights=pesos, k=1)[0]
				if w == "</s>":
					break
				salida.append(w)
				actual = w
			return " ".join(salida)

def perplejidad(modelo, corpus_tokens):
	"""
	Calcula la perplejidad sobre un conjunto de oraciones tokenizadas con <s>, </s>.
	PP = exp( - (1/N) * sum log P(oración) ), N = número total de tokens generativos.
	"""
	logp_total = 0.0
	N = 0
	for tokens in corpus_tokens:
		logp_total += modelo.prob_secuencia(tokens)
		# Para bigramas, los eventos son pares; pero como métrica educativa, usamos |tokens|-1
		N += max(1, len(tokens) - 1 if modelo.n == 2 else len(tokens))
	return math.exp(-logp_total / max(1, N))

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: Modelo de lenguaje con bigramas ---")
	oraciones = [
		"El gato come pescado",
		"El perro come carne",
		"El gato duerme",
		"La gata come pescado fresco",
	]
	corpus = preparar_corpus(oraciones)
	modelo = ModeloLenguaje(n=2, suavizado_add1=True)
	modelo.entrenar(corpus)
	print("Vocabulario:", sorted(list(modelo.vocab))[:15], "...")
	print("Generación de ejemplo:")
	for _ in range(3):
		print(" -", modelo.generar())
	pp = perplejidad(modelo, corpus)
	print(f"Perplejidad (entrenamiento): {pp:.3f}")

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Entrena tu modelo de lenguaje ---")
	try:
		texto = input("Ingresa tu corpus (múltiples oraciones separadas por punto):\n> ")
		n = int(input("n-gramas (1 o 2): "))
		suav = input("Aplicar suavizado add-one? [s/n]: ").strip().lower() == 's'
		oraciones = [o.strip() for o in re.split(r"[\.!?]+", texto) if o.strip()]
	except Exception:
		print("Entrada inválida. Usando valores por defecto.")
		oraciones = ["hola mundo", "hola a todos", "mundo pequeño"]
		n = 2
		suav = True
	corpus = preparar_corpus(oraciones)
	modelo = ModeloLenguaje(n=n, suavizado_add1=suav)
	modelo.entrenar(corpus)
	while True:
		print("\nOpciones: [g]enerar, [p]erplejidad, [q]uitar")
		op = input("> ").strip().lower()
		if op == 'g':
			print("Generado:", modelo.generar())
		elif op == 'p':
			print(f"Perplejidad: {perplejidad(modelo, corpus):.3f}")
		elif op == 'q':
			break
		else:
			print("Opción no reconocida.")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	print("\nScript 046-E2-modelo_probabilistico_del_lenguaje_corpus.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (modelo de lenguaje sobre corpus de ejemplo)")
	print("2. INTERACTIVO (carga de corpus y opciones)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
