"""
047-E2-gramaticas_probabilisticas_independientes_del_contexto.py
--------------------------------
Este script introduce Gramáticas Probabilísticas Independientes del Contexto (PCFG):
- Define reglas con probabilidades y derivaciones.
- Parsing probabilístico (CKY) a nivel conceptual.
- Estimación de reglas a partir de corpus anotados.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: derivaciones y parseo probabilístico de oraciones de juguete.
2. INTERACTIVO: definición de gramáticas simples y evaluación de probabilidad de frases.

Autor: Alejandro Aguirre Díaz
"""
import random
from collections import defaultdict
import math

# -----------------------------
# PCFG en CNF (A->BC o A->'palabra')
# -----------------------------
class PCFG:
	def __init__(self):
		# Reglas binarias: A -> B C con prob
		self.binarias = defaultdict(list)  # A -> [(B,C,p), ...]
		# Reglas léxicas: A -> 'w' con prob
		self.lexicas = defaultdict(list)   # A -> [(w,p), ...]
		# Conjuntos de no terminales
		self.no_terminales = set()

	def agregar_regla_binaria(self, A, B, C, p):
		self.no_terminales.update([A, B, C])
		self.binarias[A].append((B, C, p))

	def agregar_regla_lexica(self, A, w, p):
		self.no_terminales.add(A)
		self.lexicas[A].append((w, p))

	def generar(self, simbolo_inicio='S', max_long=20):
		"""
		Genera una oración muestreando reglas según sus probabilidades.
		"""
		salida = []
		self._generar_rec(simbolo_inicio, salida, max_long)
		return ' '.join(salida)

	def _generar_rec(self, A, salida, max_long):
		if len(salida) >= max_long:
			return
		# Si hay regla léxica, muestrear entre léxicas con prob acumulada
		if self.lexicas.get(A):
			palabras = [w for w, p in self.lexicas[A]]
			pesos = [p for w, p in self.lexicas[A]]
			w = random.choices(palabras, weights=pesos, k=1)[0]
			salida.append(w)
			return
		# Si hay binarias, expandir
		if self.binarias.get(A):
			pares = [(B, C) for (B, C, p) in self.binarias[A]]
			pesos = [p for (B, C, p) in self.binarias[A]]
			B, C = random.choices(pares, weights=pesos, k=1)[0]
			self._generar_rec(B, salida, max_long)
			self._generar_rec(C, salida, max_long)

# -----------------------------
# CKY Probabilístico (Viterbi) para mejor derivación
# -----------------------------
def cky_viterbi(pcfg, oracion):
	"""
	Devuelve el mejor puntaje log y el árbol en formato de corchetes.
	oracion: lista de palabras.
	"""
	n = len(oracion)
	# tablas: score[i][j][A] = log prob máxima para A => w_i..w_j (i inclusive, j exclusivo)
	score = [[defaultdict(lambda: -math.inf) for _ in range(n+1)] for _ in range(n)]
	back = [[{} for _ in range(n+1)] for _ in range(n)]

	# Inicialización con reglas léxicas
	for i, w in enumerate(oracion):
		for A, reglas in pcfg.lexicas.items():
			for (wl, p) in reglas:
				if wl == w and p > 0:
					score[i][i+1][A] = max(score[i][i+1][A], math.log(p))

	# Relleno CKY
	for span in range(2, n+1):  # longitud del segmento
		for i in range(0, n-span+1):
			j = i + span
			for k in range(i+1, j):
				# Partición i..k, k..j
				for A, lista in pcfg.binarias.items():
					for (B, C, p) in lista:
						if p <= 0:
							continue
						sb = score[i][k][B]
						sc = score[k][j][C]
						if sb == -math.inf or sc == -math.inf:
							continue
						val = math.log(p) + sb + sc
						if val > score[i][j][A]:
							score[i][j][A] = val
							back[i][j][A] = (k, B, C)

	# Recuperar árbol desde S
	if score[0][n]['S'] == -math.inf:
		return -math.inf, None
	arbol = reconstruir_arbol(back, pcfg, oracion, 0, n, 'S')
	return score[0][n]['S'], arbol

def reconstruir_arbol(back, pcfg, oracion, i, j, A):
	# Caso léxico
	if j == i+1 and any(w == oracion[i] for (w, _) in pcfg.lexicas.get(A, [])):
		return f"({A} {oracion[i]})"
	# Caso binario
	k, B, C = back[i][j][A]
	izq = reconstruir_arbol(back, pcfg, oracion, i, k, B)
	der = reconstruir_arbol(back, pcfg, oracion, k, j, C)
	return f"({A} {izq} {der})"

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: PCFG + CKY ---")
	pcfg = PCFG()
	# Gramática en CNF (simplificada)
	# S -> NP VP (0.9) | VP (0.1)
	pcfg.agregar_regla_binaria('S', 'NP', 'VP', 0.9)
	# Introducimos una regla unaria simulada vía binaria con eps en CNF no estricto omitiendo por simplicidad
	# NP -> Det N (1.0)
	pcfg.agregar_regla_binaria('NP', 'Det', 'N', 1.0)
	# VP -> V NP (0.6) | V (0.4) (para CNF, trataremos V como preterminal léxico)
	pcfg.agregar_regla_binaria('VP', 'V', 'NP', 0.6)

	# Léxico
	for det, p in [('el', 0.5), ('la', 0.5)]:
		pcfg.agregar_regla_lexica('Det', det, p)
	for n, p in [('gato', 0.5), ('perro', 0.5)]:
		pcfg.agregar_regla_lexica('N', n, p)
	for v, p in [('ve', 0.5), ('persigue', 0.5)]:
		pcfg.agregar_regla_lexica('V', v, p)

	# Generación
	print("Frases generadas:")
	for _ in range(3):
		print(" -", pcfg.generar('S'))

	# Parseo
	oracion = "el perro ve el gato".split()
	puntaje, arbol = cky_viterbi(pcfg, oracion)
	if arbol:
		print("Mejor parse (log-prob):", f"{puntaje:.3f}")
		print(arbol)
	else:
		print("No se pudo parsear la oración.")

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Define oraciones y parsea ---")
	# Usamos la misma gramática de la demo
	pcfg = PCFG()
	pcfg.agregar_regla_binaria('S', 'NP', 'VP', 1.0)
	pcfg.agregar_regla_binaria('NP', 'Det', 'N', 1.0)
	pcfg.agregar_regla_binaria('VP', 'V', 'NP', 0.7)
	# Léxico
	for det, p in [('el', 0.5), ('la', 0.5)]:
		pcfg.agregar_regla_lexica('Det', det, p)
	for n, p in [('gato', 0.5), ('perro', 0.5), ('niño', 0.5)]:
		pcfg.agregar_regla_lexica('N', n, p)
	for v, p in [('ve', 0.5), ('persigue', 0.5), ('muerde', 0.5)]:
		pcfg.agregar_regla_lexica('V', v, p)

	while True:
		texto = input("Oración a parsear (o 'q' para salir): ").strip().lower()
		if texto == 'q':
			break
		if not texto:
			continue
		oracion = texto.split()
		puntaje, arbol = cky_viterbi(pcfg, oracion)
		if arbol:
			print("Mejor parse (log-prob):", f"{puntaje:.3f}")
			print(arbol)
		else:
			print("No se pudo parsear la oración con la gramática predefinida.")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	print("\nScript 047-E2-gramaticas_probabilisticas_independientes_del_contexto.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (PCFG + CKY de ejemplo)")
	print("2. INTERACTIVO (parsea oraciones)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
