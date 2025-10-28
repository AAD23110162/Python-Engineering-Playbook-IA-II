"""
048-E2-gramaticas_probabilisticas_lexicalizadas.py
--------------------------------
Este script amplía PCFG a Gramáticas Probabilísticas Lexicalizadas:
- Asocia reglas con información léxica (cabezas, dependencias).
- Mejora del parseo al incorporar contexto léxico.
- Discute complejidad y estimación de parámetros.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo de derivaciones lexicalizadas.
2. INTERACTIVO: definición de reglas con léxico y evaluación de oraciones.

Autor: Alejandro Aguirre Díaz
"""
import random
import math

# -----------------------------
# Gramática lexicalizada (simplificada)
# -----------------------------
class PCFGLexicalizada:
	"""
	Gramática probabilística lexicalizada minimalista.
	- Reglas estructurales con probabilidad.
	- Distribuciones de cabeza por tipo de sintagma.
	"""
	def __init__(self):
		# Probabilidades de estructura (no lexicalizadas)
		# S -> NP VP
		self.p_S = 1.0
		# NP -> Det N | N
		self.p_NP = {('Det', 'N'): 0.7, ('N',): 0.3}
		# VP -> V NP | V
		self.p_VP = {('V', 'NP'): 0.6, ('V',): 0.4}

		# Distribuciones lexicalizadas (cabezas)
		self.head_N = {  # P(n|N)
			'gato': 0.4,
			'perro': 0.4,
			'niño': 0.2,
		}
		self.head_V = {  # P(v|V)
			've': 0.5,
			'persigue': 0.3,
			'muerde': 0.2,
		}
		self.head_Det = {  # P(det|Det)
			'el': 0.6,
			'la': 0.4,
		}

	def muestrear(self):
		"""
		Genera una oración lexicalizando las cabezas.
		"""
		# S -> NP VP (determinístico)
		np = self._gen_np()
		vp = self._gen_vp()
		return ' '.join(np + vp)

	def _gen_np(self):
		# Elegir estructura NP
		estructuras = list(self.p_NP.keys())
		pesos = [self.p_NP[e] for e in estructuras]
		e = random.choices(estructuras, weights=pesos, k=1)[0]
		if e == ('Det', 'N'):
			det = random.choices(list(self.head_Det), weights=list(self.head_Det.values()), k=1)[0]
			n = random.choices(list(self.head_N), weights=list(self.head_N.values()), k=1)[0]
			return [det, n]
		else:  # ('N',)
			n = random.choices(list(self.head_N), weights=list(self.head_N.values()), k=1)[0]
			return [n]

	def _gen_vp(self):
		estructuras = list(self.p_VP.keys())
		pesos = [self.p_VP[e] for e in estructuras]
		e = random.choices(estructuras, weights=pesos, k=1)[0]
		v = random.choices(list(self.head_V), weights=list(self.head_V.values()), k=1)[0]
		if e == ('V', 'NP'):
			np = self._gen_np()
			return [v] + np
		else:
			return [v]

	def puntuar(self, oracion):
		"""
		Pseudo-puntuación lexicalizada:
		- Asume S -> NP VP
		- Intenta segmentar como [NP] [VP] de forma heurística.
		- Suma log-probs de cabezas y estructuras usadas.
		(Educativo, no un parser completo.)
		"""
		# Heurística: si longitud >=3 y empieza con determinante, NP=2 palabras; si no, NP=1
		dets = set(self.head_Det.keys())
		if len(oracion) >= 2 and oracion[0] in dets:
			np = oracion[:2]
			resto = oracion[2:]
			estructura_np = ('Det', 'N')
			p_np = self.p_NP[estructura_np]
			p_det = self.head_Det.get(np[0], 1e-6)
			p_n = self.head_N.get(np[1], 1e-6)
		else:
			np = oracion[:1]
			resto = oracion[1:]
			estructura_np = ('N',)
			p_np = self.p_NP[estructura_np]
			p_det = 1.0
			p_n = self.head_N.get(np[0], 1e-6)

		# VP: si quedan >=2 palabras, asumimos V NP, si queda 1, solo V
		if len(resto) >= 2:
			v = resto[0]
			p_v = self.head_V.get(v, 1e-6)
			# NP en VP
			if len(resto) >= 3 and resto[1] in dets:
				p_np2 = self.p_NP[('Det', 'N')]
				p_det2 = self.head_Det.get(resto[1], 1e-6)
				p_n2 = self.head_N.get(resto[2], 1e-6)
			else:
				p_np2 = self.p_NP[('N',)]
				p_det2 = 1.0
				p_n2 = self.head_N.get(resto[1] if len(resto) > 1 else '<vacío>', 1e-6)
			p_estructura_vp = self.p_VP[('V', 'NP')]
			logp = math.log(self.p_S) + math.log(p_np) + math.log(p_det) + math.log(p_n) \
				   + math.log(p_estructura_vp) + math.log(p_v) + math.log(p_np2) + math.log(p_det2) + math.log(p_n2)
		else:
			v = resto[0] if resto else '<vacío>'
			p_v = self.head_V.get(v, 1e-6)
			p_estructura_vp = self.p_VP[('V',)]
			logp = math.log(self.p_S) + math.log(p_np) + math.log(p_det) + math.log(p_n) \
				   + math.log(p_estructura_vp) + math.log(p_v)
		return logp

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: PCFG lexicalizada (simplificada) ---")
	g = PCFGLexicalizada()
	print("Frases generadas:")
	for _ in range(3):
		print(" -", g.muestrear())
	oracion = "el perro ve el gato".split()
	print("\nPuntuación lexicalizada de:", ' '.join(oracion))
	print(f"log-prob ~ {g.puntuar(oracion):.3f}")

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Genera y puntúa oraciones ---")
	g = PCFGLexicalizada()
	while True:
		op = input("[g]enerar, [p]untuar, [q]uitar: ").strip().lower()
		if op == 'g':
			print(" -", g.muestrear())
		elif op == 'p':
			texto = input("Oración a puntuar: ").strip().lower()
			if not texto:
				continue
			print(f"log-prob ~ {g.puntuar(texto.split()):.3f}")
		elif op == 'q':
			break
		else:
			print("Opción no reconocida.")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	print("\nScript 048-E2-gramaticas_probabilisticas_lexicalizadas.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (generación y puntuación lexicalizada)")
	print("2. INTERACTIVO (opera con la gramática)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
