"""
026-E2-modelos_ocultos_de_markov.py
------------------------------------
Este script introduce Modelos Ocultos de Markov (HMM):
- Define estados ocultos, observaciones y matrices de transición/emisión.
- Implementa tareas de filtrado, predicción y decodificación (Viterbi) a nivel conceptual.
- Discute aplicaciones: habla, bioinformática, finanzas.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplos predefinidos de secuencias y decodificación.
2. INTERACTIVO: permite construir HMM simples y evaluar verosimilitudes.

Autor: Alejandro Aguirre Díaz
"""


import random
from typing import Dict, List, Tuple


class HMM:
	def __init__(self, estados: List[str], observaciones: List[str],
				 T: Dict[str, Dict[str, float]],
				 E: Dict[str, Dict[str, float]],
				 pi: Dict[str, float]):
		# S: conjunto de estados ocultos
		self.S = estados[:]
		# O: conjunto de símbolos de observación
		self.O = observaciones[:]
		# T[s][s'] = P(X_t=s' | X_{t-1}=s): matriz de transición
		self.T = {s: dict(p) for s, p in T.items()}
		# E[s][o] = P(e_t=o | X_t=s): matriz de emisión
		self.E = {s: dict(p) for s, p in E.items()}
		# pi[s] = P(X_0=s): distribución inicial
		self.pi = dict(pi)

	def _sample_from(self, dist: Dict[str, float]) -> str:
		# Muestra una clave según su probabilidad (inversa de la CDF discreta)
		r = random.random()
		acum = 0.0
		items = list(dist.items())
		for k, p in items:
			acum += p
			if r <= acum:
				return k
		# Por seguridad si el acumulado < 1 por redondeo, devuelve el último
		return items[-1][0]

	def generar(self, n: int) -> Tuple[List[str], List[str]]:
		"""Genera secuencias de estados y observaciones de longitud n."""
		# Listas donde guardaremos estados ocultos y observaciones emitidas
		estados = []
		observs = []
		# Paso inicial: muestrea estado inicial y su primera observación
		s = self._sample_from(self.pi)
		o = self._sample_from(self.E[s])
		estados.append(s)
		observs.append(o)
		for _ in range(1, n):
			# Evolución de estado según T y emisión según E
			s = self._sample_from(self.T[s])
			o = self._sample_from(self.E[s])
			estados.append(s)
			observs.append(o)
		return estados, observs

	def forward_loglike(self, obs_seq: List[str]) -> float:
		"""Devuelve el log-verosimilitud de la secuencia de observación."""
		# Forward con escalado para estabilidad numérica
		alfas = []
		escalas = []
		# Inicialización (t=0): combina prior con la primera observación
		e0 = obs_seq[0]
		a0 = {}
		z = 0.0
		for s in self.S:
			a0[s] = self.pi[s] * self.E[s][e0]
			z += a0[s]
		# Factor de escala c0 para evitar underflow
		c0 = z if z > 0 else 1.0
		for s in self.S:
			a0[s] = a0[s] / c0
		alfas.append(a0)
		escalas.append(c0)
		# Recursión (t>0): propaga hacia delante y escala en cada paso
		for t in range(1, len(obs_seq)):
			et = obs_seq[t]
			at = {}
			z = 0.0
			for s in self.S:
				# α_t(s) = [Σ_{s'} α_{t-1}(s') P(s|s')] P(e_t|s)
				at[s] = sum(alfas[t - 1][sp] * self.T[sp][s] for sp in self.S) * self.E[s][et]
				z += at[s]
			# Factor de escala ct para el tiempo t
			ct = z if z > 0 else 1.0
			for s in self.S:
				at[s] = at[s] / ct
			alfas.append(at)
			escalas.append(ct)
		# Log-verosimilitud: suma de logs de los factores de escala
		import math
		return -sum(math.log(c) for c in escalas)

	def viterbi(self, obs_seq: List[str]) -> Tuple[List[str], float]:
		"""Secuencia de estados más probable y su score (no normalizado)."""
		Tn = len(obs_seq)
		# delta[t][s]: mejor score hasta t terminando en s
		# psi[t][s]: mejor predecesor para reconstruir el camino
		delta = []
		psi = []
		e0 = obs_seq[0]
		d0 = {s: self.pi[s] * self.E[s][e0] for s in self.S}
		delta.append(d0)
		psi.append({s: None for s in self.S})
		for t in range(1, Tn):
			et = obs_seq[t]
			dt = {}
			psit = {}
			for s in self.S:
				# Elige el mejor predecesor s' que maximiza el score hacia s
				vals = [(delta[t - 1][sp] * self.T[sp][s] * self.E[s][et], sp) for sp in self.S]
				val, arg = max(vals, key=lambda x: x[0])
				dt[s] = val
				psit[s] = arg
			delta.append(dt)
			psi.append(psit)
		# Backtracking: estado final con mayor score y reconstrucción hacia atrás
		last = max(self.S, key=lambda s: delta[-1][s])
		score = delta[-1][last]
		path = [last]
		for t in range(Tn - 1, 0, -1):
			last = psi[t][last]
			path.append(last)
		path.reverse()
		return path, score


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: HMM (generación, forward, Viterbi)")
	print("=" * 70)
	# Modelo de lluvia-paraguas clásico (discreto)
	estados = ['Lluvia', 'NoLluvia']
	observ = ['Paraguas', 'NoParaguas']
	T = {
		'Lluvia':   {'Lluvia': 0.7, 'NoLluvia': 0.3},
		'NoLluvia': {'Lluvia': 0.3, 'NoLluvia': 0.7},
	}
	# E: probabilidad de observar paraguas según el clima
	E = {
		'Lluvia':   {'Paraguas': 0.9, 'NoParaguas': 0.1},
		'NoLluvia': {'Paraguas': 0.2, 'NoParaguas': 0.8},
	}
	pi = {'Lluvia': 0.5, 'NoLluvia': 0.5}

	hmm = HMM(estados, observ, T, E, pi)
	n = 12
	# Genera una secuencia sintética de longitud n
	est, obs = hmm.generar(n)
	print(f"\nSecuencia generada (n={n}):")
	print("  Estados:       ", est)
	print("  Observaciones: ", obs)

	# Calcula el log-likelihood de la secuencia observada bajo el modelo
	ll = hmm.forward_loglike(obs)
	print(f"\nLog-verosimilitud de las observaciones: {ll:.6f}")

	# Decodifica la secuencia de estados más probable (Viterbi)
	path, score = hmm.viterbi(obs)
	print("\nDecodificación Viterbi:")
	print("  Estados más probables:", path)
	print(f"  Score (no normalizado): {score:.6f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: HMM básico")
	print("=" * 70)
	try:
		n = int(input("Longitud de secuencia a generar: ").strip() or "10")
	except:
		n = 10
		print("Usando n=10 por defecto")
	# Reutiliza el mismo HMM sencillo de lluvia/paraguas
	estados = ['Lluvia', 'NoLluvia']
	observ = ['Paraguas', 'NoParaguas']
	T = {
		'Lluvia':   {'Lluvia': 0.7, 'NoLluvia': 0.3},
		'NoLluvia': {'Lluvia': 0.3, 'NoLluvia': 0.7},
	}
	E = {
		'Lluvia':   {'Paraguas': 0.9, 'NoParaguas': 0.1},
		'NoLluvia': {'Paraguas': 0.2, 'NoParaguas': 0.8},
	}
	pi = {'Lluvia': 0.5, 'NoLluvia': 0.5}

	hmm = HMM(estados, observ, T, E, pi)
	# Genera una trayectoria y muestra estados y observaciones
	est, obs = hmm.generar(n)
	print("\nGenerado:")
	print("  Estados:      ", est)
	print("  Observaciones:", obs)

	# Evalúa la verosimilitud de la trayectoria observada
	ll = hmm.forward_loglike(obs)
	print(f"Log-verosimilitud: {ll:.6f}")
	# Decodificación con Viterbi (secuencia de estados más probable)
	path, _ = hmm.viterbi(obs)
	print("Viterbi:")
	print("  ", path)


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("MODELOS OCULTOS DE MARKOV (HMM)")
	print("=" * 70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		modo_demo()
	print("\n" + "=" * 70)
	print("FIN DEL PROGRAMA")
	print("=" * 70 + "\n")


if __name__ == "__main__":
	main()


