"""
026-E2-modelos_ocultos_de_markov.py
-----------------------------------
Este script implementa un HMM básico (Modelo Oculto de Markov):
- Generación de secuencias de estados y observaciones.
- Cálculo de verosimilitud por forward.
- Decodificación Viterbi para la secuencia más probable de estados ocultos.

Modos de ejecución:
1. DEMO: genera una secuencia y la decodifica con Viterbi.
2. INTERACTIVO: permite fijar longitud de secuencia y ver resultados.

Autor: Alejandro Aguirre Díaz
"""

import random
from typing import Dict, List, Tuple


class HMM:
	def __init__(self, estados: List[str], observaciones: List[str],
				 T: Dict[str, Dict[str, float]],
				 E: Dict[str, Dict[str, float]],
				 pi: Dict[str, float]):
		self.S = estados[:]
		self.O = observaciones[:]
		self.T = {s: dict(p) for s, p in T.items()}
		self.E = {s: dict(p) for s, p in E.items()}
		self.pi = dict(pi)

	def _sample_from(self, dist: Dict[str, float]) -> str:
		r = random.random()
		acum = 0.0
		items = list(dist.items())
		for k, p in items:
			acum += p
			if r <= acum:
				return k
		return items[-1][0]

	def generar(self, n: int) -> Tuple[List[str], List[str]]:
		"""Genera secuencias de estados y observaciones de longitud n."""
		estados = []
		observs = []
		s = self._sample_from(self.pi)
		o = self._sample_from(self.E[s])
		estados.append(s)
		observs.append(o)
		for _ in range(1, n):
			s = self._sample_from(self.T[s])
			o = self._sample_from(self.E[s])
			estados.append(s)
			observs.append(o)
		return estados, observs

	def forward_loglike(self, obs_seq: List[str]) -> float:
		"""Devuelve el log-verosimilitud de la secuencia de observación."""
		# Forward con escalado
		alfas = []
		escalas = []
		e0 = obs_seq[0]
		a0 = {}
		z = 0.0
		for s in self.S:
			a0[s] = self.pi[s] * self.E[s][e0]
			z += a0[s]
		c0 = z if z > 0 else 1.0
		for s in self.S:
			a0[s] = a0[s] / c0
		alfas.append(a0)
		escalas.append(c0)
		for t in range(1, len(obs_seq)):
			et = obs_seq[t]
			at = {}
			z = 0.0
			for s in self.S:
				at[s] = sum(alfas[t - 1][sp] * self.T[sp][s] for sp in self.S) * self.E[s][et]
				z += at[s]
			ct = z if z > 0 else 1.0
			for s in self.S:
				at[s] = at[s] / ct
			alfas.append(at)
			escalas.append(ct)
		import math
		return -sum(math.log(c) for c in escalas)

	def viterbi(self, obs_seq: List[str]) -> Tuple[List[str], float]:
		"""Secuencia de estados más probable y su score (no normalizado)."""
		Tn = len(obs_seq)
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
				vals = [(delta[t - 1][sp] * self.T[sp][s] * self.E[s][et], sp) for sp in self.S]
				val, arg = max(vals, key=lambda x: x[0])
				dt[s] = val
				psit[s] = arg
			delta.append(dt)
			psi.append(psit)
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
	n = 12
	est, obs = hmm.generar(n)
	print(f"\nSecuencia generada (n={n}):")
	print("  Estados:       ", est)
	print("  Observaciones: ", obs)

	ll = hmm.forward_loglike(obs)
	print(f"\nLog-verosimilitud de las observaciones: {ll:.6f}")

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
	est, obs = hmm.generar(n)
	print("\nGenerado:")
	print("  Estados:      ", est)
	print("  Observaciones:", obs)

	ll = hmm.forward_loglike(obs)
	print(f"Log-verosimilitud: {ll:.6f}")
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

"""
026-E2-modelos_ocultos_de_markov.py
--------------------------------
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
