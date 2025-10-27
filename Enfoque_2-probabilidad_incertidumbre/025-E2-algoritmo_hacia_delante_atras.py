"""
025-E2-algoritmo_hacia_delante_atras.py
--------------------------------------
Este script implementa el algoritmo hacia delante/atrás (forward-backward) para HMM:
- Forward: calcula alfas normalizados y verosimilitud de la secuencia.
- Backward: calcula betas normalizados.
- Forward-Backward: calcula distribuciones suavizadas P(X_t|e_1:T).

Modos de ejecución:
1. DEMO: HMM lluvia/paraguas con una secuencia fija de observaciones.
2. INTERACTIVO: permite ingresar una secuencia y obtener alfas, betas y suavizados.

Autor: Alejandro Aguirre Díaz
"""

from typing import Dict, List, Tuple


def forward(estados: List[str], observaciones: List[str],
			T: Dict[str, Dict[str, float]],
			E: Dict[str, Dict[str, float]],
			pi: Dict[str, float],
			obs_seq: List[str]) -> Tuple[List[Dict[str, float]], List[float], float]:
	"""
	Retorna (alfas, escalas, log_likelihood)
	- alfas[t][s] ∝ P(e_1:t, X_t=s)
	- escalas[t]: factor de normalización de alfas[t]
	- log_likelihood = -sum(log(escalas[t]))
	"""
	alfas: List[Dict[str, float]] = []
	escalas: List[float] = []

	# t=0
	e0 = obs_seq[0]
	a0 = {}
	z = 0.0
	for s in estados:
		a0[s] = pi[s] * E[s][e0]
		z += a0[s]
	c0 = z if z > 0 else 1.0
	for s in estados:
		a0[s] = a0[s] / c0
	alfas.append(a0)
	escalas.append(c0)

	# t>0
	for t in range(1, len(obs_seq)):
		et = obs_seq[t]
		at = {}
		z = 0.0
		for s in estados:
			at[s] = sum(alfas[t - 1][sp] * T[sp][s] for sp in estados) * E[s][et]
			z += at[s]
		ct = z if z > 0 else 1.0
		for s in estados:
			at[s] = at[s] / ct
		alfas.append(at)
		escalas.append(ct)

	# log-likelihood
	import math
	log_likelihood = -sum(math.log(c) for c in escalas)
	return alfas, escalas, log_likelihood


def backward(estados: List[str], observaciones: List[str],
			 T: Dict[str, Dict[str, float]],
			 E: Dict[str, Dict[str, float]],
			 obs_seq: List[str],
			 escalas: List[float]) -> List[Dict[str, float]]:
	"""Retorna betas normalizados usando los factores de escala de forward."""
	Tn = len(obs_seq)
	betas: List[Dict[str, float]] = [{s: 1.0 for s in estados} for _ in range(Tn)]
	for t in range(Tn - 2, -1, -1):
		et1 = obs_seq[t + 1]
		bt = {}
		for s in estados:
			bt[s] = sum(T[s][s2] * E[s2][et1] * betas[t + 1][s2] for s2 in estados)
		# normalizar con la misma escala ct+1 usada en forward para estabilidad
		c = escalas[t + 1] if escalas[t + 1] > 0 else 1.0
		for s in estados:
			bt[s] = bt[s] / c
		betas[t] = bt
	return betas


def forward_backward(estados: List[str], observaciones: List[str],
					 T: Dict[str, Dict[str, float]],
					 E: Dict[str, Dict[str, float]],
					 pi: Dict[str, float],
					 obs_seq: List[str]) -> Tuple[List[Dict[str, float]], float]:
	"""Retorna (suavizados, log_likelihood)."""
	alfas, escalas, log_like = forward(estados, observaciones, T, E, pi, obs_seq)
	betas = backward(estados, observaciones, T, E, obs_seq, escalas)
	suav = []
	for t in range(len(obs_seq)):
		bel = {}
		z = 0.0
		for s in estados:
			bel[s] = alfas[t][s] * betas[t][s]
			z += bel[s]
		for s in estados:
			bel[s] = bel[s] / z if z > 0 else 0.0
		suav.append(bel)
	return suav, log_like


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Algoritmo Hacia Delante/Atrás (HMM)")
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

	obs_seq = ['Paraguas', 'Paraguas', 'NoParaguas', 'Paraguas']
	print(f"\nObservaciones: {obs_seq}")

	alfas, escalas, ll = forward(estados, observ, T, E, pi, obs_seq)
	print("\nForward (alfas normalizados):")
	for t, a in enumerate(alfas, 1):
		print(f"  t={t}: Lluvia={a['Lluvia']:.3f}, NoLluvia={a['NoLluvia']:.3f}")
	print(f"Log-verosimilitud: {ll:.6f}")

	betas = backward(estados, observ, T, E, obs_seq, escalas)
	print("\nBackward (betas normalizados):")
	for t, b in enumerate(betas, 1):
		print(f"  t={t}: Lluvia={b['Lluvia']:.3f}, NoLluvia={b['NoLluvia']:.3f}")

	suav, _ = forward_backward(estados, observ, T, E, pi, obs_seq)
	print("\nSuavizado (posteriores P(X_t|e_1:T)):")
	for t, bel in enumerate(suav, 1):
		print(f"  t={t}: Lluvia={bel['Lluvia']:.3f}, NoLluvia={bel['NoLluvia']:.3f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: Forward-Backward")
	print("=" * 70)
	print("Ingrese observaciones separadas por comas: Paraguas o NoParaguas")

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

	try:
		linea = input("\nObservaciones: ").strip()
		obs_seq = [x.strip() for x in linea.split(',') if x.strip()]
		if not obs_seq:
			obs_seq = ['Paraguas', 'NoParaguas', 'Paraguas']
	except:
		obs_seq = ['Paraguas', 'NoParaguas', 'Paraguas']

	alfas, escalas, ll = forward(estados, observ, T, E, pi, obs_seq)
	betas = backward(estados, observ, T, E, obs_seq, escalas)
	suav, _ = forward_backward(estados, observ, T, E, pi, obs_seq)

	print("\nAlfas:")
	for t, a in enumerate(alfas, 1):
		print(f"  t={t}: {a}")
	print("\nBetas:")
	for t, b in enumerate(betas, 1):
		print(f"  t={t}: {b}")
	print("\nSuavizados:")
	for t, bel in enumerate(suav, 1):
		print(f"  t={t}: {bel}")
	print(f"\nLog-verosimilitud: {ll:.6f}")


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("ALGORITMO HACIA DELANTE / ATRÁS")
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
025-E2-algoritmo_hacia_delante_atras.py
--------------------------------
Este script implementa el Algoritmo Hacia Delante-Atrás (Forward-Backward) para HMM:
- Calcula mensajes hacia delante (α) y hacia atrás (β) para suavizado.
- Obtiene creencias marginales por tiempo y la verosimilitud de la secuencia.
- Discute estabilidad numérica (escalado/normalización).
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo con HMM de juguete y trazas de α/β.
2. INTERACTIVO: permite ingresar matrices de transición/emisión y observaciones.

Autor: Alejandro Aguirre Díaz
"""
