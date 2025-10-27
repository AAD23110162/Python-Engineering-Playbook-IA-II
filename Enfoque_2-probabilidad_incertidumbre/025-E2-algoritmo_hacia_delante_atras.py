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

	# INICIALIZACIÓN t=0: combina distribución inicial con primera observación
	e0 = obs_seq[0]
	a0 = {}
	z = 0.0  # suma para normalización
	for s in estados:
		# α_0(s) = P(X_0=s) * P(e_0|s)
		a0[s] = pi[s] * E[s][e0]
		z += a0[s]
	# Factor de escala: evita underflow numérico
	c0 = z if z > 0 else 1.0
	# Normalizar alfas con el factor de escala
	for s in estados:
		a0[s] = a0[s] / c0
	alfas.append(a0)
	escalas.append(c0)

	# RECURSIÓN FORWARD t>0: propaga hacia adelante
	for t in range(1, len(obs_seq)):
		et = obs_seq[t]
		at = {}
		z = 0.0
		for s in estados:
			# α_t(s) = [Σ_{s'} α_{t-1}(s') * P(s|s')] * P(e_t|s)
			# Suma sobre todos los estados previos ponderados por transición
			at[s] = sum(alfas[t - 1][sp] * T[sp][s] for sp in estados) * E[s][et]
			z += at[s]
		# Factor de escala para este paso
		ct = z if z > 0 else 1.0
		# Normalizar para mantener estabilidad numérica
		for s in estados:
			at[s] = at[s] / ct
		alfas.append(at)
		escalas.append(ct)

	# VEROSIMILITUD: producto de todos los factores de escala
	# log P(e_1:T) = -Σ log(c_t)
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
	
	# INICIALIZACIÓN: β_T(s) = 1 para todos los estados
	# No hay evidencia futura desde el último paso
	betas: List[Dict[str, float]] = [{s: 1.0 for s in estados} for _ in range(Tn)]
	
	# RECURSIÓN BACKWARD: propaga hacia atrás desde T-1 hasta 0
	for t in range(Tn - 2, -1, -1):
		et1 = obs_seq[t + 1]  # observación en el siguiente paso
		bt = {}
		for s in estados:
			# β_t(s) = Σ_{s'} P(s'|s) * P(e_{t+1}|s') * β_{t+1}(s')
			# Suma sobre todos los estados futuros ponderados por transición y emisión
			bt[s] = sum(T[s][s2] * E[s2][et1] * betas[t + 1][s2] for s2 in estados)
		
		# Normalizar con la misma escala c_{t+1} usada en forward
		# Esto asegura que alfas y betas sean compatibles al combinarlos
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
	# PASO 1: Ejecuta forward para obtener alfas y escalas
	alfas, escalas, log_like = forward(estados, observaciones, T, E, pi, obs_seq)
	
	# PASO 2: Ejecuta backward usando las mismas escalas
	betas = backward(estados, observaciones, T, E, obs_seq, escalas)
	
	# PASO 3: Combina alfas y betas para obtener suavizado
	# P(X_t|e_1:T) ∝ α_t(s) * β_t(s)
	suav = []
	for t in range(len(obs_seq)):
		bel = {}
		z = 0.0  # suma para normalización
		for s in estados:
			# Producto de mensajes forward y backward
			bel[s] = alfas[t][s] * betas[t][s]
			z += bel[s]
		# Normalizar para obtener distribución de probabilidad
		for s in estados:
			bel[s] = bel[s] / z if z > 0 else 0.0
		suav.append(bel)
	return suav, log_like


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Algoritmo Hacia Delante/Atrás (HMM)")
	print("=" * 70)

	# Define el modelo HMM lluvia/paraguas
	estados = ['Lluvia', 'NoLluvia']
	observ = ['Paraguas', 'NoParaguas']
	
	# Matriz de transición: probabilidades de cambio de clima
	T = {
		'Lluvia':   {'Lluvia': 0.7, 'NoLluvia': 0.3},
		'NoLluvia': {'Lluvia': 0.3, 'NoLluvia': 0.7},
	}
	# Matriz de emisión: probabilidad de observar paraguas dado el clima
	E = {
		'Lluvia':   {'Paraguas': 0.9, 'NoParaguas': 0.1},
		'NoLluvia': {'Paraguas': 0.2, 'NoParaguas': 0.8},
	}
	# Distribución inicial: equiprobable
	pi = {'Lluvia': 0.5, 'NoLluvia': 0.5}

	obs_seq = ['Paraguas', 'Paraguas', 'NoParaguas', 'Paraguas']
	print(f"\nObservaciones: {obs_seq}")

	# FORWARD: calcula mensajes alfa (probabilidad hacia adelante)
	alfas, escalas, ll = forward(estados, observ, T, E, pi, obs_seq)
	print("\nForward (alfas normalizados):")
	for t, a in enumerate(alfas, 1):
		print(f"  t={t}: Lluvia={a['Lluvia']:.3f}, NoLluvia={a['NoLluvia']:.3f}")
	print(f"Log-verosimilitud: {ll:.6f}")

	# BACKWARD: calcula mensajes beta (probabilidad hacia atrás)
	betas = backward(estados, observ, T, E, obs_seq, escalas)
	print("\nBackward (betas normalizados):")
	for t, b in enumerate(betas, 1):
		print(f"  t={t}: Lluvia={b['Lluvia']:.3f}, NoLluvia={b['NoLluvia']:.3f}")

	# SUAVIZADO: combina alfas y betas para obtener P(X_t|e_1:T)
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

	# Define el mismo modelo que en la demo
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

	# Lee las observaciones del usuario
	try:
		linea = input("\nObservaciones: ").strip()
		obs_seq = [x.strip() for x in linea.split(',') if x.strip()]
		if not obs_seq:
			# Usa secuencia por defecto si no hay entrada
			obs_seq = ['Paraguas', 'NoParaguas', 'Paraguas']
	except:
		obs_seq = ['Paraguas', 'NoParaguas', 'Paraguas']

	# Ejecuta forward-backward
	alfas, escalas, ll = forward(estados, observ, T, E, pi, obs_seq)
	betas = backward(estados, observ, T, E, obs_seq, escalas)
	suav, _ = forward_backward(estados, observ, T, E, pi, obs_seq)

	# Muestra resultados detallados
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


