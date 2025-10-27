"""
024-E2-filtrado_prediccion_suavizado_explicacion.py
--------------------------------
Este script presenta tareas clave en inferencia temporal:
- Filtrado (belief en t), Predicción (t+k), Suavizado (t pasado) y Explicación (mejor secuencia).
- Unifica estas tareas en el marco de modelos dinámicos bayesianos.
- Muestra ejemplos con evidencia parcial y ruidosa.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: secuencias predefinidas mostrando actualización de creencias.
2. INTERACTIVO: configuración de modelos y consulta de tareas específicas.

Autor: Alejandro Aguirre Díaz
"""


from typing import Dict, List, Tuple


class HMM:
	"""HMM discreto con estados y observaciones finitas."""

	def __init__(self, estados: List[str], observaciones: List[str],
				 transicion: Dict[str, Dict[str, float]],
				 emision: Dict[str, Dict[str, float]],
				 inicial: Dict[str, float]):
		# S: lista de estados posibles
		self.S = estados[:]
		# O: lista de observaciones posibles
		self.O = observaciones[:]
		# T: matriz de transición T[s][s'] = P(s'|s)
		self.T = {s: dict(p) for s, p in transicion.items()}
		# E: matriz de emisión E[s][o] = P(o|s)
		self.E = {s: dict(p) for s, p in emision.items()}
		# pi: distribución inicial P(X_0)
		self.pi = dict(inicial)

	# --------- FILTRADO (FORWARD) ---------
	def filtrar(self, obs: List[str]) -> List[Dict[str, float]]:
		"""Retorna lista de creencias bel_t = P(X_t|e_1:t) normalizadas."""
		bels = []
		
		# Paso inicial t=1: combina prior con la primera observación
		e0 = obs[0]
		bel = {}
		z = 0.0
		for s in self.S:
			# bel(X_1) = P(X_1) * P(e_1|X_1)
			bel[s] = self.pi[s] * self.E[s][e0]
			z += bel[s]
		# Normalizar para obtener P(X_1|e_1)
		for s in self.S:
			bel[s] = bel[s] / z if z > 0 else 0.0
		bels.append(bel)
		
		# Pasos subsecuentes t>1: predicción + actualización
		for t in range(1, len(obs)):
			et = obs[t]
			prev = bels[-1]  # creencia en t-1
			
			# PREDICCIÓN: P(X_t|e_1:t-1) = sum_{x_{t-1}} P(X_t|x_{t-1}) * P(x_{t-1}|e_1:t-1)
			pred = {}
			for s in self.S:
				pred[s] = sum(prev[s_prev] * self.T[s_prev][s] for s_prev in self.S)
			
			# ACTUALIZACIÓN: multiplicar por verosimilitud de la observación
			bel = {}
			z = 0.0
			for s in self.S:
				bel[s] = pred[s] * self.E[s][et]
				z += bel[s]
			# Normalizar para obtener P(X_t|e_1:t)
			for s in self.S:
				bel[s] = bel[s] / z if z > 0 else 0.0
			bels.append(bel)
		return bels

	# --------- PREDICCIÓN ---------
	def predecir(self, bel_t: Dict[str, float], k: int = 1) -> Dict[str, float]:
		"""P(X_{t+k}|e_1:t) aplicando k veces la transición."""
		# Comienza con la creencia actual en t
		bel = dict(bel_t)
		
		# Aplica la matriz de transición k veces
		for _ in range(k):
			nuevo = {s: 0.0 for s in self.S}
			# P(X_{t+1}|e_1:t) = sum_x P(X_{t+1}|x) * P(x|e_1:t)
			for s_prev in self.S:
				for s in self.S:
					nuevo[s] += bel[s_prev] * self.T[s_prev][s]
			bel = nuevo
		
		# Normalizar por seguridad (debería sumar 1, pero por precisión numérica)
		z = sum(bel.values())
		if z > 0:
			for s in self.S:
				bel[s] /= z
		return bel

	# --------- SUAVIZADO (FORWARD-BACKWARD) ---------
	def suavizar(self, obs: List[str]) -> List[Dict[str, float]]:
		"""Retorna P(X_t|e_1:T) para t=1..T usando forward-backward."""
		Tn = len(obs)
		
		# PASO FORWARD: calcula P(X_t|e_1:t) para todo t
		bel_f = self.filtrar(obs)
		
		# PASO BACKWARD: calcula mensajes beta_t = P(e_{t+1:T}|X_t)
		beta = [{s: 1.0 for s in self.S} for _ in range(Tn)]
		# Inicialización: beta[T] = 1 para todos los estados
		
		# Propaga hacia atrás desde T-1 hasta 1
		for t in range(Tn - 2, -1, -1):
			et1 = obs[t + 1]
			for s in self.S:
				# beta[t][s] = sum_{s'} P(s'|s) * P(e_{t+1}|s') * beta[t+1][s']
				beta[t][s] = sum(self.T[s][s2] * self.E[s2][et1] * beta[t + 1][s2] for s2 in self.S)
			# Normalizar beta[t] para evitar underflow
			z = sum(beta[t].values())
			if z > 0:
				for s in self.S:
					beta[t][s] /= z
		
		# COMBINAR forward y backward: P(X_t|e_1:T) ∝ P(X_t|e_1:t) * P(e_{t+1:T}|X_t)
		suav = []
		for t in range(Tn):
			bel = {}
			z = 0.0
			for s in self.S:
				bel[s] = bel_f[t][s] * beta[t][s]
				z += bel[s]
			# Normalizar para obtener distribución de probabilidad
			for s in self.S:
				bel[s] = bel[s] / z if z > 0 else 0.0
			suav.append(bel)
		return suav

	# --------- EXPLICACIÓN (VITERBI) ---------
	def viterbi(self, obs: List[str]) -> Tuple[List[str], float]:
		"""Secuencia de estados más probable y su probabilidad (no normalizada)."""
		Tn = len(obs)
		delta = []  # delta[t][s] = max prob de llegar a s en t
		psi = []    # psi[t][s] = estado previo óptimo para llegar a s en t

		# Inicialización t=1: combina prior con primera observación
		e0 = obs[0]
		d0 = {s: self.pi[s] * self.E[s][e0] for s in self.S}
		delta.append(d0)
		psi.append({s: None for s in self.S})  # no hay estado previo en t=1

		# Recursión forward t>1: encuentra el mejor camino a cada estado
		for t in range(1, Tn):
			et = obs[t]
			dt = {}
			psit = {}
			for s in self.S:
				# Para cada estado s en t, encuentra el mejor predecesor
				# delta[t][s] = max_{s'} delta[t-1][s'] * P(s|s') * P(e_t|s)
				vals = [(delta[t - 1][s_prev] * self.T[s_prev][s] * self.E[s][et], s_prev) for s_prev in self.S]
				val, arg = max(vals, key=lambda x: x[0])
				dt[s] = val      # máxima probabilidad para llegar a s
				psit[s] = arg    # estado previo óptimo
			delta.append(dt)
			psi.append(psit)

		# Backtracking: reconstruye el camino óptimo desde T hacia 1
		last_state = max(self.S, key=lambda s: delta[-1][s])  # mejor estado final
		prob = delta[-1][last_state]
		path = [last_state]
		# Sigue los punteros psi hacia atrás
		for t in range(Tn - 1, 0, -1):
			last_state = psi[t][last_state]
			path.append(last_state)
		path.reverse()  # invierte para obtener orden cronológico
		return path, prob


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Filtrado/Predicción/Suavizado/Explicación (HMM)")
	print("=" * 70)

	# Define el modelo HMM: lluvia/no-lluvia según observación de paraguas
	estados = ['Lluvia', 'NoLluvia']
	observ = ['Paraguas', 'NoParaguas']
	
	# Matriz de transición: probabilidad de cambio de clima día a día
	T = {
		'Lluvia':   {'Lluvia': 0.7, 'NoLluvia': 0.3},
		'NoLluvia': {'Lluvia': 0.3, 'NoLluvia': 0.7},
	}
	# Matriz de emisión: probabilidad de ver paraguas dado el clima
	E = {
		'Lluvia':   {'Paraguas': 0.9, 'NoParaguas': 0.1},
		'NoLluvia': {'Paraguas': 0.2, 'NoParaguas': 0.8},
	}
	# Distribución inicial: asumimos equiprobable
	pi = {'Lluvia': 0.5, 'NoLluvia': 0.5}

	hmm = HMM(estados, observ, T, E, pi)

	# Secuencia de observaciones para la demo
	obs_seq = ['Paraguas', 'Paraguas', 'NoParaguas', 'Paraguas', 'Paraguas']
	print(f"\nObservaciones: {obs_seq}")

	# FILTRADO: creencias incrementales conforme vemos cada observación
	bels = hmm.filtrar(obs_seq)
	print("\nFiltrado (creencias):")
	for t, bel in enumerate(bels, 1):
		print(f"  t={t}: Lluvia={bel['Lluvia']:.3f}, NoLluvia={bel['NoLluvia']:.3f}")

	# PREDICCIÓN: ¿qué esperamos para el siguiente paso?
	pred_1 = hmm.predecir(bels[-1], k=1)
	print("\nPredicción 1-paso desde t=T:")
	print(f"  Lluvia={pred_1['Lluvia']:.3f}, NoLluvia={pred_1['NoLluvia']:.3f}")

	# SUAVIZADO: creencias retrospectivas usando toda la secuencia
	suav = hmm.suavizar(obs_seq)
	print("\nSuavizado (posteriores con toda la secuencia):")
	for t, bel in enumerate(suav, 1):
		print(f"  t={t}: Lluvia={bel['Lluvia']:.3f}, NoLluvia={bel['NoLluvia']:.3f}")

	# EXPLICACIÓN (Viterbi): secuencia de estados más probable
	path, p = hmm.viterbi(obs_seq)
	print("\nExplicación (Viterbi):")
	print(f"  Secuencia más probable: {path}")
	print(f"  Score (no normalizado): {p:.6f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: HMM lluvia/paraguas")
	print("=" * 70)
	print("Ingrese observaciones separadas por comas: Paraguas o NoParaguas")
	print("Ejemplo: Paraguas,Paraguas,NoParaguas,Paraguas")

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

	hmm = HMM(estados, observ, T, E, pi)

	# Lee la secuencia de observaciones del usuario
	try:
		linea = input("\nObservaciones: ").strip()
		obs_seq = [x.strip() for x in linea.split(',') if x.strip()]
		if not obs_seq:
			# Si no se ingresa nada, usa secuencia por defecto
			obs_seq = ['Paraguas', 'Paraguas', 'NoParaguas', 'Paraguas']
	except:
		obs_seq = ['Paraguas', 'Paraguas', 'NoParaguas', 'Paraguas']

	# Ejecuta las tareas de inferencia
	bels = hmm.filtrar(obs_seq)
	print("\nFiltrado:")
	for t, bel in enumerate(bels, 1):
		print(f"  t={t}: Lluvia={bel['Lluvia']:.3f}, NoLluvia={bel['NoLluvia']:.3f}")

	suav = hmm.suavizar(obs_seq)
	print("\nSuavizado:")
	for t, bel in enumerate(suav, 1):
		print(f"  t={t}: Lluvia={bel['Lluvia']:.3f}, NoLluvia={bel['NoLluvia']:.3f}")

	path, _ = hmm.viterbi(obs_seq)
	print("\nExplicación (Viterbi):")
	print(f"  {path}")


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("FILTRADO, PREDICCIÓN, SUAVIZADO Y EXPLICACIÓN")
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


