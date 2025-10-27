"""
023-E2-hipotesis_de_markov_procesos_de_markov.py
-------------------------------------------------
Este script ilustra la hipótesis de Markov y los procesos de Markov discretos:
- Simula cadenas de Markov de tiempo discreto con matriz de transición.
- Estima la distribución estacionaria empírica y verifica convergencia.
- Verifica empíricamente la propiedad de Markov: P(X_{t+1}|X_t)=P(X_{t+1}|X_t,X_{t-1}).

Modos de ejecución:
1. DEMO: cadena simple de 3 estados con simulación y diagnósticos.
2. INTERACTIVO: permite definir pasos e imprimir estadísticas básicas.

Autor: Alejandro Aguirre Díaz
"""

import random
from typing import Dict, List, Tuple


class CadenaDeMarkov:
	"""Cadena de Markov finita con matriz de transición discreta."""

	def __init__(self, estados: List[str], transiciones: Dict[str, Dict[str, float]]):
		self.estados = estados[:]
		self.trans = {s: dict(p) for s, p in transiciones.items()}
		# Normalizar por seguridad
		for s in self.estados:
			total = sum(self.trans[s].values())
			if total > 0:
				for k in self.trans[s]:
					self.trans[s][k] /= total

	def siguiente(self, estado_actual: str) -> str:
		"""Muestra el siguiente estado según P(·|estado_actual)."""
		r = random.random()
		acum = 0.0
		for s2, p in self.trans[estado_actual].items():
			acum += p
			if r <= acum:
				return s2
		# fallback por redondeo
		return self.estados[-1]

	def simular(self, n: int, estado_inicial: str) -> List[str]:
		"""Genera una trayectoria de longitud n empezando en estado_inicial."""
		tray = [estado_inicial]
		for _ in range(1, n):
			tray.append(self.siguiente(tray[-1]))
		return tray

	def estacionaria_empirica(self, trayectoria: List[str]) -> Dict[str, float]:
		cuenta = {s: 0 for s in self.estados}
		for s in trayectoria:
			cuenta[s] += 1
		total = len(trayectoria)
		return {s: cuenta[s] / total for s in self.estados}

	def estimar_transicion_empirica(self, trayectoria: List[str]) -> Dict[Tuple[str, str], float]:
		"""Estima P(X_{t+1}=j | X_t=i) a partir de frecuencias de la trayectoria."""
		conteo_ij = {}
		conteo_i = {s: 0 for s in self.estados}
		for i in range(len(trayectoria) - 1):
			i_state = trayectoria[i]
			j_state = trayectoria[i + 1]
			conteo_i[i_state] += 1
			conteo_ij[(i_state, j_state)] = conteo_ij.get((i_state, j_state), 0) + 1
		probs = {}
		for (i_state, j_state), c in conteo_ij.items():
			if conteo_i[i_state] > 0:
				probs[(i_state, j_state)] = c / conteo_i[i_state]
		return probs


def verificar_propiedad_markov(trayectoria: List[str]) -> float:
	"""
	Verifica empíricamente la propiedad de Markov con un score de discrepancia:
	Compara P(X_{t+1}|X_t) vs P(X_{t+1}|X_t, X_{t-1}).
	Retorna el error absoluto promedio entre ambas estimaciones (más bajo es mejor).
	"""
	# Conteos para P(j|i)
	conteo_i = {}
	conteo_ij = {}
	# Conteos para P(j|i,h)
	conteo_ih = {}
	conteo_ijh = {}

	for t in range(1, len(trayectoria) - 1):
		h, i, j = trayectoria[t - 1], trayectoria[t], trayectoria[t + 1]
		conteo_i[i] = conteo_i.get(i, 0) + 1
		conteo_ij[(i, j)] = conteo_ij.get((i, j), 0) + 1
		conteo_ih[(i, h)] = conteo_ih.get((i, h), 0) + 1
		conteo_ijh[(i, j, h)] = conteo_ijh.get((i, j, h), 0) + 1

	# Calcular discrepancias donde ambos términos están definidos
	errores = []
	for (i, j), cij in conteo_ij.items():
		p_j_i = cij / conteo_i[i]
		# Promedio ponderado de P(j|i,h) sobre h observados
		contribs = []
		for (i2, h), cih in conteo_ih.items():
			if i2 != i:
				continue
			cijh = conteo_ijh.get((i, j, h), 0)
			p_j_ih = cijh / cih if cih > 0 else 0.0
			peso = cih / conteo_i[i]
			contribs.append(peso * p_j_ih)
		if contribs:
			p_j_i_prom = sum(contribs)
			errores.append(abs(p_j_i - p_j_i_prom))
	return sum(errores) / len(errores) if errores else 0.0


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "=" * 70)
	print("MODO DEMO: Hipótesis de Markov")
	print("=" * 70)

	estados = ['A', 'B', 'C']
	trans = {
		'A': {'A': 0.1, 'B': 0.6, 'C': 0.3},
		'B': {'A': 0.2, 'B': 0.3, 'C': 0.5},
		'C': {'A': 0.5, 'B': 0.2, 'C': 0.3},
	}
	cadena = CadenaDeMarkov(estados, trans)

	n = 20000
	estado0 = 'A'
	print(f"\nSimulando {n} pasos desde '{estado0}'...")
	tray = cadena.simular(n, estado0)

	# Distribución estacionaria empírica
	pi_emp = cadena.estacionaria_empirica(tray[5000:])  # descartamos burn-in
	print("\nDistribución estacionaria empírica (post burn-in):")
	for s in estados:
		print(f"  {s}: {pi_emp[s]:.3f}")

	# Estimar matriz de transición empírica y comparar una fila
	p_emp = cadena.estimar_transicion_empirica(tray)
	print("\nEjemplo P_emp(X_{t+1}|X_t='A'):")
	for s2 in estados:
		print(f"  → {s2}: {p_emp.get(('A', s2), 0.0):.3f} (teórico: {trans['A'][s2]:.3f})")

	# Verificación de Markovidad
	err = verificar_propiedad_markov(tray)
	print(f"\nError promedio Markovidad (bajo es mejor): {err:.4f}")
	print("Nota: Con suficiente datos, el error debería ser pequeño por la propiedad de Markov.")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "=" * 70)
	print("MODO INTERACTIVO: Cadena de Markov")
	print("=" * 70)
	print("Usaremos la misma cadena de la DEMO.")

	estados = ['A', 'B', 'C']
	trans = {
		'A': {'A': 0.1, 'B': 0.6, 'C': 0.3},
		'B': {'A': 0.2, 'B': 0.3, 'C': 0.5},
		'C': {'A': 0.5, 'B': 0.2, 'C': 0.3},
	}
	cadena = CadenaDeMarkov(estados, trans)

	try:
		n = int(input("Número de pasos: ").strip() or "1000")
		s0 = input("Estado inicial (A/B/C): ").strip().upper() or 'A'
		if s0 not in estados:
			s0 = 'A'
	except:
		n, s0 = 1000, 'A'
		print("Usando valores por defecto")

	tray = cadena.simular(n, s0)
	pi_emp = cadena.estacionaria_empirica(tray[max(0, n // 10):])
	print("\nDistribución empírica (descartando 10% inicial):")
	for s in estados:
		print(f"  {s}: {pi_emp[s]:.4f}")

	err = verificar_propiedad_markov(tray)
	print(f"\nError promedio Markovidad: {err:.4f}")


# ========== MAIN ==========

def main():
	print("\n" + "=" * 70)
	print("HIPÓTESIS DE MARKOV Y PROCESOS DE MARKOV")
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
023-E2-hipotesis_de_markov_procesos_de_markov.py
--------------------------------
Este script explica la Hipótesis de Markov y los Procesos de Markov:
- Expone la propiedad de memoria limitada (primer orden y orden-k).
- Define procesos estocásticos y su transición entre estados.
- Relaciona cadenas de Markov con grafos y dinámica temporal.
- Discute supuestos y limitaciones de modelado bajo la hipótesis de Markov.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: simulación de cadenas de Markov simples.
2. INTERACTIVO: permite definir estados, transiciones y ejecutar trayectorias.

Autor: Alejandro Aguirre Díaz
"""
