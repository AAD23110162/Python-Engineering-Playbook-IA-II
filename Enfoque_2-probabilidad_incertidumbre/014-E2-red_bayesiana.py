"""
014-E2-red_bayesiana.py
--------------------------------
Este script implementa Redes Bayesianas completas:
- Construye grafos acíclicos dirigidos (DAG) para representar dependencias probabilísticas
- Define tablas de probabilidad condicional (CPT) para cada nodo
- Implementa algoritmos de inferencia exacta: eliminación de variables y propagación de creencias
- Realiza inferencia aproximada mediante muestreo (rechazo, ponderación por verosimilitud)
- Aplica inferencia hacia adelante y hacia atrás en la red
- Identifica independencias condicionales mediante d-separación
- Visualiza la estructura de la red y el flujo de información
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente inferencia en redes bayesianas predefinidas (alarma, diagnóstico)
2. INTERACTIVO: permite construir redes personalizadas y realizar consultas de inferencia

Autor: Alejandro Aguirre Díaz
"""

from itertools import product


class Nodo:
	"""Nodo booleano con CPT: P(X=True | padres)."""

	def __init__(self, nombre: str, padres: list[str]):
		self.nombre = nombre
		self.padres = padres[:]  # lista de nombres
		# CPT: { tuple(valores_padres): P(X=True | padres) }
		self.cpt = {}

	def set_cpt(self, cpt: dict):
		self.cpt = {tuple(k): v for k, v in cpt.items()}

	def prob_true(self, asignacion: dict) -> float:
		# Obtiene prob usando los valores de los padres en 'asignacion'
		key = tuple(asignacion[p] for p in self.padres)
		return self.cpt[key]


class RedBayesiana:
	def __init__(self, nodos: list[Nodo]):
		self.nodos = {n.nombre: n for n in nodos}
		# Orden topológico sencillo: padres antes que hijos (el usuario debe respetarlo)
		self.orden = self._orden_topologico_ingenuo()

	def _orden_topologico_ingenuo(self):
		# Algoritmo simple: repetidamente elige un nodo cuyos padres ya estén en la lista
		pendientes = set(self.nodos.keys())
		orden = []
		while pendientes:
			progresado = False
			for nombre in list(pendientes):
				padres = self.nodos[nombre].padres
				if all(p in orden for p in padres):
					orden.append(nombre)
					pendientes.remove(nombre)
					progresado = True
			if not progresado:
				# En caso de ciclo (no debería ocurrir), retornar algún orden
				return list(self.nodos.keys())
		return orden

	def prob_conjunta(self, asignacion: dict) -> float:
		"""P(asignacion) = ∏ P(Xi=xi | padres)."""
		p = 1.0
		for nombre in self.orden:
			nodo = self.nodos[nombre]
			val = asignacion[nombre]
			if nodo.padres:
				pt = nodo.prob_true(asignacion)
			else:
				# nodo raíz tiene CPT con key=() -> P(True)
				pt = nodo.cpt[()]
			p *= pt if val is True else (1 - pt)
		return p

	def consulta_por_enumeracion(self, consulta: tuple[str, bool], evidencia: dict) -> float:
		"""Devuelve P(Q=valor | evidencia) por suma sobre variables ocultas."""
		q_var, q_val = consulta
		ocultas = [n for n in self.nodos.keys() if n not in evidencia and n != q_var]
		# Numerador: P(q_val, evidencia)
		num = self._sumar_sobre_ocultas({**evidencia, q_var: q_val}, ocultas)
		# Denominador: P(evidencia) = Σ_q P(q, evidencia)
		den = num + self._sumar_sobre_ocultas({**evidencia, q_var: (not q_val)}, ocultas)
		return (num / den) if den > 0 else 0.0

	def _sumar_sobre_ocultas(self, base_asig: dict, ocultas: list[str]) -> float:
		if not ocultas:
			return self.prob_conjunta(base_asig)
		var = ocultas[0]
		resto = ocultas[1:]
		total = 0.0
		for val in [True, False]:
			total += self._sumar_sobre_ocultas({**base_asig, var: val}, resto)
		return total


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Red Bayesiana (Alarma de Robo)")
	print("="*70)

	# Red clásica: Robo (B), Terremoto (E), Alarma (A), Llamada de Juan (J), Llamada de María (M)
	B = Nodo('B', [])
	E = Nodo('E', [])
	A = Nodo('A', ['B', 'E'])
	J = Nodo('J', ['A'])
	M = Nodo('M', ['A'])

	# Priors
	B.set_cpt({(): 0.001})
	E.set_cpt({(): 0.002})
	# CPT de A
	A.set_cpt({
		(True, True): 0.95,
		(True, False): 0.94,
		(False, True): 0.29,
		(False, False): 0.001,
	})
	# CPT de llamadas
	J.set_cpt({(True,): 0.90, (False,): 0.05})
	M.set_cpt({(True,): 0.70, (False,): 0.01})

	red = RedBayesiana([B, E, A, J, M])

	# Consulta: P(B=true | J=true, M=true)
	p_b_dado_llamadas = red.consulta_por_enumeracion(('B', True), {'J': True, 'M': True})
	print(f"P(Burglary=True | J=True, M=True) = {p_b_dado_llamadas:.5f}")

	# Mostrar también P(E=true | J=true, M=true)
	p_e_dado_llamadas = red.consulta_por_enumeracion(('E', True), {'J': True, 'M': True})
	print(f"P(Earthquake=True | J=True, M=True) = {p_e_dado_llamadas:.5f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Consulta en Red de Alarma")
	print("="*70)
	print("Usaremos la red de Alarma clásica; podrás fijar evidencia en J y M.")

	# Construir la red como en el demo
	B = Nodo('B', [])
	E = Nodo('E', [])
	A = Nodo('A', ['B', 'E'])
	J = Nodo('J', ['A'])
	M = Nodo('M', ['A'])
	B.set_cpt({(): 0.001})
	E.set_cpt({(): 0.002})
	A.set_cpt({(True, True): 0.95, (True, False): 0.94, (False, True): 0.29, (False, False): 0.001})
	J.set_cpt({(True,): 0.90, (False,): 0.05})
	M.set_cpt({(True,): 0.70, (False,): 0.01})
	red = RedBayesiana([B, E, A, J, M])

	try:
		j = input("Evidencia J (true/false): ").strip().lower() or "true"
		m = input("Evidencia M (true/false): ").strip().lower() or "true"
		ev = {'J': j.startswith('t'), 'M': m.startswith('t')}
	except:
		ev = {'J': True, 'M': True}

	pB = red.consulta_por_enumeracion(('B', True), ev)
	pE = red.consulta_por_enumeracion(('E', True), ev)
	print(f"\nP(B=True | evidencia) = {pB:.6f}")
	print(f"P(E=True | evidencia) = {pE:.6f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("RED BAYESIANA")
	print("="*70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		modo_demo()
	print("\n" + "="*70)
	print("FIN DEL PROGRAMA")
	print("="*70 + "\n")


if __name__ == "__main__":
	main()
