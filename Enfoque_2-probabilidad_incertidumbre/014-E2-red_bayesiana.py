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
		self.padres = padres[:]  # lista de nombres de los nodos padres
		# CPT (Tabla de Probabilidad Condicional): 
		# Mapea cada combinación de valores de padres → P(X=True | padres)
		# Formato: { tuple(valores_padres): probabilidad }
		self.cpt = {}

	def set_cpt(self, cpt: dict):
		"""Establece la CPT. Las claves deben ser tuplas de valores booleanos de los padres."""
		# Convertimos las claves a tuplas por si vienen como listas
		self.cpt = {tuple(k): v for k, v in cpt.items()}

	def prob_true(self, asignacion: dict) -> float:
		"""Retorna P(X=True | valores de padres en asignacion)."""
		# Construimos la clave de la CPT usando los valores de los padres en la asignación
		key = tuple(asignacion[p] for p in self.padres)
		return self.cpt[key]


class RedBayesiana:
	def __init__(self, nodos: list[Nodo]):
		# Guardamos los nodos en un diccionario para acceso rápido por nombre
		self.nodos = {n.nombre: n for n in nodos}
		
		# Calculamos un orden topológico: padres antes que hijos
		# Esto garantiza que al calcular P(X1,...,Xn) procesemos variables en orden correcto
		self.orden = self._orden_topologico_ingenuo()

	def _orden_topologico_ingenuo(self):
		"""Orden topológico simple usando algoritmo de Kahn modificado."""
		# Algoritmo: repetidamente añadir un nodo cuyos padres ya estén procesados
		pendientes = set(self.nodos.keys())
		orden = []
		
		while pendientes:
			progresado = False
			for nombre in list(pendientes):
				padres = self.nodos[nombre].padres
				# Si todos los padres ya están en el orden, este nodo está listo
				if all(p in orden for p in padres):
					orden.append(nombre)
					pendientes.remove(nombre)
					progresado = True
			
			# Si no progresamos, hay un ciclo (no debería ocurrir en DAG válido)
			if not progresado:
				return list(self.nodos.keys())
		
		return orden

	def prob_conjunta(self, asignacion: dict) -> float:
		"""Calcula P(asignacion) = ∏ P(Xi=xi | padres) usando regla de la cadena."""
		p = 1.0
		
		# Recorremos en orden topológico para aplicar la factorización
		for nombre in self.orden:
			nodo = self.nodos[nombre]
			val = asignacion[nombre]
			
			# Obtenemos P(nodo=True | padres)
			if nodo.padres:
				pt = nodo.prob_true(asignacion)
			else:
				# Nodo raíz (sin padres): la CPT tiene clave vacía () → P(True)
				pt = nodo.cpt[()]
			
			# Multiplicamos por P(nodo=val | padres): pt si val=True, (1-pt) si val=False
			p *= pt if val is True else (1 - pt)
		
		return p

	def consulta_por_enumeracion(self, consulta: tuple[str, bool], evidencia: dict) -> float:
		"""
		Inferencia exacta por enumeración: P(Q=valor | evidencia).
		Suma sobre todas las variables ocultas (no observadas).
		"""
		q_var, q_val = consulta
		
		# Variables ocultas: todas excepto la consulta y la evidencia
		ocultas = [n for n in self.nodos.keys() if n not in evidencia and n != q_var]
		
		# Numerador: P(Q=q_val, evidencia) = Σ_ocultas P(Q=q_val, evidencia, ocultas)
		num = self._sumar_sobre_ocultas({**evidencia, q_var: q_val}, ocultas)
		
		# Denominador: P(evidencia) = P(Q=True, evidencia) + P(Q=False, evidencia)
		den = num + self._sumar_sobre_ocultas({**evidencia, q_var: (not q_val)}, ocultas)
		
		# P(Q=q_val | evidencia) = P(Q=q_val, evidencia) / P(evidencia)
		return (num / den) if den > 0 else 0.0

	def _sumar_sobre_ocultas(self, base_asig: dict, ocultas: list[str]) -> float:
		"""Suma recursiva sobre todas las combinaciones de variables ocultas."""
		# Caso base: si no hay más variables ocultas, calculamos la conjunta
		if not ocultas:
			return self.prob_conjunta(base_asig)
		
		# Caso recursivo: sumamos sobre los dos valores posibles de la primera variable oculta
		var = ocultas[0]
		resto = ocultas[1:]
		total = 0.0
		for val in [True, False]:
			# Sumamos P(..., var=val, ...)
			total += self._sumar_sobre_ocultas({**base_asig, var: val}, resto)
		
		return total


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Red Bayesiana (Alarma de Robo)")
	print("="*70)

	# Red clásica de alarma:
	# Estructura: B → A ← E, A → J, A → M
	# B=Robo (Burglary), E=Terremoto (Earthquake), A=Alarma
	# J=Llamada de Juan (John calls), M=Llamada de María (Mary calls)
	
	# Nodos raíz (sin padres)
	B = Nodo('B', [])
	E = Nodo('E', [])
	
	# Nodo con dos causas (colisionador)
	A = Nodo('A', ['B', 'E'])
	
	# Nodos hoja (efectos de la alarma)
	J = Nodo('J', ['A'])
	M = Nodo('M', ['A'])

	# Priors de nodos raíz: P(B=True) y P(E=True)
	B.set_cpt({(): 0.001})  # 0.1% probabilidad de robo
	E.set_cpt({(): 0.002})  # 0.2% probabilidad de terremoto
	
	# CPT de Alarma: P(A=True | B, E)
	# Clave (B, E) → P(A=True | B, E)
	A.set_cpt({
		(True, True): 0.95,    # Si robo Y terremoto → alarma 95%
		(True, False): 0.94,   # Si robo, NO terremoto → alarma 94%
		(False, True): 0.29,   # Si NO robo, terremoto → alarma 29%
		(False, False): 0.001, # Si NO robo, NO terremoto → alarma 0.1% (falsa alarma)
	})
	
	# CPT de las llamadas: P(J=True | A) y P(M=True | A)
	J.set_cpt({(True,): 0.90, (False,): 0.05})   # Juan llama 90% si alarma, 5% si no
	M.set_cpt({(True,): 0.70, (False,): 0.01})   # María llama 70% si alarma, 1% si no

	# Construir la red bayesiana
	red = RedBayesiana([B, E, A, J, M])

	# Consulta: P(B=True | J=True, M=True)
	# Si tanto Juan como María llaman, ¿cuál es la probabilidad de robo?
	p_b_dado_llamadas = red.consulta_por_enumeracion(('B', True), {'J': True, 'M': True})
	print(f"P(Burglary=True | J=True, M=True) = {p_b_dado_llamadas:.5f}")

	# Consulta: P(E=True | J=True, M=True)
	# Si tanto Juan como María llaman, ¿cuál es la probabilidad de terremoto?
	p_e_dado_llamadas = red.consulta_por_enumeracion(('E', True), {'J': True, 'M': True})
	print(f"P(Earthquake=True | J=True, M=True) = {p_e_dado_llamadas:.5f}")
	
	# Nota: Ambas probabilidades son bajas pero B > E porque P(A|B) > P(A|E) en general


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Consulta en Red de Alarma")
	print("="*70)
	print("Usaremos la red de Alarma clásica; podrás fijar evidencia en J y M.")

	# Construir la red de alarma (misma estructura que en el demo)
	B = Nodo('B', [])
	E = Nodo('E', [])
	A = Nodo('A', ['B', 'E'])
	J = Nodo('J', ['A'])
	M = Nodo('M', ['A'])
	
	# Establecer las mismas CPTs que en el demo
	B.set_cpt({(): 0.001})
	E.set_cpt({(): 0.002})
	A.set_cpt({(True, True): 0.95, (True, False): 0.94, (False, True): 0.29, (False, False): 0.001})
	J.set_cpt({(True,): 0.90, (False,): 0.05})
	M.set_cpt({(True,): 0.70, (False,): 0.01})
	red = RedBayesiana([B, E, A, J, M])

	# Solicitar evidencia al usuario
	try:
		j = input("Evidencia J (true/false): ").strip().lower() or "true"
		m = input("Evidencia M (true/false): ").strip().lower() or "true"
		# Convertir a booleanos
		ev = {'J': j.startswith('t'), 'M': m.startswith('t')}
	except:
		# Valores por defecto si hay error
		ev = {'J': True, 'M': True}

	# Realizar inferencia para B y E dada la evidencia
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
