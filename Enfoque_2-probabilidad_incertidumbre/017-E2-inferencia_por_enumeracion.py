"""
017-E2-inferencia_por_enumeracion.py
--------------------------------
Este script implementa Inferencia por Enumeración en modelos probabilísticos discretos:
- Calcula P(Consulta | Evidencia) enumerando todas las asignaciones de variables ocultas.
- Factoriza distribuciones conjuntas usando la regla de la cadena.
- Mide complejidad temporal/espacial y limita por orden de enumeración.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: consultas predefinidas sobre una red pequeña.
2. INTERACTIVO: permite definir variables, factores y evidencias y ejecutar consultas.

Autor: Alejandro Aguirre Díaz
"""

from itertools import product
from typing import Dict, List, Tuple


class NodoBooleano:
	"""Nodo booleano con CPT para red bayesiana."""
	
	def __init__(self, nombre: str, padres: List[str]):
		self.nombre = nombre
		self.padres = padres[:]
		# CPT: {tuple(valores_padres_bool): P(X=True|padres)}
		# Para nodos raíz la clave es la tupla vacía ()
		# Ejemplo: si padres=['B','E'], la clave es (asig['B'], asig['E'])
		self.cpt: Dict[tuple, float] = {}
	
	def establecer_cpt(self, cpt: Dict[tuple, float]):
		"""Establece la tabla de probabilidad condicional."""
		# Normalizamos las claves a tuplas para evitar claves tipo lista no hashable
		self.cpt = {tuple(k): v for k, v in cpt.items()}
	
	def prob_dado_padres(self, asignacion: Dict[str, bool]) -> float:
		"""Retorna P(X=True | valores de padres en asignacion)."""
		if not self.padres:
			# Nodo raíz: clave vacía
			return self.cpt[()]
		# Construir clave con valores de padres en el orden declarado
		# Importante: el orden en self.padres define el orden de la tupla-clave
		clave = tuple(asignacion[p] for p in self.padres)
		return self.cpt[clave]


class RedBayesianaSimple:
	"""Red bayesiana para inferencia por enumeración."""
	
	def __init__(self, nodos: List[NodoBooleano]):
		self.nodos = {n.nombre: n for n in nodos}
		# Orden topológico simple (asumimos que el usuario provee en orden correcto)
		# En producción convendría validar que no hay ciclos y calcular el orden topológico
		self.orden = [n.nombre for n in nodos]
	
	def prob_conjunta(self, asignacion: Dict[str, bool]) -> float:
		"""
		Calcula P(asignacion) usando la regla de la cadena.
		P(X1,...,Xn) = ∏ P(Xi | padres(Xi))
		"""
		p = 1.0
		# Recorremos en orden topológico para asegurar que los padres ya están en 'asignacion'
		for nombre in self.orden:
			nodo = self.nodos[nombre]
			val = asignacion[nombre]
			# P(nodo=True|padres)
			pt = nodo.prob_dado_padres(asignacion)
			# Multiplicamos por la prob. correspondiente al valor asignado
			# Si la variable es True usamos pt, si es False usamos (1-pt)
			p *= pt if val else (1.0 - pt)
		return p
	
	def enumerar_todas(self, vars_restantes: List[str], asignacion: Dict[str, bool]) -> float:
		"""
		Suma recursiva sobre todas las asignaciones de variables restantes.
		Retorna la suma de P(asignacion completa) sobre todas las extensiones.
		"""
		# Caso base: no quedan variables por asignar
		# La asignación actual ya está completa para las variables consideradas
		if not vars_restantes:
			return self.prob_conjunta(asignacion)
		
		# Caso recursivo: elegir primera variable y sumar sobre sus dos valores booleanos
		var = vars_restantes[0]
		resto = vars_restantes[1:]
		total = 0.0
		
		for valor in [True, False]:
			# Extender la asignación con var=valor (sin mutar el dict original)
			nueva_asig = {**asignacion, var: valor}
			total += self.enumerar_todas(resto, nueva_asig)
		
		return total
	
	def inferencia_por_enumeracion(self, consulta: Tuple[str, bool], evidencia: Dict[str, bool]) -> float:
		"""
		Calcula P(consulta | evidencia) mediante enumeración completa.
		
		P(Q=q | E=e) = P(Q=q, E=e) / P(E=e)
		               = Σ_h P(Q=q, E=e, H=h) / Σ_{q',h} P(Q=q', E=e, H=h)
		donde H son las variables ocultas (no observadas ni consultadas).
		"""
		q_var, q_val = consulta
		
		# Identificar variables ocultas: todas menos evidencia y la variable de consulta
		ocultas = [v for v in self.orden if v not in evidencia and v != q_var]
		
		# Numerador: P(Q=q_val, evidencia) = Σ_ocultas P(Q=q_val, evidencia, ocultas)
		asig_num = {**evidencia, q_var: q_val}
		numerador = self.enumerar_todas(ocultas, asig_num)
		
		# Denominador: P(evidencia) = Σ_{q'∈{True,False}} Σ_h P(Q=q',evidencia,h)
		# Se calcula sumando los dos casos de Q
		asig_den_true = {**evidencia, q_var: True}
		asig_den_false = {**evidencia, q_var: False}
		denominador = (self.enumerar_todas(ocultas, asig_den_true) +
		               self.enumerar_todas(ocultas, asig_den_false))
		
		# Retornar probabilidad condicional (protegido ante denominador 0)
		return (numerador / denominador) if denominador > 0 else 0.0


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Inferencia por Enumeración")
	print("="*70)
	
	# Construir red de alarma: B → A ← E, A → J, A → M
	print("\n--- Red de Alarma ---")
	print("Estructura: B → A ← E, A → J, A → M")
	
	# Definir nodos
	B = NodoBooleano('B', [])
	E = NodoBooleano('E', [])
	A = NodoBooleano('A', ['B', 'E'])
	J = NodoBooleano('J', ['A'])
	M = NodoBooleano('M', ['A'])
	
	# Establecer CPTs (priors y condicionales)
	B.establecer_cpt({(): 0.001})  # P(Burglary)
	E.establecer_cpt({(): 0.002})  # P(Earthquake)
	
	# P(Alarm | Burglary, Earthquake)
	A.establecer_cpt({
		(True, True): 0.95,   # Alta prob. de alarma si hay robo y terremoto
		(True, False): 0.94,  # Alta prob. con robo pero sin terremoto
		(False, True): 0.29,  # Prob. moderada con terremoto sin robo
		(False, False): 0.001 # Casi imposible que suene sin causas
	})
	
	# P(JohnCalls | Alarm), P(MaryCalls | Alarm)
	J.establecer_cpt({(True,): 0.90, (False,): 0.05})   # Juan llama si suena la alarma
	M.establecer_cpt({(True,): 0.70, (False,): 0.01})   # María llama con menor probabilidad
	
	# Crear red
	red = RedBayesianaSimple([B, E, A, J, M])
	
	# Consulta 1: P(B=True | J=True, M=True)
	print("\nConsulta 1: P(Burglary=True | JohnCalls=True, MaryCalls=True)")
	evidencia1 = {'J': True, 'M': True}
	prob1 = red.inferencia_por_enumeracion(('B', True), evidencia1)
	print(f"Resultado: {prob1:.6f}")
	
	# Consulta 2: P(E=True | J=True, M=False)
	print("\nConsulta 2: P(Earthquake=True | JohnCalls=True, MaryCalls=False)")
	evidencia2 = {'J': True, 'M': False}
	prob2 = red.inferencia_por_enumeracion(('E', True), evidencia2)
	print(f"Resultado: {prob2:.6f}")
	
	print("\n>>> Nota sobre complejidad:")
	print("    La enumeración completa tiene complejidad O(2^n) donde n = variables ocultas.")
	print("    Para redes grandes, se requieren métodos más eficientes (eliminación de variables).")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Inferencia en Red de Alarma")
	print("="*70)
	print("Red predefinida: B → A ← E, A → J, A → M")
	
	# Construir la misma red del demo
	B = NodoBooleano('B', [])
	E = NodoBooleano('E', [])
	A = NodoBooleano('A', ['B', 'E'])
	J = NodoBooleano('J', ['A'])
	M = NodoBooleano('M', ['A'])
	
	B.establecer_cpt({(): 0.001})
	E.establecer_cpt({(): 0.002})
	A.establecer_cpt({
		(True, True): 0.95, (True, False): 0.94,
		(False, True): 0.29, (False, False): 0.001
	})
	J.establecer_cpt({(True,): 0.90, (False,): 0.05})
	M.establecer_cpt({(True,): 0.70, (False,): 0.01})
	
	red = RedBayesianaSimple([B, E, A, J, M])
	
	# Solicitar evidencia al usuario
	print("\nEvidencia disponible: J (JohnCalls), M (MaryCalls)")
	try:
		j_obs = input("¿Juan llamó? (true/false): ").strip().lower()
		m_obs = input("¿María llamó? (true/false): ").strip().lower()
		
		# Normalizamos varias formas de escribir booleanos en español/inglés
		evidencia = {}
		if j_obs in ['true', 't', 'yes', 'si', 's']:
			evidencia['J'] = True
		elif j_obs in ['false', 'f', 'no', 'n']:
			evidencia['J'] = False
		
		if m_obs in ['true', 't', 'yes', 'si', 's']:
			evidencia['M'] = True
		elif m_obs in ['false', 'f', 'no', 'n']:
			evidencia['M'] = False
		
		if not evidencia:
			print("No se proporcionó evidencia válida. Usando J=True, M=True por defecto.")
			evidencia = {'J': True, 'M': True}
	except:
		evidencia = {'J': True, 'M': True}
	
	# Calcular posteriores para B y E
	print(f"\nCalculando con evidencia: {evidencia}")
	prob_B = red.inferencia_por_enumeracion(('B', True), evidencia)
	prob_E = red.inferencia_por_enumeracion(('E', True), evidencia)
	
	print(f"\nP(Burglary=True | evidencia) = {prob_B:.6f}")
	print(f"P(Earthquake=True | evidencia) = {prob_E:.6f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("INFERENCIA POR ENUMERACIÓN")
	print("="*70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		# Entrada no reconocida: por defecto ejecutamos la DEMO para ilustrar el método
		modo_demo()
	print("\n" + "="*70)
	print("FIN DEL PROGRAMA")
	print("="*70 + "\n")


if __name__ == "__main__":
	main()
