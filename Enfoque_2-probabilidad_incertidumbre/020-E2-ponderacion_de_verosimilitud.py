"""
020-E2-ponderacion_de_verosimilitud.py
---------------------------------------
Este script implementa Ponderación de Verosimilitud (Likelihood Weighting) para inferencia aproximada:
- Genera muestras fijando la evidencia y ponderándolas por su verosimilitud.
- Reduce el problema del rechazo cuando la evidencia es poco probable.
- Estima distribuciones posteriori a partir de pesos normalizados.
- Compara varianza frente a muestreo por rechazo en distintos escenarios.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo predefinido para comparación de estimadores con y sin ponderación.
2. INTERACTIVO: permite definir evidencia y número de muestras para estimación.

Autor: Alejandro Aguirre Díaz
"""

import random
from typing import Dict, List, Tuple


class NodoPonderado:
	"""Nodo booleano con CPT para muestreo ponderado."""
	
	def __init__(self, nombre: str, padres: List[str]):
		self.nombre = nombre
		self.padres = padres[:]
		# CPT: {tuple(valores_padres_bool): P(X=True|padres)}
		# Para nodos raíz, usamos la clave vacía () → P(X=True)
		self.cpt: Dict[tuple, float] = {}
	
	def establecer_cpt(self, cpt: Dict[tuple, float]):
		"""Establece la CPT del nodo."""
		# Normalizamos claves a tuplas en caso de que vengan como listas
		self.cpt = {tuple(k): v for k, v in cpt.items()}
	
	def muestrear(self, asignacion_padres: Dict[str, bool]) -> bool:
		"""Muestrea un valor para el nodo dados los valores de sus padres."""
		if not self.padres:
			p_true = self.cpt[()]
		else:
			# Construimos la clave en el mismo orden de 'padres'
			clave = tuple(asignacion_padres[p] for p in self.padres)
			p_true = self.cpt[clave]
		
		# Ensayo Bernoulli: True con probabilidad p_true
		return random.random() < p_true
	
	def prob_dado_padres(self, valor: bool, asignacion_padres: Dict[str, bool]) -> float:
		"""Retorna P(X=valor | padres)."""
		if not self.padres:
			p_true = self.cpt[()]
		else:
			# Notar que solo miramos padres; asignacion_padres puede incluir más variables
			clave = tuple(asignacion_padres[p] for p in self.padres)
			p_true = self.cpt[clave]
		
		# Si se pide P(X=False|padres), devolvemos el complemento
		return p_true if valor else (1.0 - p_true)


class RedBayesianaPonderada:
	"""Red bayesiana para muestreo con ponderación de verosimilitud."""
	
	def __init__(self, nodos: List[NodoPonderado]):
		self.nodos = {n.nombre: n for n in nodos}
		# Asumimos que los nodos se entregan en orden topológico (padres antes que hijos)
		self.orden = [n.nombre for n in nodos]
	
	def ponderacion_verosimilitud(self, evidencia: Dict[str, bool], num_muestras: int) -> List[Tuple[Dict[str, bool], float]]:
		"""
		Genera muestras ponderadas:
		- Las variables de evidencia se fijan a sus valores observados.
		- Las variables no observadas se muestrean normalmente.
		- Cada muestra recibe un peso = producto de P(evidencia_i | padres_i).
		
		Retorna lista de (muestra, peso).
		"""
		muestras_ponderadas = []
		
		for _ in range(num_muestras):
			muestra = {}
			peso = 1.0
			
			for nombre in self.orden:
				nodo = self.nodos[nombre]
				
				if nombre in evidencia:
					# Variable de evidencia: fijar al valor observado y actualizar peso
					valor_observado = evidencia[nombre]
					muestra[nombre] = valor_observado
					
					# Peso *= P(valor_observado | padres)
					# Importante: los padres ya han sido asignados por el orden topológico
					prob = nodo.prob_dado_padres(valor_observado, muestra)
					peso *= prob
				else:
					# Variable no observada: muestrear normalmente
					muestra[nombre] = nodo.muestrear(muestra)
			
			muestras_ponderadas.append((muestra, peso))
		
		return muestras_ponderadas
	
	def estimar_probabilidad_ponderada(self, muestras_ponderadas: List[Tuple[Dict[str, bool], float]], 
	                                     consulta_var: str, consulta_val: bool) -> float:
		"""
		Estima P(consulta_var=consulta_val | evidencia) usando muestras ponderadas.
		
		P(Q | E) ≈ Σ (peso * I[Q=q]) / Σ peso
		donde I[Q=q] = 1 si la muestra tiene Q=q, 0 en caso contrario.
		"""
		if not muestras_ponderadas:
			return 0.0
		
		suma_pesos_coincidentes = 0.0
		suma_pesos_totales = 0.0
		
		for muestra, peso in muestras_ponderadas:
			suma_pesos_totales += peso
			if muestra[consulta_var] == consulta_val:
				suma_pesos_coincidentes += peso
		
		if suma_pesos_totales == 0:
			return 0.0
		
		# Normalizamos por la suma total de pesos: estimador insesgado para consultas sobre evidencia fija
		return suma_pesos_coincidentes / suma_pesos_totales


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Ponderación de Verosimilitud")
	print("="*70)
	
	# Construir red: A → B, A → C
	print("\n--- Red: A → B, A → C ---")
	
	A = NodoPonderado('A', [])
	B = NodoPonderado('B', ['A'])
	C = NodoPonderado('C', ['A'])
	
	A.establecer_cpt({(): 0.6})
	B.establecer_cpt({(True,): 0.7, (False,): 0.2})
	C.establecer_cpt({(True,): 0.8, (False,): 0.1})
	
	red = RedBayesianaPonderada([A, B, C])
	
	# Evidencia: C=True
	evidencia = {'C': True}
	num_muestras = 10000
	
	print(f"\n>>> Generando {num_muestras} muestras ponderadas con evidencia {evidencia}")
	muestras = red.ponderacion_verosimilitud(evidencia, num_muestras)
	
	# Estadísticas de pesos: útil para diagnosticar degeneración (pesos muy pequeños)
	pesos = [peso for _, peso in muestras]
	peso_promedio = sum(pesos) / len(pesos)
	print(f"Peso promedio de muestras: {peso_promedio:.4f}")
	
	# Estimaciones
	p_a_dado_c = red.estimar_probabilidad_ponderada(muestras, 'A', True)
	p_b_dado_c = red.estimar_probabilidad_ponderada(muestras, 'B', True)
	
	# Valores teóricos
	# P(A=True|C=True) = P(C=True|A=True) * P(A=True) / P(C=True)
	p_c_dado_a_true = 0.8
	p_a_true = 0.6
	p_c_total = 0.6 * 0.8 + 0.4 * 0.1
	p_a_teorico = (p_c_dado_a_true * p_a_true) / p_c_total
	
	# P(B=True|C=True) requiere marginalizar sobre A
	p_b_c_a_true = 0.7
	p_b_c_a_false = 0.2
	p_a_true_dado_c = p_a_teorico
	p_a_false_dado_c = 1 - p_a_teorico
	p_b_teorico = p_b_c_a_true * p_a_true_dado_c + p_b_c_a_false * p_a_false_dado_c
	
	print(f"\nP(A=True | C=True) ≈ {p_a_dado_c:.4f} (teórico: {p_a_teorico:.4f})")
	print(f"P(B=True | C=True) ≈ {p_b_dado_c:.4f} (teórico: {p_b_teorico:.4f})")
	
	print("\n>>> Ventaja sobre muestreo por rechazo:")
	print("    - Todas las muestras son útiles (ninguna se rechaza)")
	print("    - Eficiente incluso cuando la evidencia es improbable")
	print("    - Los pesos reflejan qué tan consistente es cada muestra con la evidencia")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Ponderación de Verosimilitud")
	print("="*70)
	print("Red: A → B, A → C (parámetros predefinidos)")
	
	# Construir red
	A = NodoPonderado('A', [])
	B = NodoPonderado('B', ['A'])
	C = NodoPonderado('C', ['A'])
	
	A.establecer_cpt({(): 0.6})
	B.establecer_cpt({(True,): 0.7, (False,): 0.2})
	C.establecer_cpt({(True,): 0.8, (False,): 0.1})
	
	red = RedBayesianaPonderada([A, B, C])
	
	try:
		c_obs = input("\nValor observado de C (true/false): ").strip().lower()
		# Interpretamos 't', 'true', 'si', 's' como True
		c_valor = c_obs.startswith('t')
		num = int(input("Número de muestras: ").strip() or "10000")
	except:
		c_valor = True
		num = 10000
		print("Usando C=true y 10000 muestras por defecto")
	
	evidencia = {'C': c_valor}
	print(f"\nGenerando {num} muestras ponderadas con evidencia {evidencia}...")
	
	muestras = red.ponderacion_verosimilitud(evidencia, num)
	
	# Estadísticas: media, mínimo y máximo de pesos para evaluar dispersión
	pesos = [peso for _, peso in muestras]
	peso_promedio = sum(pesos) / len(pesos)
	peso_min = min(pesos)
	peso_max = max(pesos)
	
	print(f"\nEstadísticas de pesos:")
	print(f"  Promedio: {peso_promedio:.4f}")
	print(f"  Mínimo: {peso_min:.4f}")
	print(f"  Máximo: {peso_max:.4f}")
	
	# Estimaciones
	p_a = red.estimar_probabilidad_ponderada(muestras, 'A', True)
	p_b = red.estimar_probabilidad_ponderada(muestras, 'B', True)
	
	print(f"\nEstimaciones con evidencia C={c_valor}:")
	print(f"  P(A=True | evidencia) ≈ {p_a:.4f}")
	print(f"  P(B=True | evidencia) ≈ {p_b:.4f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("PONDERACIÓN DE VEROSIMILITUD")
	print("="*70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		# Opción no reconocida → ejecutamos DEMO por defecto
		modo_demo()
	print("\n" + "="*70)
	print("FIN DEL PROGRAMA")
	print("="*70 + "\n")


if __name__ == "__main__":
	main()

