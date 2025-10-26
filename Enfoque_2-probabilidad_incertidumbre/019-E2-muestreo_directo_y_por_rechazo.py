"""
019-E2-muestreo_directo_y_por_rechazo.py
--------------------------------
Este script presenta Muestreo Directo y Muestreo por Rechazo para inferencia aproximada:
- Genera muestras desde la distribución conjunta siguiendo orden topológico.
- Implementa rechazo de muestras inconsistentes con la evidencia.
- Analiza eficiencia vs. escasez de evidencia y variables raras.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: comparación de estimaciones con y sin rechazo en una red pequeña.
2. INTERACTIVO: permite definir evidencia y tamaño de muestreo.

Autor: Alejandro Aguirre Díaz
"""

import random
from typing import Dict, List, Tuple


class NodoMuestreo:
	"""Nodo booleano con CPT para muestreo en red bayesiana."""
	
	def __init__(self, nombre: str, padres: List[str]):
		self.nombre = nombre
		self.padres = padres[:]
		self.cpt: Dict[tuple, float] = {}
	
	def establecer_cpt(self, cpt: Dict[tuple, float]):
		"""Establece la CPT del nodo."""
		self.cpt = {tuple(k): v for k, v in cpt.items()}
	
	def muestrear(self, asignacion_padres: Dict[str, bool]) -> bool:
		"""
		Genera un valor muestreado para este nodo dados los valores de sus padres.
		Retorna True o False según P(X=True|padres).
		"""
		if not self.padres:
			# Nodo raíz: muestrear según P(X=True)
			p_true = self.cpt[()]
		else:
			# Obtener P(X=True | valores de padres)
			clave = tuple(asignacion_padres[p] for p in self.padres)
			p_true = self.cpt[clave]
		
		# Muestreo: retornar True con probabilidad p_true
		return random.random() < p_true


class RedBayesianaMuestreo:
	"""Red bayesiana para muestreo directo y por rechazo."""
	
	def __init__(self, nodos: List[NodoMuestreo]):
		self.nodos = {n.nombre: n for n in nodos}
		# Orden topológico (asumimos que se proporciona en orden correcto)
		self.orden = [n.nombre for n in nodos]
	
	def muestreo_directo(self) -> Dict[str, bool]:
		"""
		Genera una muestra completa de la red siguiendo orden topológico.
		Cada variable se muestrea condicionada a los valores ya muestreados de sus padres.
		"""
		muestra = {}
		for nombre in self.orden:
			nodo = self.nodos[nombre]
			# Muestrear el nodo dado los valores actuales de sus padres
			muestra[nombre] = nodo.muestrear(muestra)
		return muestra
	
	def muestreo_por_rechazo(self, evidencia: Dict[str, bool], num_muestras: int) -> List[Dict[str, bool]]:
		"""
		Genera muestras aceptando solo aquellas consistentes con la evidencia.
		Retorna una lista de muestras válidas (puede ser menor que num_muestras).
		"""
		muestras_validas = []
		intentos = 0
		max_intentos = num_muestras * 1000  # Límite para evitar bucles infinitos
		
		while len(muestras_validas) < num_muestras and intentos < max_intentos:
			# Generar una muestra
			muestra = self.muestreo_directo()
			intentos += 1
			
			# Verificar si es consistente con la evidencia
			consistente = all(muestra[var] == val for var, val in evidencia.items())
			
			if consistente:
				muestras_validas.append(muestra)
		
		return muestras_validas
	
	def estimar_probabilidad(self, muestras: List[Dict[str, bool]], 
	                          consulta_var: str, consulta_val: bool) -> float:
		"""
		Estima P(consulta_var=consulta_val) a partir de las muestras.
		"""
		if not muestras:
			return 0.0
		
		count = sum(1 for m in muestras if m[consulta_var] == consulta_val)
		return count / len(muestras)


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Muestreo Directo y por Rechazo")
	print("="*70)
	
	# Construir red simple: A → B, A → C
	print("\n--- Red: A → B, A → C ---")
	
	A = NodoMuestreo('A', [])
	B = NodoMuestreo('B', ['A'])
	C = NodoMuestreo('C', ['A'])
	
	A.establecer_cpt({(): 0.6})  # P(A=True)=0.6
	B.establecer_cpt({
		(True,): 0.7,   # P(B=True|A=True)
		(False,): 0.2   # P(B=True|A=False)
	})
	C.establecer_cpt({
		(True,): 0.8,   # P(C=True|A=True)
		(False,): 0.1   # P(C=True|A=False)
	})
	
	red = RedBayesianaMuestreo([A, B, C])
	
	# Muestreo directo
	print("\n>>> Muestreo Directo (10,000 muestras)")
	num_muestras_directo = 10000
	muestras_directas = [red.muestreo_directo() for _ in range(num_muestras_directo)]
	
	p_a = red.estimar_probabilidad(muestras_directas, 'A', True)
	p_b = red.estimar_probabilidad(muestras_directas, 'B', True)
	p_c = red.estimar_probabilidad(muestras_directas, 'C', True)
	
	print(f"P(A=True) ≈ {p_a:.3f} (teórico: 0.600)")
	print(f"P(B=True) ≈ {p_b:.3f} (teórico: {0.6*0.7 + 0.4*0.2:.3f})")
	print(f"P(C=True) ≈ {p_c:.3f} (teórico: {0.6*0.8 + 0.4*0.1:.3f})")
	
	# Muestreo por rechazo
	print("\n>>> Muestreo por Rechazo con evidencia C=True (1,000 muestras válidas)")
	evidencia = {'C': True}
	muestras_rechazo = red.muestreo_por_rechazo(evidencia, 1000)
	
	print(f"Muestras obtenidas: {len(muestras_rechazo)}")
	
	if muestras_rechazo:
		p_a_dado_c = red.estimar_probabilidad(muestras_rechazo, 'A', True)
		p_b_dado_c = red.estimar_probabilidad(muestras_rechazo, 'B', True)
		
		# Cálculo teórico P(A=True|C=True) usando Bayes
		p_c_dado_a = 0.8
		p_c_dado_not_a = 0.1
		p_a_prior = 0.6
		p_c_total = 0.6 * 0.8 + 0.4 * 0.1
		p_a_teorico = (p_c_dado_a * p_a_prior) / p_c_total
		
		print(f"P(A=True | C=True) ≈ {p_a_dado_c:.3f} (teórico: {p_a_teorico:.3f})")
		print(f"P(B=True | C=True) ≈ {p_b_dado_c:.3f}")
	
	print("\n>>> Nota sobre eficiencia:")
	print("    El rechazo puede ser ineficiente si la evidencia es improbable.")
	print("    Ejemplo: si P(evidencia)=0.01, se necesitan ~100 muestras por cada válida.")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Muestreo por Rechazo")
	print("="*70)
	print("Red: A → B, A → C (parámetros predefinidos)")
	
	# Construir la red
	A = NodoMuestreo('A', [])
	B = NodoMuestreo('B', ['A'])
	C = NodoMuestreo('C', ['A'])
	
	A.establecer_cpt({(): 0.6})
	B.establecer_cpt({(True,): 0.7, (False,): 0.2})
	C.establecer_cpt({(True,): 0.8, (False,): 0.1})
	
	red = RedBayesianaMuestreo([A, B, C])
	
	try:
		c_obs = input("\nValor observado de C (true/false): ").strip().lower()
		c_valor = c_obs.startswith('t')
		num = int(input("Número de muestras válidas deseadas: ").strip() or "1000")
	except:
		c_valor = True
		num = 1000
		print("Usando C=true y 1000 muestras por defecto")
	
	evidencia = {'C': c_valor}
	print(f"\nGenerando muestras con evidencia {evidencia}...")
	
	muestras = red.muestreo_por_rechazo(evidencia, num)
	
	print(f"Muestras obtenidas: {len(muestras)}/{num}")
	
	if muestras:
		p_a = red.estimar_probabilidad(muestras, 'A', True)
		p_b = red.estimar_probabilidad(muestras, 'B', True)
		
		print(f"\nEstimaciones con evidencia C={c_valor}:")
		print(f"  P(A=True | evidencia) ≈ {p_a:.4f}")
		print(f"  P(B=True | evidencia) ≈ {p_b:.4f}")
	else:
		print("No se pudieron generar suficientes muestras válidas.")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("MUESTREO DIRECTO Y POR RECHAZO")
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

