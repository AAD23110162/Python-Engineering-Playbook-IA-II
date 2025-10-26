"""
021-E2-monte_carlo_para_cadenas_de_markov.py
--------------------------------
Este script presenta métodos de Monte Carlo para Cadenas de Markov (MCMC):
- Introduce cadenas de Markov ergódicas y su distribución estacionaria.
- Implementa conceptualmente Metropolis-Hastings y Gibbs Sampling.
- Mide convergencia y mezcla mediante diagnósticos cualitativos.
- Aplica MCMC para aproximar distribuciones posteriori complejas.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: muestra trayectorias de cadenas y estimaciones de momentos.
2. INTERACTIVO: permite ajustar propuestas y número de iteraciones.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import Dict, List, Callable


class MetropolisHastings:
	"""
	Implementación de Metropolis-Hastings para muestreo de distribuciones complejas.
	
	Dado una función de densidad (no normalizada) π(x), genera muestras
	que convergen a la distribución π.
	"""
	
	def __init__(self, densidad_no_normalizada: Callable[[float], float], 
	             propuesta_std: float = 1.0):
		"""
		densidad_no_normalizada: función π(x) proporcional a la densidad objetivo
		propuesta_std: desviación estándar de la distribución de propuesta (Gaussiana)
		"""
		self.densidad = densidad_no_normalizada
		self.propuesta_std = propuesta_std
	
	def muestrear(self, num_iteraciones: int, x_inicial: float = 0.0) -> List[float]:
		"""
		Genera una cadena de Markov de longitud num_iteraciones.
		
		Algoritmo:
		1. Partir de x_inicial
		2. En cada paso:
		   - Proponer x' ~ N(x_actual, propuesta_std²)
		   - Calcular ratio α = π(x') / π(x_actual)
		   - Aceptar x' con probabilidad min(1, α)
		   - Si se acepta, x_siguiente = x'; sino x_siguiente = x_actual
		"""
		cadena = []
		x_actual = x_inicial
		aceptaciones = 0
		
		for i in range(num_iteraciones):
			# Registrar estado actual
			cadena.append(x_actual)
			
			# Proponer nuevo estado: x' = x_actual + N(0, propuesta_std)
			x_propuesto = x_actual + random.gauss(0, self.propuesta_std)
			
			# Calcular densidades
			densidad_actual = self.densidad(x_actual)
			densidad_propuesta = self.densidad(x_propuesto)
			
			# Calcular ratio de aceptación
			if densidad_actual > 0:
				alpha = min(1.0, densidad_propuesta / densidad_actual)
			else:
				alpha = 1.0  # Aceptar si estamos en un punto de densidad 0
			
			# Aceptar o rechazar
			if random.random() < alpha:
				x_actual = x_propuesto
				aceptaciones += 1
		
		tasa_aceptacion = aceptaciones / num_iteraciones
		return cadena, tasa_aceptacion


class GibbsSampling:
	"""
	Muestreador de Gibbs para distribuciones conjuntas multivariadas.
	
	En cada iteración, actualiza una variable a la vez condicionada al resto.
	"""
	
	def __init__(self, variables: List[str], 
	             condicionales: Dict[str, Callable[[Dict[str, float]], float]]):
		"""
		variables: lista de nombres de variables
		condicionales: {var: función_muestra_condicional}
		  donde función_muestra_condicional(estado_actual) → nuevo valor para var
		"""
		self.variables = variables
		self.condicionales = condicionales
	
	def muestrear(self, num_iteraciones: int, estado_inicial: Dict[str, float]) -> List[Dict[str, float]]:
		"""
		Genera muestras usando Gibbs Sampling.
		
		En cada iteración:
		- Para cada variable, muestrear de P(var | todas las demás)
		- Actualizar el estado con el nuevo valor
		"""
		cadena = []
		estado = dict(estado_inicial)
		
		for i in range(num_iteraciones):
			# Registrar estado actual
			cadena.append(dict(estado))
			
			# Actualizar cada variable en secuencia
			for var in self.variables:
				# Muestrear var condicionado al estado actual de las demás
				nuevo_valor = self.condicionales[var](estado)
				estado[var] = nuevo_valor
		
		return cadena


# ========== FUNCIONES DE EJEMPLO ==========

def densidad_bimodal(x: float) -> float:
	"""
	Distribución bimodal: mezcla de dos gaussianas.
	π(x) ∝ 0.3·N(-2,1) + 0.7·N(3,1.5)
	"""
	# Componente 1: N(-2, 1)
	c1 = 0.3 * math.exp(-0.5 * ((x + 2) ** 2))
	
	# Componente 2: N(3, 1.5²)
	c2 = 0.7 * math.exp(-0.5 * ((x - 3) / 1.5) ** 2)
	
	return c1 + c2


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Monte Carlo para Cadenas de Markov (MCMC)")
	print("="*70)
	
	# Ejemplo 1: Metropolis-Hastings para distribución bimodal
	print("\n--- Metropolis-Hastings: Distribución Bimodal ---")
	print("Objetivo: mezcla 0.3·N(-2,1) + 0.7·N(3,1.5)")
	
	mh = MetropolisHastings(densidad_bimodal, propuesta_std=2.0)
	
	# Generar cadena
	num_iter = 10000
	burn_in = 1000  # Descartar primeras iteraciones (burn-in)
	
	print(f"\nGenerando {num_iter} muestras (burn-in={burn_in})...")
	cadena, tasa = mh.muestrear(num_iter, x_inicial=0.0)
	
	print(f"Tasa de aceptación: {tasa:.2%}")
	
	# Descartar burn-in
	muestras = cadena[burn_in:]
	
	# Estadísticas
	media = sum(muestras) / len(muestras)
	varianza = sum((x - media) ** 2 for x in muestras) / len(muestras)
	
	print(f"\nEstadísticas de la cadena (post burn-in):")
	print(f"  Media: {media:.3f}")
	print(f"  Desviación estándar: {math.sqrt(varianza):.3f}")
	
	# Mostrar algunas muestras
	print(f"\nPrimeras 10 muestras post burn-in: {[f'{x:.2f}' for x in muestras[:10]]}")
	
	# Ejemplo 2: Gibbs Sampling (ejemplo simplificado 2D)
	print("\n--- Gibbs Sampling: Distribución 2D ---")
	print("Ejemplo conceptual: P(X,Y) con condicionales conocidas")
	
	# Condicionales de ejemplo (distribuciones correlacionadas)
	# P(X|Y) ~ N(0.5*Y, 1), P(Y|X) ~ N(0.5*X, 1)
	def muestra_x_dado_y(estado):
		y = estado['Y']
		return random.gauss(0.5 * y, 1.0)
	
	def muestra_y_dado_x(estado):
		x = estado['X']
		return random.gauss(0.5 * x, 1.0)
	
	gibbs = GibbsSampling(
		variables=['X', 'Y'],
		condicionales={
			'X': muestra_x_dado_y,
			'Y': muestra_y_dado_x
		}
	)
	
	print("\nGenerando 5,000 muestras con Gibbs...")
	cadena_gibbs = gibbs.muestrear(5000, estado_inicial={'X': 0.0, 'Y': 0.0})
	
	# Descartar burn-in
	muestras_gibbs = cadena_gibbs[500:]
	
	# Estadísticas
	media_x = sum(s['X'] for s in muestras_gibbs) / len(muestras_gibbs)
	media_y = sum(s['Y'] for s in muestras_gibbs) / len(muestras_gibbs)
	
	print(f"\nEstadísticas (post burn-in):")
	print(f"  Media de X: {media_x:.3f} (teórico ≈ 0)")
	print(f"  Media de Y: {media_y:.3f} (teórico ≈ 0)")
	
	print("\n>>> Nota sobre convergencia:")
	print("    - Burn-in: descartar iteraciones iniciales antes de que la cadena converja")
	print("    - Tasa de aceptación óptima en MH: típicamente 20-50%")
	print("    - Gibbs: siempre acepta, pero requiere condicionales tratables")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Metropolis-Hastings")
	print("="*70)
	print("Muestreo de distribución bimodal: 0.3·N(-2,1) + 0.7·N(3,1.5)")
	
	try:
		num_iter = int(input("\nNúmero de iteraciones: ").strip() or "10000")
		prop_std = float(input("Desviación estándar de propuesta: ").strip() or "2.0")
		burn_in = int(input("Burn-in (iteraciones a descartar): ").strip() or "1000")
	except:
		num_iter, prop_std, burn_in = 10000, 2.0, 1000
		print("Usando valores por defecto: 10000 iter, std=2.0, burn-in=1000")
	
	mh = MetropolisHastings(densidad_bimodal, propuesta_std=prop_std)
	
	print(f"\nGenerando cadena...")
	cadena, tasa = mh.muestrear(num_iter, x_inicial=0.0)
	
	print(f"\nTasa de aceptación: {tasa:.2%}")
	
	if tasa < 0.15:
		print("  ⚠ Tasa muy baja: considere reducir propuesta_std")
	elif tasa > 0.60:
		print("  ⚠ Tasa muy alta: considere aumentar propuesta_std")
	else:
		print("  ✓ Tasa aceptable")
	
	# Análisis post burn-in
	muestras = cadena[burn_in:]
	media = sum(muestras) / len(muestras)
	varianza = sum((x - media) ** 2 for x in muestras) / len(muestras)
	
	print(f"\nEstadísticas post burn-in ({len(muestras)} muestras):")
	print(f"  Media: {media:.3f}")
	print(f"  Desviación estándar: {math.sqrt(varianza):.3f}")
	print(f"  Mínimo: {min(muestras):.3f}")
	print(f"  Máximo: {max(muestras):.3f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("MONTE CARLO PARA CADENAS DE MARKOV (MCMC)")
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

