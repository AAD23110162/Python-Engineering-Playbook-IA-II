"""
018-E2-eliminacion_de_variables.py
--------------------------------
Este script implementa Eliminación de Variables para inferencia exacta en redes bayesianas:
- Usa factores para representar distribuciones parciales.
- Aplica suma marginal y multiplicación de factores en un orden de eliminación.
- Compara distintas heurísticas de orden (min-degree, min-fill) a nivel conceptual.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplo predefinido con trazas de factores por paso.
2. INTERACTIVO: permite cargar factores, escoger orden y ejecutar la eliminación.

Autor: Alejandro Aguirre Díaz
"""

from itertools import product
from typing import Dict, List, Set, Tuple


class Factor:
	"""
	Representa un factor probabilístico sobre un conjunto de variables.
	Factor(X1, X2, ...) mapea cada asignación (x1, x2, ...) a un valor numérico.
	"""
	
	def __init__(self, variables: List[str], tabla: Dict[tuple, float]):
		"""
		variables: lista de nombres de variables en orden
		tabla: {tuple(valores): probabilidad/peso}
		- El orden en 'variables' define el orden en las tuplas-clave de 'tabla'.
		- Los valores en la tabla son típicamente probabilidades (0..1),
		  pero durante la eliminación pueden ser productos no normalizados.
		"""
		self.variables = variables[:]
		# Copiamos para evitar efectos secundarios si se reutiliza el dict original
		self.tabla = dict(tabla)
	
	def __repr__(self):
		return f"Factor({self.variables}, {len(self.tabla)} entradas)"
	
	def multiplicar(self, otro: 'Factor') -> 'Factor':
		"""
		Multiplica dos factores: f1(X,Y) * f2(Y,Z) = f3(X,Y,Z).
		El producto combina las variables y multiplica los valores correspondientes.
		"""
		# Variables resultantes: unión manteniendo el orden relativo (estilo join)
		vars_resultado = list(dict.fromkeys(self.variables + otro.variables))
		
		# Índices de cada variable en los factores originales
		# Si una variable no está en un factor, su índice queda en None (no se usa)
		idx_self = {v: self.variables.index(v) if v in self.variables else None 
		            for v in vars_resultado}
		idx_otro = {v: otro.variables.index(v) if v in otro.variables else None 
		            for v in vars_resultado}
		
		# Nueva tabla del producto
		nueva_tabla = {}
		
		# Generar todas las asignaciones posibles para las variables resultado
		for asignacion in product([True, False], repeat=len(vars_resultado)):
			# Construir claves para cada factor original respetando su orden interno
			clave_self = tuple(asignacion[idx_self[v]] 
			                   for v in self.variables) if idx_self else ()
			clave_otro = tuple(asignacion[idx_otro[v]] 
			                   for v in otro.variables) if idx_otro else ()
			
			# Multiplicar valores si ambas claves existen en las tablas
			# Nota: si alguna combinación no aparece en un factor (modelo esparso),
			# simplemente se omite (equivale a valor 0).
			if clave_self in self.tabla and clave_otro in otro.tabla:
				nueva_tabla[asignacion] = self.tabla[clave_self] * otro.tabla[clave_otro]
		
		return Factor(vars_resultado, nueva_tabla)
	
	def sumar_variable(self, var: str) -> 'Factor':
		"""
		Suma (marginaliza) una variable del factor.
		f(X,Y,Z) --sumar Y--> f'(X,Z) = Σ_Y f(X,Y,Z)
		"""
		if var not in self.variables:
			# Si la variable no está en el factor, retornar una copia
			# Esto permite componer operaciones sin chequear pertenencia externamente
			return Factor(self.variables, self.tabla)
		
		# Variables resultantes: todas menos la que se marginaliza
		vars_resultado = [v for v in self.variables if v != var]
		idx_var = self.variables.index(var)
		
		# Nueva tabla: sumamos sobre los valores de var
		nueva_tabla = {}
		
		for clave, valor in self.tabla.items():
			# Construir clave sin la variable a marginalizar (eliminando idx_var)
			nueva_clave = tuple(clave[i] for i in range(len(clave)) if i != idx_var)
			
			# Acumular suma para las dos asignaciones posibles de 'var'
			if nueva_clave in nueva_tabla:
				nueva_tabla[nueva_clave] += valor
			else:
				nueva_tabla[nueva_clave] = valor
		
		return Factor(vars_resultado, nueva_tabla)


def eliminacion_de_variables(factores: List[Factor], orden_eliminacion: List[str], 
                              consulta_vars: List[str]) -> Factor:
	"""
	Algoritmo de Eliminación de Variables para inferencia exacta.
	
	1. Para cada variable en orden_eliminacion:
	   - Multiplicar todos los factores que contienen la variable
	   - Sumar (marginalizar) la variable del producto
	   - Reemplazar esos factores por el resultado
	2. Multiplicar los factores restantes
	3. Normalizar si es necesario
	"""
	# Copiar lista de factores para no modificar la original
	factores_actuales = list(factores)
	
	print(f"\n>>> Eliminación de variables en orden: {orden_eliminacion}")
	
	for var in orden_eliminacion:
		# Encontrar factores que contienen esta variable
		factores_con_var = [f for f in factores_actuales if var in f.variables]
		factores_sin_var = [f for f in factores_actuales if var not in f.variables]
		
		if not factores_con_var:
			continue
		
		print(f"\nEliminando variable '{var}':")
		print(f"  Factores a multiplicar: {len(factores_con_var)}")
		
		# Multiplicar todos los factores que contienen var (asociatividad del producto)
		producto = factores_con_var[0]
		for f in factores_con_var[1:]:
			producto = producto.multiplicar(f)
		
		print(f"  Producto resultante: {producto}")
		
		# Sumar (marginalizar) var del producto → elimina la variable del grafo de factores
		marginalizado = producto.sumar_variable(var)
		print(f"  Después de marginalizar '{var}': {marginalizado}")
		
		# Actualizar lista de factores: reemplazar los usados por el nuevo factor reducido
		factores_actuales = factores_sin_var + [marginalizado]
	
	# Multiplicar todos los factores restantes (contienen solo variables de consulta/evidencia)
	print(f"\n>>> Multiplicando {len(factores_actuales)} factores restantes...")
	resultado = factores_actuales[0]
	for f in factores_actuales[1:]:
		resultado = resultado.multiplicar(f)
	
	print(f"Factor final: {resultado}")
	return resultado


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Eliminación de Variables")
	print("="*70)
	
	# Red simple: A → B, A → C
	print("\n--- Red Simple: A → B, A → C ---")
	print("Consulta: P(B | C=true)")
	print("Factores iniciales: P(A), P(B|A), P(C|A)")
	
	# Factor P(A) (prior de A)
	f_A = Factor(['A'], {
		(True,): 0.6,
		(False,): 0.4
	})
	
	# Factor P(B|A) (CPT de B)
	f_B_dado_A = Factor(['A', 'B'], {
		(True, True): 0.7,
		(True, False): 0.3,
		(False, True): 0.2,
		(False, False): 0.8
	})
	
	# Factor P(C|A) (CPT de C) — se reducirá con la evidencia C=true
	f_C_dado_A = Factor(['A', 'C'], {
		(True, True): 0.8,
		(True, False): 0.2,
		(False, True): 0.1,
		(False, False): 0.9
	})
	
	# Evidencia: C=true → reducir factor de C eliminando la variable C
	# Queda un factor solo sobre A con los términos P(C=true|A)
	f_C_true = Factor(['A'], {
		(True,): 0.8,   # P(C=true|A=true)
		(False,): 0.1   # P(C=true|A=false)
	})
	
	# Orden de eliminación: eliminar A
	factores = [f_A, f_B_dado_A, f_C_true]
	orden = ['A']
	
	resultado = eliminacion_de_variables(factores, orden, ['B'])
	
	# Normalizar para obtener P(B|C=true)
	total = sum(resultado.tabla.values())
	print(f"\n>>> Normalizando:")
	for clave, valor in resultado.tabla.items():
		b_val = clave[0]
		prob_normalizada = valor / total
		print(f"  P(B={b_val} | C=true) = {prob_normalizada:.4f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Eliminación de Variables")
	print("="*70)
	print("Ejemplo simplificado con red A → B, A → C")
	print("\nUsaremos parámetros predefinidos para la demo.")
	
	try:
		c_obs = input("Valor observado de C (true/false): ").strip().lower()
		c_valor = c_obs.startswith('t')
	except:
		c_valor = True
		print("Usando C=true por defecto")
	
	# Factores predefinidos (reutilizamos los del demo)
	f_A = Factor(['A'], {(True,): 0.6, (False,): 0.4})
	f_B_dado_A = Factor(['A', 'B'], {
		(True, True): 0.7, (True, False): 0.3,
		(False, True): 0.2, (False, False): 0.8
	})
	f_C_dado_A = Factor(['A', 'C'], {
		(True, True): 0.8, (True, False): 0.2,
		(False, True): 0.1, (False, False): 0.9
	})
	
	# Reducir según evidencia
	if c_valor:
		f_C_reducido = Factor(['A'], {(True,): 0.8, (False,): 0.1})
	else:
		f_C_reducido = Factor(['A'], {(True,): 0.2, (False,): 0.9})
	
	factores = [f_A, f_B_dado_A, f_C_reducido]
	orden = ['A']
	
	print(f"\nCalculando P(B | C={c_valor})...")
	resultado = eliminacion_de_variables(factores, orden, ['B'])
	
	total = sum(resultado.tabla.values())
	print(f"\n>>> Distribución posterior de B:")
	for clave, valor in resultado.tabla.items():
		b_val = clave[0]
		prob = valor / total
		print(f"  P(B={b_val} | C={c_valor}) = {prob:.4f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("ELIMINACIÓN DE VARIABLES")
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

