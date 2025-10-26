"""
016-E2-manto_de_markov.py
--------------------------------
Este script introduce el concepto de "Manto de Markov" (contexto/vecindad de Markov) sobre variables:
- Define el conjunto de variables que hacen a una variable condicionalmente independiente del resto.
- Relaciona mantos de Markov con d-separación en grafos dirigidos y no dirigidos.
- Ilustra cómo el manto de Markov simplifica el cálculo de probabilidades condicionadas.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplos sobre pequeños grafos para identificar mantos de Markov.
2. INTERACTIVO: permite cargar estructuras y consultar mantos de Markov de nodos.

Autor: Alejandro Aguirre Díaz
"""

from typing import Dict, Set, List


class GrafoDirigido:
	"""Representa un grafo dirigido (DAG) para red bayesiana."""
	
	def __init__(self):
		# Almacena padres de cada nodo
		# Clave: nombre del nodo | Valor: conjunto de nombres de nodos padres
		# Estructuras dispersas (sets) para evitar duplicados y permitir búsquedas O(1)
		self.padres: Dict[str, Set[str]] = {}
		# Almacena hijos de cada nodo (calculado al añadir aristas)
		# Clave: nombre del nodo | Valor: conjunto de nombres de nodos hijos
		self.hijos: Dict[str, Set[str]] = {}
	
	def agregar_nodo(self, nombre: str):
		"""Agrega un nodo al grafo."""
		# Inicializamos sus conjuntos si no existían para evitar KeyError al acceder
		if nombre not in self.padres:
			self.padres[nombre] = set()
		if nombre not in self.hijos:
			self.hijos[nombre] = set()
	
	def agregar_arista(self, desde: str, hacia: str):
		"""Agrega arista dirigida: desde → hacia."""
		# Aseguramos que los nodos existan
		self.agregar_nodo(desde)
		self.agregar_nodo(hacia)
		
		# Añadimos la relación padre-hijo
		# Nota: No validamos aciclicidad aquí; asumimos que el usuario construye un DAG
		# Usar 'set' evita registrar la misma arista dos veces
		self.padres[hacia].add(desde)
		self.hijos[desde].add(hacia)
	
	def obtener_padres(self, nodo: str) -> Set[str]:
		"""Retorna el conjunto de padres de un nodo."""
		# Si el nodo no existe, devolvemos conjunto vacío para una API más segura
		return self.padres.get(nodo, set())
	
	def obtener_hijos(self, nodo: str) -> Set[str]:
		"""Retorna el conjunto de hijos de un nodo."""
		# Si el nodo no existe, devolvemos conjunto vacío para una API más segura
		return self.hijos.get(nodo, set())
	
	def obtener_conyuge(self, nodo: str) -> Set[str]:
		"""
		Retorna los 'cónyuges' (co-padres) de un nodo: otros padres de sus hijos.
		En una red bayesiana, si X→Z y Y→Z, entonces Y es cónyuge de X.
		"""
		conyuges = set()
		# Para cada hijo del nodo
		for hijo in self.obtener_hijos(nodo):
			# Obtenemos todos los padres del hijo
			otros_padres = self.obtener_padres(hijo)
			# Añadimos los que no son el nodo mismo
			# Si el nodo no tiene hijos, este bucle no itera y conyuges queda vacío
			conyuges.update(otros_padres - {nodo})
		return conyuges
	
	def manto_markov(self, nodo: str) -> Set[str]:
		"""
		Calcula el Manto de Markov de un nodo en una red bayesiana dirigida.
		
		El Manto de Markov de X consiste en:
		1. Los padres de X
		2. Los hijos de X
		3. Los otros padres de los hijos de X (cónyuges)
		
		Dado el Manto de Markov, X es condicionalmente independiente del resto
		de variables en la red.
		"""
		# Conjunto acumulador del manto (el propio nodo NO se incluye por construcción)
		manto = set()
		
		# Paso 1: añadir todos los padres directos de 'nodo'
		manto.update(self.obtener_padres(nodo))
		
		# Paso 2: añadir todos los hijos directos de 'nodo'
		hijos = self.obtener_hijos(nodo)
		manto.update(hijos)
		
		# Paso 3: añadir cónyuges (co-padres de los hijos del 'nodo')
		manto.update(self.obtener_conyuge(nodo))
		
		# Nota: No agregamos 'nodo' al manto; el manto son las variables mínimas que, al condicionarlas,
		# hacen a 'nodo' independiente del resto de la red.
		return manto


# ========== FUNCIONES AUXILIARES ==========

def construir_red_alarma() -> GrafoDirigido:
	"""
	Construye la red clásica de alarma:
	B → A ← E
	A → J
	A → M
	"""
	g = GrafoDirigido()
	# Estructura de la red
	# B: Robo (Burglary)
	# E: Terremoto (Earthquake)
	# A: Alarma (Alarm)
	# J: Llamada de John (JohnCalls)
	# M: Llamada de Mary (MaryCalls)
	g.agregar_arista('B', 'A')  # Burglary → Alarm
	g.agregar_arista('E', 'A')  # Earthquake → Alarm
	g.agregar_arista('A', 'J')  # Alarm → JohnCalls
	g.agregar_arista('A', 'M')  # Alarm → MaryCalls
	return g


def construir_red_simple() -> GrafoDirigido:
	"""
	Construye una red simple:
	A → B → C
	A → D
	"""
	g = GrafoDirigido()
	# Dos caminos que salen de A: uno hacia B (y luego C), y otro hacia D
	g.agregar_arista('A', 'B')
	g.agregar_arista('B', 'C')
	g.agregar_arista('A', 'D')
	return g


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Manto de Markov")
	print("="*70)
	
	# Ejemplo 1: Red de alarma
	print("\n--- Red de Alarma: B → A ← E, A → J, A → M ---")
	red_alarma = construir_red_alarma()
	
	# Calculamos y mostramos el manto de Markov de cada nodo de la red de alarma
	nodos = ['B', 'E', 'A', 'J', 'M']
	for nodo in nodos:
		manto = red_alarma.manto_markov(nodo)
		print(f"Manto de Markov de {nodo}: {manto if manto else '∅ (vacío)'}")
	
	print("\nInterpretación:")
	print("- A tiene el manto más grande: padres (B,E) + hijos (J,M)")
	print("- B y E son cónyuges (ambos son padres de A)")
	print("- J y M solo tienen a A como padre (manto = {A})")
	
	# Ejemplo 2: Red simple
	print("\n--- Red Simple: A → B → C, A → D ---")
	red_simple = construir_red_simple()
	
	# Repetimos el análisis para la red simple
	nodos_simple = ['A', 'B', 'C', 'D']
	for nodo in nodos_simple:
		manto = red_simple.manto_markov(nodo)
		print(f"Manto de Markov de {nodo}: {manto if manto else '∅ (vacío)'}")
	
	print("\nInterpretación:")
	print("- B está en el manto de A porque es hijo de A")
	print("- C está en el manto de B (hijo), pero NO en el manto de A")
	print("- Dado el manto, cada variable es independiente del resto")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Consultar Manto de Markov")
	print("="*70)
	print("Usaremos la red de alarma predefinida.")
	print("Red: B → A ← E, A → J, A → M")
	
	# Construimos la red base y definimos el universo de nodos válidos
	red = construir_red_alarma()
	nodos_disponibles = ['B', 'E', 'A', 'J', 'M']
	
	print(f"\nNodos disponibles: {', '.join(nodos_disponibles)}")
	# Normalizamos la entrada a mayúsculas para coincidir con las etiquetas
	nodo = input("Ingresa el nodo para consultar su Manto de Markov: ").strip().upper()
	
	if nodo in nodos_disponibles:
		# Calculamos componentes del manto para mostrar un desglose útil
		manto = red.manto_markov(nodo)
		padres = red.obtener_padres(nodo)
		hijos = red.obtener_hijos(nodo)
		conyuges = red.obtener_conyuge(nodo)
		
		print(f"\n--- Análisis del nodo {nodo} ---")
		print(f"Padres: {padres if padres else '∅'}")
		print(f"Hijos: {hijos if hijos else '∅'}")
		print(f"Cónyuges (otros padres de hijos): {conyuges if conyuges else '∅'}")
		print(f"Manto de Markov: {manto if manto else '∅'}")
		
		# Propiedad clave: condicionando en el manto, el nodo queda aislado del resto de la red
		print(f"\nPropiedad: Dado {manto if manto else 'nada'}, {nodo} es independiente del resto.")
	else:
		print(f"Nodo '{nodo}' no encontrado en la red.")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("MANTO DE MARKOV")
	print("="*70)
	print("1. DEMO")
	print("2. INTERACTIVO")
	opcion = input("\nIngresa opción (1/2): ").strip()
	if opcion == '1':
		modo_demo()
	elif opcion == '2':
		modo_interactivo()
	else:
		# Entrada no válida: por defecto ejecutamos la DEMO para mostrar ejemplos
		modo_demo()
	print("\n" + "="*70)
	print("FIN DEL PROGRAMA")
	print("="*70 + "\n")


if __name__ == "__main__":
	main()
