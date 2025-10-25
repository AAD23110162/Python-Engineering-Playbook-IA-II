"""
034-E1-refuerzo_activo.py
-------------------------
Este script implementa un algoritmo de Aprendizaje por Refuerzo Activo usando Q-Learning:
- A diferencia del refuerzo pasivo, aquí el agente APRENDE una política óptima π* seleccionando acciones.
- Soporta entornos discretos definidos por:
  • Estados (lista de etiquetas)
  • Transiciones P(s'|s,a) como diccionario {(s,a): {s': prob}}
  • Recompensas R(s,a,s') como diccionario {(s,a,s'): r}
- Estrategia de exploración: ε-greedy (con parámetro ε configurable).
- Dos modos de uso:
  1) MODO DEMO: problema sencillo de 3 estados (coherente con 033).
  2) MODO INTERACTIVO: seleccionar escenario (3 estados o GridWorld 2x2) y parámetros de aprendizaje.

Notas didácticas:
- Q-Learning actualiza Q(s,a) con la regla: Q ← Q + α [ r + γ·max_a' Q(s',a') − Q ]
- La política inducida es greedy respecto a Q: π(s) = argmax_a Q(s,a)

Autor: Alejandro Aguirre Díaz
"""

import random
from collections import defaultdict


# ========== Funciones utilitarias ==========

def acciones_disponibles(estado, transiciones):
	"""
	Devuelve la lista de acciones disponibles en un estado según el diccionario de transiciones.
	Si una acción no está definida para (estado, acción), se considera no disponible.
	"""
	# Recorremos todas las claves (s,a) del modelo de transición
	# y recolectamos aquellas acciones 'a' cuyo estado 's' coincide con el solicitado.
	disponibles = []
	for (s, a) in transiciones.keys():
		# Si el estado coincide y no hemos agregado la acción aún, la agregamos
		if s == estado and a not in disponibles:
			disponibles.append(a)
	# Puede ser lista vacía si el estado no tiene acciones disponibles (p.ej., terminal)
	return disponibles


def elegir_accion_epsilon_greedy(Q, estado, transiciones, epsilon):
	"""
	Selecciona una acción ε-greedy respecto a Q en el estado dado.
	- Con prob ε elige una acción aleatoria entre las disponibles.
	- Con prob 1-ε elige la acción con mayor Q.
	Si no hay acciones disponibles, devuelve None.
	"""
	# Obtener conjunto de acciones permitidas en el estado actual
	acciones = acciones_disponibles(estado, transiciones)
	if not acciones:
		# Estado sin acciones: se interpreta como terminal o sin dinámica definida
		return None
	# Regla ε-greedy: con probabilidad ε exploramos una acción al azar
	if random.random() < epsilon:
		# Explorar (acción aleatoria entre las disponibles)
		return random.choice(acciones)
	# Con probabilidad 1-ε explotamos: elegimos la acción con mayor Q(s,a)
	mejor_a = max(acciones, key=lambda a: Q[(estado, a)])
	return mejor_a


# ========== Núcleo de aprendizaje: Q-Learning ==========

def q_learning(estados, transiciones, recompensas, gamma, alpha, epsilon, episodios, max_pasos=100):
	"""
	Ejecuta Q-Learning en un MDP finito.
	:parametro estados: lista de estados
	:parametro transiciones: dict {(s,a): {s': prob}}
	:parametro recompensas: dict {(s,a,s'): r}
	:parametro gamma: factor de descuento (0<=γ<1)
	:parametro alpha: tasa de aprendizaje (0<α<=1)
	:parametro epsilon: prob de exploración ε en ε-greedy
	:param episodios: cantidad de episodios de entrenamiento
	:param max_pasos: límite de pasos por episodio para evitar bucles largos
	:return: (Q, V, politica) donde
			 Q: dict {(s,a): valor}
			 V: dict {s: max_a Q(s,a)}
			 politica: dict {s: argmax_a Q(s,a)}
	"""
	# Inicialización: Q(s,a) = 0 para todo par (s,a) implícitamente
	# Usamos defaultdict(float) para devolver 0.0 cuando la clave no exista aún
	Q = defaultdict(float)

	# Bucle principal de entrenamiento: iteramos sobre episodios
	for ep in range(episodios):
		# Seleccionamos un estado inicial al azar (no hay estado inicial fijo)
		s = random.choice(estados)

		# Para cada episodio, limitamos el número de pasos para evitar lazos largos
		for t in range(max_pasos):
			# 1) Selección de acción según política ε-greedy inducida por Q
			a = elegir_accion_epsilon_greedy(Q, s, transiciones, epsilon)
			if a is None:
				# Sin acciones: detenemos el episodio (estado terminal)
				break

			# 2) Dinámica del entorno: muestrear s' ~ P(·|s,a)
			distrib = transiciones.get((s, a), {})
			if not distrib:
				# No hay transiciones definidas para (s,a): terminar episodio
				break
			estados_sig = list(distrib.keys())
			probs = list(distrib.values())
			s_prime = random.choices(estados_sig, weights=probs)[0]

			# 3) Obtener recompensa inmediata r = R(s,a,s')
			r = recompensas.get((s, a, s_prime), 0.0)

			# 4) Objetivo de TD (off-policy): r + γ · max_{a'} Q(s',a')
			acciones_s_prime = acciones_disponibles(s_prime, transiciones)
			max_q_s_prime = max((Q[(s_prime, ap)] for ap in acciones_s_prime), default=0.0)

			# 5) Actualización Q-Learning: incremento hacia el objetivo TD
			td_objetivo = r + gamma * max_q_s_prime
			td_error = td_objetivo - Q[(s, a)]
			Q[(s, a)] += alpha * td_error

			# 6) Transición al siguiente estado
			s = s_prime

	# Una vez entrenado, derivamos V(s) y la política greedy π(s)
	V = {}
	politica = {}
	for s in estados:
		# Conjunto de acciones disponibles en s
		accs = acciones_disponibles(s, transiciones)
		if accs:
			# Acción con mayor valor Q(s,a)
			mejor_a = max(accs, key=lambda a: Q[(s, a)])
			V[s] = Q[(s, mejor_a)]
			politica[s] = mejor_a
		else:
			# Si no hay acciones, consideramos valor 0 y política indefinida
			V[s] = 0.0
			politica[s] = None

	return Q, V, politica


# ========== Modo DEMO ==========

def modo_demo():
	print("\nMODO DEMO: Refuerzo Activo con Q-Learning")
	print("=" * 60)

	# Problema sencillo coherente con 033: estados A, B, C
	estados = ['A', 'B', 'C']

	# Transiciones P(s'|s,a)
	transiciones = {
		('A','x'): {'A':0.8,'B':0.2},
		('A','y'): {'B':1.0},
		('B','x'): {'C':1.0},
		('B','y'): {'A':0.5,'C':0.5},
		('C','x'): {'C':1.0},
		('C','y'): {'A':1.0}
	}

	# Recompensas R(s,a,s')
	recompensas = {
		('A','x','A'): 2,
		('A','x','B'): 0,
		('A','y','B'): 1,
		('B','x','C'): 4,
		('B','y','A'): -1,
		('B','y','C'): 3,
		('C','x','C'): 0,
		('C','y','A'): 2
	}

	# Hiperparámetros razonables para demo
	gamma = 0.9    # descuento
	alpha = 0.5    # tasa de aprendizaje
	epsilon = 0.2  # exploración
	episodios = 3000

	print("\nCONFIGURACIÓN:")
	print(f"Estados: {estados}")
	print(f"γ={gamma}, α={alpha}, ε={epsilon}, episodios={episodios}")

	# Entrenar la Q-table con los hiperparámetros indicados
	print("\nEntrenando con Q-Learning...")
	Q, V, politica = q_learning(estados, transiciones, recompensas, gamma, alpha, epsilon, episodios)

	print("\nRESULTADOS:")
	print("Política aprendida π*(aprox):")
	for s in estados:
		print(f"  {s} -> {politica[s]}")
	# V(s) se calcula como el máximo valor Q(s,a) sobre las acciones disponibles en s
	print("\nValores de estado V(s) ≈ max_a Q(s,a):")
	for s in estados:
		print(f"  {s}: {V[s]:.3f}")


# ========== Modo INTERACTIVO ==========

def modo_interactivo():
	print("\nMODO INTERACTIVO: Refuerzo Activo con Q-Learning")
	print("=" * 60)
	print("\nEscenarios predefinidos:")
	print("1) Entorno simple 3 estados (A, B, C)")
	print("2) Entorno GridWorld 2x2")

	# Solicitar escenario al usuario
	opcion = input("\nIntroduce el número de escenario: ").strip()

	if opcion == '2':
		# GridWorld 2x2
		# Estados son celdas de una cuadrícula 2x2; objetivo es alcanzar (1,1)
		estados = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
		transiciones = {
			('(0,0)','derecha'): {'(0,1)':1.0},
			('(0,0)','abajo'): {'(1,0)':1.0},
			('(0,1)','izquierda'): {'(0,0)':1.0},
			('(0,1)','abajo'): {'(1,1)':1.0},
			('(1,0)','arriba'): {'(0,0)':1.0},
			('(1,0)','derecha'): {'(1,1)':1.0},
			('(1,1)','arriba'): {'(0,1)':1.0},
			('(1,1)','izquierda'): {'(1,0)':1.0},
			('(1,1)','abajo'): {'(1,1)':1.0}
		}
		recompensas = {
			('(0,0)','derecha','(0,1)'): -1,
			('(0,0)','abajo','(1,0)'): -1,
			('(0,1)','izquierda','(0,0)'): -1,
			('(0,1)','abajo','(1,1)'): 10,
			('(1,0)','arriba','(0,0)'): -1,
			('(1,0)','derecha','(1,1)'): 10,
			('(1,1)','arriba','(0,1)'): -1,
			('(1,1)','izquierda','(1,0)'): -1,
			('(1,1)','abajo','(1,1)'): 0
		}
		print("\nHas elegido GridWorld 2x2")
	else:
		# 3 estados
		# Entorno sencillo coherente con el modo DEMO
		estados = ['A','B','C']
		transiciones = {
			('A','x'): {'A':0.8,'B':0.2},
			('A','y'): {'B':1.0},
			('B','x'): {'C':1.0},
			('B','y'): {'A':0.5,'C':0.5},
			('C','x'): {'C':1.0},
			('C','y'): {'A':1.0}
		}
		recompensas = {
			('A','x','A'): 2,
			('A','x','B'): 0,
			('A','y','B'): 1,
			('B','x','C'): 4,
			('B','y','A'): -1,
			('B','y','C'): 3,
			('C','x','C'): 0,
			('C','y','A'): 2
		}
		print("\nHas elegido el entorno simple de 3 estados")

	# Mostrar resumen del escenario elegido
	print(f"\nEstados disponibles: {estados}")

	# Pedir hiperparámetros
	# γ: descuento; α: tasa aprendizaje; ε: prob exploración; episodios: iteraciones de entrenamiento
	try:
		gamma = float(input("\nIntroduce factor de descuento γ (0-1, ej 0.9): ").strip())
		alpha = float(input("Introduce tasa de aprendizaje α (0-1, ej 0.5): ").strip())
		epsilon = float(input("Introduce exploración ε (0-1, ej 0.2): ").strip())
		episodios = int(input("Introduce número de episodios (ej 5000): ").strip())
	except Exception:
		print("Parámetros inválidos. Usando valores por defecto.")
		gamma, alpha, epsilon, episodios = 0.9, 0.5, 0.2, 3000

	# Entrenamiento principal de Q-Learning con los parámetros elegidos
	print("\nEntrenando con Q-Learning...")
	Q, V, politica = q_learning(estados, transiciones, recompensas, gamma, alpha, epsilon, episodios)

	print("\nRESULTADOS:")
	print("Política aprendida π*(aprox):")
	for s in estados:
		print(f"  {s} -> {politica[s]}")
	# V(s) aproximado a partir de Q aprendido
	print("\nValores de estado V(s) ≈ max_a Q(s,a):")
	for s in estados:
		print(f"  {s}: {V[s]:.3f}")


def main():
	print("Seleccione modo de ejecución:")
	print("1) Modo DEMO")
	print("2) Modo INTERACTIVO\n")
	opcion = input("Ingrese opción: ").strip()
	if opcion == '2':
		modo_interactivo()
	else:
		modo_demo()


if __name__ == "__main__":
	main()

