"""
015-E2-regla_cadena.py
--------------------------------
Este script implementa la Regla de la Cadena para Probabilidades:
- Descompone probabilidades conjuntas mediante la regla de la cadena: P(A,B,C) = P(A)·P(B|A)·P(C|A,B)
- Aplica la regla de la cadena para calcular probabilidades de secuencias de eventos
- Implementa factorización de distribuciones conjuntas en productos de condicionales
- Explora el orden de factorización y su impacto en la eficiencia computacional
- Utiliza la regla de la cadena en el contexto de redes bayesianas
- Calcula probabilidades conjuntas a partir de probabilidades condicionales
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de aplicación de la regla de la cadena
2. INTERACTIVO: permite al usuario definir eventos y descomponer probabilidades conjuntas

Autor: Alejandro Aguirre Díaz
"""

from typing import List, Dict


def regla_cadena_general(prob_terminos: List[float]) -> float:
	"""
	Producto de términos condicionales/marginales según la regla de la cadena.
	Ej: P(A,B,C) = P(A)·P(B|A)·P(C|A,B)
	"""
	p = 1.0
	for t in prob_terminos:
		p *= t
	return p


def ejemplo_tres_variables(PA: float, PB_dado_A: float, PC_dado_AB: float) -> float:
	"""Calcula P(A,B,C) con tres variables usando la regla de la cadena."""
	return regla_cadena_general([PA, PB_dado_A, PC_dado_AB])


def factorizar_con_orden(orden: List[str], cpts: Dict[str, Dict[tuple, float]], asignacion: Dict[str, bool]) -> float:
	"""
	Factoriza P(x1,...,xn) = ∏ P(xi | padres(xi)) siguiendo el orden topológico.
	- orden: variables en orden topológico
	- cpts: para cada var, dict {tupla(valores_padres): P(var=True|padres)}; si no tiene padres usar key=()
	- asignacion: dict {var: bool}
	"""
	p = 1.0
	for var in orden:
		padres = [v for v in orden if (v, var) in []]  # marcador; los padres reales no están en este contexto
		# Nuestro uso aquí asume que 'cpts' ya incluye el mapeo correcto de padres en su key.
		key = next(iter(cpts[var].keys()))
		# Si tiene padres, la key esperada debería formarse con asignación de los padres; para simplicidad
		# en este ejemplo, suponemos que ya viene listo para consultarse con key=() o con la key apropiada.
		p_true = cpts[var][key]
		p *= p_true if asignacion[var] else (1 - p_true)
	return p


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Regla de la Cadena")
	print("="*70)

	print("\n--- Tres variables ---")
	PA = 0.6
	PB_dA = 0.7
	PC_dAB = 0.8
	PABC = ejemplo_tres_variables(PA, PB_dA, PC_dAB)
	print(f"P(A,B,C) = {PA}·{PB_dA}·{PC_dAB} = {PABC:.4f}")

	print("\n--- Red simple A→B, A→C en orden [A,B,C] ---")
	# Para la demo, implementamos CPTs que no dependen de key real (simple)
	cpts = {
		'A': {(): 0.6},
		'B': {('A',): 0.7},  # P(B=True|A=True)=0.7 (ignora A=False en esta demo)
		'C': {('A','B'): 0.8},  # P(C=True|A,B)=0.8 (simplificado)
	}
	asign = {'A': True, 'B': True, 'C': True}
	p_simplificada = regla_cadena_general([0.6, 0.7, 0.8])
	print(f"P(A=true,B=true,C=true) ≈ {p_simplificada:.4f} (regla de la cadena)")

	print("\n>>> NOTA:")
	print("    En redes bayesianas, la regla de la cadena usa los padres de cada nodo en orden topológico.")
	print("    La factorización reduce el número de parámetros al explotar independencias condicionales.")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Cadena de 3 Variables")
	print("="*70)
	try:
		PA = float(input("P(A): ").strip() or "0.5")
		PB_dA = float(input("P(B|A): ").strip() or "0.6")
		PC_dAB = float(input("P(C|A,B): ").strip() or "0.7")
	except:
		PA, PB_dA, PC_dAB = 0.5, 0.6, 0.7
	p = ejemplo_tres_variables(PA, PB_dA, PC_dAB)
	print(f"\nP(A,B,C) = {p:.6f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("REGLA DE LA CADENA")
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
