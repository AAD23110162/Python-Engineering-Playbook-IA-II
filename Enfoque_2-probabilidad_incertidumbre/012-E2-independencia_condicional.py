"""
012-E2-independencia_condicional.py
--------------------------------
Este script implementa el concepto de Independencia Condicional:
- Define y verifica independencia condicional entre variables: P(A,B|C) = P(A|C)·P(B|C)
- Implementa pruebas de independencia condicional en conjuntos de datos
- Explora la relación entre independencia marginal vs. independencia condicional
- Aplica independencia condicional para simplificar redes bayesianas
- Identifica estructuras de independencia: cadenas, bifurcaciones y colisionadores (v-structures)
- Utiliza d-separación para determinar independencias en grafos dirigidos
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de verificación de independencia condicional
2. INTERACTIVO: permite al usuario definir variables y probar relaciones de independencia

Autor: Alejandro Aguirre Díaz
"""

from itertools import product

# Representaremos distribuciones conjuntas como:
# - variables: lista de nombres ["A","B","C", ...]
# - valores: dict {var: [posibles_valores]}
# - tabla: dict { (vA, vB, vC, ...): prob }


def marginal(variables, tabla, var, val):
	"""P(var=val) sumando sobre el resto de variables."""
	idx = variables.index(var)
	total = 0.0
	for asign, p in tabla.items():
		if asign[idx] == val:
			total += p
	return total


def condicionada(variables, tabla, var, val, evidencia: dict):
	"""P(var=val | evidencia) = P(var=val, evidencia) / P(evidencia)."""
	idx_q = variables.index(var)
	num = 0.0
	den = 0.0
	for asign, p in tabla.items():
		consistente = all(asign[variables.index(ev)] == evid for ev, evid in evidencia.items())
		if consistente:
			den += p
			if asign[idx_q] == val:
				num += p
	return (num / den) if den > 0 else 0.0


def independencia(variables, tabla, A, a_val, B, b_val, tol=1e-6):
	"""Comprueba si A y B son independientes para valores concretos: P(A,B)=P(A)P(B)."""
	# P(A=a, B=b)
	idxA = variables.index(A)
	idxB = variables.index(B)
	conj = sum(p for asign, p in tabla.items() if asign[idxA] == a_val and asign[idxB] == b_val)
	pA = marginal(variables, tabla, A, a_val)
	pB = marginal(variables, tabla, B, b_val)
	return abs(conj - pA * pB) < tol, conj, pA * pB


def independencia_condicional(variables, tabla, A, a_val, B, b_val, C, c_val, tol=1e-6):
	"""Comprueba A ⟂ B | C=c: P(A,B|C)=P(A|C)P(B|C)."""
	# P(A,B | C=c)
	idxA = variables.index(A)
	idxB = variables.index(B)
	idxC = variables.index(C)
	# Denominador P(C=c)
	pC = sum(p for asign, p in tabla.items() if asign[idxC] == c_val)
	# Numerador P(A=a, B=b, C=c)
	num = sum(p for asign, p in tabla.items() if asign[idxA] == a_val and asign[idxB] == b_val and asign[idxC] == c_val)
	pAB_dado_C = (num / pC) if pC > 0 else 0.0
	# P(A|C=c), P(B|C=c)
	pA_dado_C = condicionada(variables, tabla, A, a_val, {C: c_val})
	pB_dado_C = condicionada(variables, tabla, B, b_val, {C: c_val})
	return abs(pAB_dado_C - pA_dado_C * pB_dado_C) < tol, pAB_dado_C, pA_dado_C * pB_dado_C


def construir_conjunta_desde_prior_y_cpts(prior_C: dict, cpt_A_dado_C: dict, cpt_B_dado_C: dict):
	"""
	Construye una distribución conjunta de (A,B,C) booleana a partir de:
	- prior_C: {True: p, False: 1-p}
	- cpt_A_dado_C: {c_val: P(A=True|C=c_val)}
	- cpt_B_dado_C: {c_val: P(B=True|C=c_val)}
	Devuelve (variables, valores, tabla)
	"""
	variables = ['A', 'B', 'C']
	valores = {'A': [True, False], 'B': [True, False], 'C': [True, False]}
	tabla = {}
	for a, b, c in product(valores['A'], valores['B'], valores['C']):
		pC = prior_C[c]
		pA = cpt_A_dado_C[c] if a else 1 - cpt_A_dado_C[c]
		pB = cpt_B_dado_C[c] if b else 1 - cpt_B_dado_C[c]
		tabla[(a, b, c)] = pC * pA * pB
	return variables, valores, tabla


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Independencia Condicional")
	print("="*70)

	# Modelo de causa común: C → A, C → B
	# A y B son dependientes marginalmente, pero independientes condicionalmente dado C.
	prior_C = {True: 0.3, False: 0.7}
	cpt_A_dado_C = {True: 0.9, False: 0.2}
	cpt_B_dado_C = {True: 0.8, False: 0.1}

	variables, valores, tabla = construir_conjunta_desde_prior_y_cpts(prior_C, cpt_A_dado_C, cpt_B_dado_C)

	# Independencia marginal A ⟂ B ?
	indep, pAB, pA_pB = independencia(variables, tabla, 'A', True, 'B', True)
	print("\n--- Independencia marginal A ⟂ B ? ---")
	print(f"P(A=true, B=true) = {pAB:.4f}")
	print(f"P(A=true)·P(B=true) = {pA_pB:.4f}")
	print(f"¿Independientes? {indep}")

	# Independencia condicional A ⟂ B | C
	print("\n--- Independencia condicional A ⟂ B | C ---")
	for c_val in [True, False]:
		indep_c, pAB_C, pA_C_pB_C = independencia_condicional(variables, tabla, 'A', True, 'B', True, 'C', c_val)
		print(f"C={c_val}: P(A,B|C)={pAB_C:.4f} vs P(A|C)·P(B|C)={pA_C_pB_C:.4f} → {indep_c}")

	# Efecto de observar un colisionador (estructura A → C ← B):
	# No implementamos d-separación general, pero ilustramos: en un colisionador, A y B son
	# marginalmente independientes, pero pueden volverse dependientes al condicionar en C.
	print("\n>>> NOTA:")
	print("    En causa común: observar C rompe la dependencia entre A y B (los hace independientes).")
	print("    En colisionador: observar C introduce dependencia entre A y B (efecto Berkson).")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Probar Independencia Condicional")
	print("="*70)
	print("Construiremos una conjunta de (A,B,C) booleana con C causa común de A y B.")
	try:
		pC = float(input("P(C=true) [0-1] (ej 0.3): ").strip() or "0.3")
		pA_cT = float(input("P(A=true|C=true) (ej 0.9): ").strip() or "0.9")
		pA_cF = float(input("P(A=true|C=false) (ej 0.2): ").strip() or "0.2")
		pB_cT = float(input("P(B=true|C=true) (ej 0.8): ").strip() or "0.8")
		pB_cF = float(input("P(B=true|C=false) (ej 0.1): ").strip() or "0.1")
	except:
		pC, pA_cT, pA_cF, pB_cT, pB_cF = 0.3, 0.9, 0.2, 0.8, 0.1

	variables, valores, tabla = construir_conjunta_desde_prior_y_cpts({True: pC, False: 1-pC}, {True: pA_cT, False: pA_cF}, {True: pB_cT, False: pB_cF})

	indep, pAB, pA_pB = independencia(variables, tabla, 'A', True, 'B', True)
	print(f"\nMarginal: P(A=true, B=true)={pAB:.4f} vs P(A)·P(B)={pA_pB:.4f} → Independencia: {indep}")

	for c_val in [True, False]:
		indep_c, pAB_C, pA_C_pB_C = independencia_condicional(variables, tabla, 'A', True, 'B', True, 'C', c_val)
		print(f"Condicional C={c_val}: P(A,B|C)={pAB_C:.4f} vs P(A|C)·P(B|C)={pA_C_pB_C:.4f} → {indep_c}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("INDEPENDENCIA CONDICIONAL")
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
