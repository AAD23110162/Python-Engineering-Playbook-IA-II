"""
051-E2-traduccion_automatica_estadistica.py
--------------------------------
Este script presenta Traducción Automática Estadística (SMT):
- Modelos de alineación palabra-palabra (IBM) a nivel conceptual.
- Modelos de frase y decodificación basada en búsquedas aproximadas.
- Funciones de puntuación y ajuste con MERT a nivel conceptual.
- Contraste con enfoques neuronales modernos.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: traducción de frases simples con tablas de ejemplo.
2. INTERACTIVO: configuración de tablas de traducción y puntuación básica.

Autor: Alejandro Aguirre Díaz
"""
import math
from collections import defaultdict

# -----------------------------
# IBM Model 1 (simplificado)
# -----------------------------
def entrenar_ibm1(pares_bitexto, iteraciones=5):
	"""
	Entrena t(f|e) con EM a partir de pares (f, e).
	f: frase origen (lista de palabras)  e: frase destino (lista de palabras)
	"""
	# Vocabularios
	V_e = set()
	V_f = set()
	for f, e in pares_bitexto:
		V_e.update(e)
		V_f.update(f)
	V_e.add('NULL')  # palabra nula

	# Inicialización uniforme
	t = defaultdict(lambda: 1.0 / len(V_e))  # t[(f,e)]

	for _ in range(iteraciones):
		count_fe = defaultdict(float)
		total_e = defaultdict(float)

		for f_sent, e_sent in pares_bitexto:
			e_ext = ['NULL'] + e_sent
			# s_total(f) = sum_e t(f|e)
			s_total = {}
			for f in f_sent:
				s_total[f] = sum(t[(f, e)] for e in e_ext)
			# Acumular cuentas
			for f in f_sent:
				for e in e_ext:
					c = t[(f, e)] / s_total[f]
					count_fe[(f, e)] += c
					total_e[e] += c

		# Maximización
		for (f, e), val in count_fe.items():
			t[(f, e)] = val / max(1e-12, total_e[e])

	return t

def traducir_palabra_a_palabra(oracion_f, t, V_e):
	"""
	Traduce por argmax_e t(f|e) para cada palabra f.
	"""
	traduccion = []
	candidatos = list(V_e - {'NULL'})
	for f in oracion_f:
		mejor_e = max(candidatos, key=lambda e: t[(f, e)]) if candidatos else 'NULL'
		traduccion.append(mejor_e)
	return traduccion

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: IBM Model 1 simplificado ---")
	# Pares español-inglés (f=es, e=en)
	bitexto = [
		("el gato come".split(), "the cat eats".split()),
		("el perro corre".split(), "the dog runs".split()),
		("el gato corre".split(), "the cat runs".split()),
	]
	t = entrenar_ibm1(bitexto, iteraciones=8)
	V_e = set(['NULL'] + "the cat eats dog runs".split())
	frase = "el gato corre".split()
	print("Origen:", ' '.join(frase))
	traduccion = traducir_palabra_a_palabra(frase, t, V_e)
	print("Traducción:", ' '.join(traduccion))

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Entrena con tu bitexto ---")
	try:
		n = int(input("Número de pares de entrenamiento: "))
		pares = []
		for i in range(n):
			f = input(f"Origen (palabras separadas por espacio) {i+1}: ").strip().lower().split()
			e = input(f"Destino (palabras separadas por espacio) {i+1}: ").strip().lower().split()
			pares.append((f, e))
		it = int(input("Iteraciones EM (ej. 6): "))
	except Exception:
		print("Entrada inválida. Usando bitexto por defecto.")
		pares = [
			("la casa azul".split(), "the blue house".split()),
			("la casa grande".split(), "the big house".split()),
		]
		it = 6
	t = entrenar_ibm1(pares, iteraciones=it)
	V_e = set(['NULL']) | {w for _, e in pares for w in e}
	while True:
		texto = input("Frase origen para traducir (o 'q' para salir): ").strip().lower()
		if texto == 'q':
			break
		f = texto.split()
		trad = traducir_palabra_a_palabra(f, t, V_e)
		print("Traducción:", ' '.join(trad))

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	print("\nScript 051-E2-traduccion_automatica_estadistica.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (IBM Model 1 simplificado)")
	print("2. INTERACTIVO (entrena y traduce)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
