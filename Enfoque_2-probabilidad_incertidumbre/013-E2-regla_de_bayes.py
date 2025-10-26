"""
013-E2-regla_de_bayes.py
--------------------------------
Este script implementa la Regla de Bayes:
- Aplica el Teorema de Bayes: P(H|E) = P(E|H)·P(H) / P(E)
- Calcula probabilidades a posteriori a partir de verosimilitud y probabilidad a priori
- Implementa clasificadores bayesianos ingenuos (Naive Bayes)
- Actualiza creencias mediante observación de nueva evidencia
- Aplica la regla de Bayes en diagnóstico médico, filtrado de spam y clasificación
- Muestra cómo la evidencia modifica las probabilidades previas
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de aplicación del teorema de Bayes
2. INTERACTIVO: permite al usuario ingresar probabilidades y calcular posteriores

Autor: Alejandro Aguirre Díaz
"""

import numpy as np

# ========== TEOREMA DE BAYES BÁSICO ==========

def bayes_posterior(prior: dict, likelihood: dict) -> dict:
	"""
	Calcula el posterior P(H|E) ∝ P(E|H)·P(H) para hipótesis H en 'prior'.
	- prior: {H: P(H)}
	- likelihood: {H: P(E|H)} para la misma familia de H
	Retorna un dict normalizado {H: P(H|E)}.
	"""
	numeradores = {h: prior.get(h, 0.0) * likelihood.get(h, 0.0) for h in prior}
	Z = sum(numeradores.values())
	if Z == 0:
		return {h: 0.0 for h in prior}
	return {h: v / Z for h, v in numeradores.items()}


# ========== NAIVE BAYES (DEMO PEQUEÑA) ==========

def naive_bayes_binario(doc: list[str], vocab: set[str], modelo: dict) -> dict:
	"""
	Clasificador Naive Bayes Bernoulli muy simple.
	- doc: lista de palabras
	- vocab: conjunto del vocabulario considerado
	- modelo: {
		'prior': {'spam': p, 'ham': 1-p},
		'likelihood': {
			'spam': {'palabra': P(w=1|spam), ...},
			'ham' : {'palabra': P(w=1|ham),  ...}
		}
	  }
	Retorna posteriors por clase.
	"""
	present = set(w.lower() for w in doc)
	log_prob = {c: np.log(modelo['prior'][c]) for c in modelo['prior']}
	for c in modelo['likelihood']:
		for w in vocab:
			pw1 = modelo['likelihood'][c].get(w, 0.5)  # smoothing implícito si falta
			pw0 = 1 - pw1
			if w in present:
				log_prob[c] += np.log(max(pw1, 1e-12))
			else:
				log_prob[c] += np.log(max(pw0, 1e-12))
	# Normalizar logs → probabilidades
	m = max(log_prob.values())
	exps = {c: np.exp(log_prob[c] - m) for c in log_prob}
	Z = sum(exps.values())
	return {c: exps[c] / Z for c in exps}


# ========== MODO DEMO ==========

def modo_demo():
	print("\n" + "="*70)
	print("MODO DEMO: Regla de Bayes")
	print("="*70)

	# Ejemplo 1: Diagnóstico médico (test, sensibilidad, especificidad)
	print("\n--- Diagnóstico Médico ---")
	prevalencia = 0.01  # P(Enfermedad)
	sensibilidad = 0.95 # P(+|Enfermedad)
	especificidad = 0.98 # P(-|Sano)
	prior = {'Enfermedad': prevalencia, 'Sano': 1 - prevalencia}
	like_pos = {'Enfermedad': sensibilidad, 'Sano': 1 - especificidad}
	posterior_pos = bayes_posterior(prior, like_pos)
	print(f"P(Enfermedad | Test=+) = {posterior_pos['Enfermedad']:.4f}")

	# Ejemplo 2: Naive Bayes (spam vs ham)
	print("\n--- Naive Bayes (Spam/Ham) ---")
	vocab = {"oferta", "gratis", "hola", "reunión"}
	modelo = {
		'prior': {'spam': 0.3, 'ham': 0.7},
		'likelihood': {
			'spam': {"oferta": 0.8, "gratis": 0.7, "hola": 0.2, "reunión": 0.1},
			'ham' : {"oferta": 0.1, "gratis": 0.1, "hola": 0.6, "reunión": 0.7}
		}
	}
	doc = ["Hola", "tienes", "una", "oferta", "gratis"]
	post = naive_bayes_binario(doc, vocab, modelo)
	print(f"Documento: {' '.join(doc)}")
	print(f"P(spam|doc)={post['spam']:.3f}, P(ham|doc)={post['ham']:.3f}")


# ========== MODO INTERACTIVO ==========

def modo_interactivo():
	print("\n" + "="*70)
	print("MODO INTERACTIVO: Calculadora de Bayes")
	print("="*70)
	print("Escenario binario: Enfermedad vs Sano con test diagnóstico.")
	try:
		prev = float(input("Prevalencia P(Enfermedad) (ej 0.01): ").strip() or "0.01")
		sens = float(input("Sensibilidad P(+|Enfermedad) (ej 0.95): ").strip() or "0.95")
		esp  = float(input("Especificidad P(-|Sano) (ej 0.98): ").strip() or "0.98")
	except:
		prev, sens, esp = 0.01, 0.95, 0.98
	prior = {'Enfermedad': prev, 'Sano': 1 - prev}
	like_pos = {'Enfermedad': sens, 'Sano': 1 - esp}
	like_neg = {'Enfermedad': 1 - sens, 'Sano': esp}
	post_pos = bayes_posterior(prior, like_pos)
	post_neg = bayes_posterior(prior, like_neg)
	print(f"\nP(Enfermedad|+)={post_pos['Enfermedad']:.4f},  P(Enfermedad|-)={post_neg['Enfermedad']:.4f}")


# ========== MAIN ==========

def main():
	print("\n" + "="*70)
	print("REGLA DE BAYES")
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
