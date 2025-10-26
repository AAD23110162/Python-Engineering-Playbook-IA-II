"""
009-E2-probabilidad_a_priori.py
--------------------------------
Este script implementa el concepto de Probabilidad A Priori:
- Define y calcula probabilidades previas sin información observacional
- Establece distribuciones de probabilidad basadas en conocimiento previo
- Implementa diferentes métodos para asignar probabilidades a priori (uniforme, informativa)
- Compara probabilidades a priori vs. a posteriori después de observar evidencia
- Aplica el principio de indiferencia para casos sin información previa
- Muestra cómo las probabilidades a priori influyen en la inferencia bayesiana
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos con diferentes distribuciones a priori
2. INTERACTIVO: permite al usuario definir creencias previas y observar su impacto

Autor: Alejandro Aguirre Díaz
"""

import numpy as np
from collections import Counter

# ========== DISTRIBUCIONES A PRIORI ==========

class DistribucionAPriori:
    """Representa una distribución de probabilidad a priori."""
    
    def __init__(self, nombre, valores, probabilidades):
        """
        Inicializa una distribución a priori.
        
        :parametro nombre: nombre de la distribución
        :parametro valores: lista de valores posibles
        :parametro probabilidades: probabilidad de cada valor
        """
        # Validación básica: la distribución debe estar normalizada
        assert abs(sum(probabilidades) - 1.0) < 0.001, "Las probabilidades deben sumar 1"
        self.nombre = nombre
        self.valores = valores
        self.probabilidades = np.array(probabilidades)
    
    def obtener_probabilidad(self, valor):
        """Obtiene la probabilidad a priori de un valor."""
        try:
            idx = self.valores.index(valor)
            return self.probabilidades[idx]
        except ValueError:
            return 0.0
    
    def mostrar(self):
        """Muestra la distribución."""
        print(f"\nDistribución A Priori: {self.nombre}")
        print("Valor | Probabilidad")
        print("-" * 25)
        for val, prob in zip(self.valores, self.probabilidades):
            print(f"{val:5} | {prob:.4f}")

def crear_distribucion_uniforme(valores):
    """
    Crea una distribución a priori uniforme (principio de indiferencia).
    Asigna la misma probabilidad a todos los valores cuando no hay información previa.
    
    :parametro valores: lista de valores posibles
    :return: DistribucionAPriori
    """
    n = len(valores)
    # Principio de indiferencia: repartir 1 por igual entre n resultados
    probabilidades = [1.0/n] * n
    return DistribucionAPriori("Uniforme (No Informativa)", valores, probabilidades)

def crear_distribucion_informativa(valores, pesos):
    """
    Crea una distribución a priori informativa basada en conocimiento previo.
    
    :parametro valores: lista de valores posibles
    :parametro pesos: pesos relativos (se normalizarán)
    :return: DistribucionAPriori
    """
    total = sum(pesos)
    # Normalizar pesos relativos para obtener probabilidades (proporcionales a 'pesos')
    probabilidades = [p/total for p in pesos]
    return DistribucionAPriori("Informativa (Con Conocimiento Previo)", valores, probabilidades)

# ========== ACTUALIZACIÓN BAYESIANA ==========

def actualizacion_bayesiana(prior, evidencia, verosimilitud):
    """
    Actualiza probabilidades usando el teorema de Bayes.
    P(H|E) = P(E|H) × P(H) / P(E)

    :parametro prior: DistribucionAPriori
    :parametro evidencia: evidencia observada
    :parametro verosimilitud: dict {valor: P(evidencia|valor)}
    :return: DistribucionAPriori posterior
    """
    # Calcular la constante de normalización (evidencia):
    # P(E) = Σ_i P(E|Hi) × P(Hi)
    prob_evidencia = 0.0
    for i, valor in enumerate(prior.valores):
        P_E_dado_H = verosimilitud.get(valor, 0.0)
        P_H = prior.probabilidades[i]
        prob_evidencia += P_E_dado_H * P_H
    
    # Calcular posterior para cada hipótesis
    posterior_probs = []
    for i, valor in enumerate(prior.valores):
        P_E_dado_H = verosimilitud.get(valor, 0.0)
        P_H = prior.probabilidades[i]
        
        # Teorema de Bayes (para cada hipótesis Hi):
        # P(Hi|E) = [P(E|Hi) × P(Hi)] / P(E)
        if prob_evidencia > 0:
            P_H_dado_E = (P_E_dado_H * P_H) / prob_evidencia
        else:
            P_H_dado_E = 0.0
        
        posterior_probs.append(P_H_dado_E)
    
    return DistribucionAPriori("A Posteriori (Después de Evidencia)", prior.valores, posterior_probs)

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con ejemplos de probabilidades a priori."""
    print("\n" + "="*70)
    print("MODO DEMO: Probabilidad A Priori e Inferencia Bayesiana")
    print("="*70)
    
    # ========== EJEMPLO 1: Principio de Indiferencia ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Principio de Indiferencia (Distribución Uniforme)")
    print("="*70)
    
    print("\n--- Escenario: Lanzamiento de un dado justo ---")
    print("Sin información previa, asignamos probabilidad uniforme a cada cara.")
    
    valores_dado = list(range(1, 7))
    prior_uniforme = crear_distribucion_uniforme(valores_dado)
    prior_uniforme.mostrar()
    
    print("\n>>> OBSERVACIÓN:")
    print("    Cuando no tenemos información previa, el principio de indiferencia")
    print("    sugiere asignar probabilidades iguales a todas las posibilidades.")
    
    # ========== EJEMPLO 2: Prior Informativo vs No Informativo ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Comparación de Priors Informativos vs No Informativos")
    print("="*70)
    
    print("\n--- Escenario: Pronóstico del clima ---")
    print("Posibles estados: Soleado, Nublado, Lluvioso")
    
    estados_clima = ['Soleado', 'Nublado', 'Lluvioso']
    
    # Prior no informativo (sin conocimiento previo)
    prior_no_info = crear_distribucion_uniforme(estados_clima)
    
    # Prior informativo (conocemos que en esta región llueve poco)
    pesos_info = [0.6, 0.3, 0.1]  # Favorece días soleados
    prior_info = crear_distribucion_informativa(estados_clima, pesos_info)
    
    print("\n--- Prior No Informativo ---")
    prior_no_info.mostrar()
    
    print("\n--- Prior Informativo (región soleada) ---")
    prior_info.mostrar()
    
    print("\n>>> OBSERVACIÓN:")
    print("    El prior informativo incorpora conocimiento del dominio.")
    print("    En regiones desérticas, asignamos mayor probabilidad a 'Soleado'.")
    
    # ========== EJEMPLO 3: Actualización Bayesiana ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: De A Priori a A Posteriori con Evidencia")
    print("="*70)
    
    print("\n--- Escenario: Diagnóstico médico ---")
    print("Paciente con síntoma: Fiebre alta")
    print("Posibles enfermedades: Gripe, Resfriado, COVID")
    
    enfermedades = ['Gripe', 'Resfriado', 'COVID']
    
    # Prior basado en prevalencia (cuán comunes son)
    prior_prevalencia = crear_distribucion_informativa(
        enfermedades,
        [0.15, 0.60, 0.25]  # Resfriado más común
    )
    
    print("\n--- Probabilidad A Priori (antes de observar síntomas) ---")
    prior_prevalencia.mostrar()
    
    # Verosimilitud: P(Fiebre Alta | Enfermedad)
    verosimilitud_fiebre = {
        'Gripe': 0.8,      # La gripe causa fiebre alta con frecuencia
        'Resfriado': 0.2,  # El resfriado rara vez causa fiebre alta
        'COVID': 0.7       # COVID puede causar fiebre alta
    }
    
    print("\n--- Verosimilitud: P(Fiebre Alta | Enfermedad) ---")
    for enf, prob in verosimilitud_fiebre.items():
        print(f"P(Fiebre Alta | {enf}) = {prob:.2f}")
    
    # Actualizar con evidencia
    posterior = actualizacion_bayesiana(prior_prevalencia, "Fiebre Alta", verosimilitud_fiebre)
    
    print("\n--- Probabilidad A Posteriori (después de observar fiebre) ---")
    posterior.mostrar()
    
    print("\n>>> ANÁLISIS:")
    print("    Cambios en probabilidades:")
    for i, enf in enumerate(enfermedades):
        cambio = posterior.probabilidades[i] - prior_prevalencia.probabilidades[i]
        signo = "↑" if cambio > 0 else "↓"
        print(f"    {enf}: {prior_prevalencia.probabilidades[i]:.3f} → {posterior.probabilidades[i]:.3f} ({signo} {abs(cambio):.3f})")
    
    print("\n    La probabilidad de Gripe aumentó significativamente")
    print("    porque la fiebre alta es común en gripe (verosimilitud alta).")
    
    # ========== EJEMPLO 4: Impacto del Prior en el Posterior ==========
    print("\n" + "="*70)
    print("EJEMPLO 4: Sensibilidad al Prior")
    print("="*70)
    
    print("\n--- Comparando diferentes priors con la misma evidencia ---")
    
    # Crear diferentes priors
    prior_pesimista = crear_distribucion_informativa(enfermedades, [0.1, 0.1, 0.8])  # Asume COVID
    prior_optimista = crear_distribucion_informativa(enfermedades, [0.1, 0.8, 0.1])  # Asume resfriado
    
    # Actualizar ambos con la misma evidencia
    post_pesimista = actualizacion_bayesiana(prior_pesimista, "Fiebre", verosimilitud_fiebre)
    post_optimista = actualizacion_bayesiana(prior_optimista, "Fiebre", verosimilitud_fiebre)
    
    print("\nPrior Pesimista → Posterior:")
    for i, enf in enumerate(enfermedades):
        print(f"  {enf}: {prior_pesimista.probabilidades[i]:.3f} → {post_pesimista.probabilidades[i]:.3f}")
    
    print("\nPrior Optimista → Posterior:")
    for i, enf in enumerate(enfermedades):
        print(f"  {enf}: {prior_optimista.probabilidades[i]:.3f} → {post_optimista.probabilidades[i]:.3f}")
    
    print("\n>>> CONCLUSIÓN:")
    print("    Diferentes priors llevan a diferentes posteriores,")
    print("    pero con suficiente evidencia, los posteriores convergen.")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Define tus Probabilidades A Priori")
    print("="*70)
    
    print("\n--- Define el espacio de hipótesis ---")
    entrada = input("Ingresa las hipótesis separadas por comas (ej: H1,H2,H3): ").strip()
    
    if not entrada:
        hipotesis = ['H1', 'H2', 'H3']
    else:
        hipotesis = [h.strip() for h in entrada.split(',')]
    
    print(f"\nHipótesis definidas: {hipotesis}")
    
    print("\n--- Selecciona el tipo de prior ---")
    print("1. Uniforme (sin información previa)")
    print("2. Informativo (especificar probabilidades)")
    
    opcion = input("Opción (1 o 2): ").strip()
    
    if opcion == '1':
        prior = crear_distribucion_uniforme(hipotesis)
    else:
        print("\nIngresa las probabilidades para cada hipótesis (separadas por espacios)")
        print(f"Ejemplo para {len(hipotesis)} hipótesis: {' '.join(['0.3'] * len(hipotesis))}")
        
        try:
            probs_str = input("Probabilidades: ").strip().split()
            probs = [float(p) for p in probs_str[:len(hipotesis)]]
            
            # Completar si faltan
            while len(probs) < len(hipotesis):
                probs.append(1.0 / len(hipotesis))
            
            # Normalizar
            suma = sum(probs)
            probs = [p/suma for p in probs]
            
            prior = DistribucionAPriori("Tu Prior Personalizado", hipotesis, probs)
        except:
            print("Error al procesar. Usando prior uniforme.")
            prior = crear_distribucion_uniforme(hipotesis)
    
    print("\n--- Tu distribución a priori ---")
    prior.mostrar()
    
    print(f"\n✓ Prior creado exitosamente")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("PROBABILIDAD A PRIORI E INFERENCIA BAYESIANA")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (ejemplos de priors y actualización bayesiana)")
    print("2. INTERACTIVO (define tus propias probabilidades a priori)")
    
    opcion = input("\nIngresa el número de opción (1 o 2): ").strip()
    
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida. Ejecutando modo DEMO por defecto...")
        modo_demo()
    
    print("\n" + "="*70)
    print("FIN DEL PROGRAMA")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
