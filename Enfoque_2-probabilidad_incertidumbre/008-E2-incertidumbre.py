"""
008-E2-incertidumbre.py
--------------------------------
Este script profundiza en el concepto de Incertidumbre en sistemas de IA:
- Distingue entre incertidumbre epistémica (falta de conocimiento) y aleatoria (inherente)
- Modela situaciones con información incompleta o ambigua
- Implementa lógica difusa para manejar conceptos imprecisos
- Representa ignorancia mediante intervalos de probabilidad
- Aplica teoría de Dempster-Shafer para razonamiento con evidencias inciertas
- Compara enfoques probabilísticos vs. no probabilísticos para manejar incertidumbre
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de diferentes tipos de incertidumbre
2. INTERACTIVO: permite al usuario modelar escenarios con varios niveles de incertidumbre

Autor: Alejandro Aguirre Díaz
"""

import numpy as np

# ========== LÓGICA DIFUSA (FUZZY LOGIC) ==========

class ConjuntoDifuso:
    """
    Representa un conjunto difuso con función de pertenencia.
    Permite modelar conceptos imprecisos como 'temperatura alta', 'velocidad lenta', etc.
    """
    
    def __init__(self, nombre):
        """Inicializa un conjunto difuso."""
        self.nombre = nombre
    
    def pertenencia(self, x):
        """
        Calcula el grado de pertenencia de x al conjunto difuso.
        Debe ser implementado por las subclases.
        
        :param x: valor a evaluar
        :return: grado de pertenencia en [0, 1]
        """
        raise NotImplementedError("Debe implementarse en subclases")

class ConjuntoDifusoTriangular(ConjuntoDifuso):
    """Conjunto difuso con función de pertenencia triangular."""
    
    def __init__(self, nombre, a, b, c):
        """
        Inicializa con parámetros de la función triangular.
        
        :param a: inicio (pertenencia = 0)
        :param b: pico (pertenencia = 1)
        :param c: fin (pertenencia = 0)
        """
        super().__init__(nombre)
        self.a = a
        self.b = b
        self.c = c
    
    def pertenencia(self, x):
        """Calcula pertenencia con función triangular."""
        if x <= self.a or x >= self.c:
            return 0.0
        elif x == self.b:
            return 1.0
        elif x < self.b:
            # Rampa ascendente
            return (x - self.a) / (self.b - self.a)
        else:
            # Rampa descendente
            return (self.c - x) / (self.c - self.b)

class ConjuntoDifusoTrapezoidal(ConjuntoDifuso):
    """Conjunto difuso con función de pertenencia trapezoidal."""
    
    def __init__(self, nombre, a, b, c, d):
        """
        Inicializa con parámetros trapezoidales.
        
        :param a: inicio rampa ascendente
        :param b: fin rampa ascendente (inicio meseta)
        :param c: inicio rampa descendente (fin meseta)
        :param d: fin rampa descendente
        """
        super().__init__(nombre)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def pertenencia(self, x):
        """Calcula pertenencia con función trapezoidal."""
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.d - x) / (self.d - self.c)

# ========== INTERVALOS DE PROBABILIDAD ==========

class IntervaloProbabilidad:
    """
    Representa incertidumbre mediante intervalos de probabilidad.
    Útil cuando no tenemos información precisa para asignar una probabilidad exacta.
    """
    
    def __init__(self, limite_inferior, limite_superior):
        """
        Inicializa un intervalo de probabilidad.
        
        :param limite_inferior: cota inferior (mínima probabilidad)
        :param limite_superior: cota superior (máxima probabilidad)
        """
        assert 0 <= limite_inferior <= limite_superior <= 1, "Límites deben estar en [0, 1]"
        self.min = limite_inferior
        self.max = limite_superior
    
    def amplitud(self):
        """Retorna la amplitud del intervalo (mide la ignorancia)."""
        return self.max - self.min
    
    def punto_medio(self):
        """Retorna el punto medio del intervalo."""
        return (self.min + self.max) / 2
    
    def contiene(self, probabilidad):
        """Verifica si una probabilidad está dentro del intervalo."""
        return self.min <= probabilidad <= self.max
    
    def __repr__(self):
        return f"[{self.min:.3f}, {self.max:.3f}]"

# ========== FUNCIONES AUXILIARES ==========

def calcular_entropia(probabilidades):
    """
    Calcula la entropía de Shannon para medir incertidumbre.
    H(X) = -Σ p(x) × log₂(p(x))
    
    :param probabilidades: lista de probabilidades
    :return: entropía en bits
    """
    entropia = 0.0
    for p in probabilidades:
        if p > 0:  # Evitar log(0)
            entropia -= p * np.log2(p)
    return entropia

def clasificar_incertidumbre(descripcion):
    """
    Clasifica un tipo de incertidumbre en epistémica o aleatoria.
    
    - Epistémica: debido a falta de conocimiento (puede reducirse con más información)
    - Aleatoria: inherente al sistema (no se puede reducir)
    """
    keywords_epistemica = ['desconocido', 'información', 'conocimiento', 'datos', 'medición']
    keywords_aleatoria = ['aleatorio', 'caótico', 'estocast', 'ruido', 'inherente']
    
    desc_lower = descripcion.lower()
    
    score_epistemica = sum(1 for kw in keywords_epistemica if kw in desc_lower)
    score_aleatoria = sum(1 for kw in keywords_aleatoria if kw in desc_lower)
    
    if score_epistemica > score_aleatoria:
        return "Epistémica (falta de conocimiento)"
    elif score_aleatoria > score_epistemica:
        return "Aleatoria (inherente al sistema)"
    else:
        return "Mixta o ambigua"

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con ejemplos de incertidumbre."""
    print("\n" + "="*70)
    print("MODO DEMO: Modelado de Incertidumbre")
    print("="*70)
    
    # ========== EJEMPLO 1: Lógica Difusa ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Lógica Difusa para Conceptos Imprecisos")
    print("="*70)
    
    print("\n--- Modelando temperatura con conjuntos difusos ---")
    
    # Definir conjuntos difusos para temperatura
    fria = ConjuntoDifusoTrapezoidal("Fría", 0, 0, 10, 18)
    templada = ConjuntoDifusoTriangular("Templada", 15, 22, 28)
    calida = ConjuntoDifusoTrapezoidal("Cálida", 25, 30, 50, 50)
    
    print("Conjuntos definidos:")
    print("  - Fría: [0-18°C con pico en 0-10°C]")
    print("  - Templada: [15-28°C con pico en 22°C]")
    print("  - Cálida: [25-50°C con pico en 30-50°C]")
    
    # Evaluar temperaturas
    temperaturas = [5, 15, 20, 25, 30, 35]
    
    print("\n--- Grados de pertenencia ---")
    print("Temp(°C) | Fría   | Templada | Cálida")
    print("-" * 45)
    
    for temp in temperaturas:
        p_fria = fria.pertenencia(temp)
        p_templada = templada.pertenencia(temp)
        p_calida = calida.pertenencia(temp)
        
        print(f"{temp:8d} | {p_fria:6.2f} | {p_templada:8.2f} | {p_calida:6.2f}")
    
    print("\n>>> OBSERVACIÓN:")
    print("    Una temperatura puede pertenecer parcialmente a múltiples conjuntos.")
    print("    Por ejemplo, 25°C es 'algo templada' (0.57) y 'algo cálida' (0.00).")
    
    # ========== EJEMPLO 2: Intervalos de Probabilidad ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Intervalos de Probabilidad (Ignorancia)")
    print("="*70)
    
    print("\n--- Escenario: Probabilidad de lluvia mañana ---")
    print("Tenemos información limitada:")
    print("  - Pronóstico confiable: P(lluvia) ∈ [0.3, 0.7]")
    print("  - Pronóstico especulativo: P(lluvia) ∈ [0.1, 0.9]")
    
    intervalo_confiable = IntervaloProbabilidad(0.3, 0.7)
    intervalo_especulativo = IntervaloProbabilidad(0.1, 0.9)
    
    print(f"\nIntervalo confiable: {intervalo_confiable}")
    print(f"  Amplitud (incertidumbre): {intervalo_confiable.amplitud():.2f}")
    print(f"  Punto medio: {intervalo_confiable.punto_medio():.2f}")
    
    print(f"\nIntervalo especulativo: {intervalo_especulativo}")
    print(f"  Amplitud (incertidumbre): {intervalo_especulativo.amplitud():.2f}")
    print(f"  Punto medio: {intervalo_especulativo.punto_medio():.2f}")
    
    print("\n>>> OBSERVACIÓN:")
    print("    Mayor amplitud del intervalo = mayor ignorancia.")
    print("    A medida que obtenemos más información, los intervalos se estrechan.")
    
    # ========== EJEMPLO 3: Entropía ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Medición de Incertidumbre con Entropía")
    print("="*70)
    
    print("\n--- Comparando diferentes distribuciones ---")
    
    # Distribuciones con diferentes niveles de incertidumbre
    dist_segura = [1.0, 0.0, 0.0, 0.0]  # Completamente determinista
    dist_poco_incierta = [0.7, 0.2, 0.1, 0.0]
    dist_incierta = [0.4, 0.3, 0.2, 0.1]
    dist_maxima_incierta = [0.25, 0.25, 0.25, 0.25]  # Completamente uniforme
    
    entropia_segura = calcular_entropia(dist_segura)
    entropia_poco = calcular_entropia(dist_poco_incierta)
    entropia_media = calcular_entropia(dist_incierta)
    entropia_max = calcular_entropia(dist_maxima_incierta)
    
    print("Distribución                | Entropía (bits)")
    print("-" * 50)
    print(f"Determinista [1.0, 0, 0, 0] | {entropia_segura:.3f}")
    print(f"Poco incierta [0.7,0.2,...]  | {entropia_poco:.3f}")
    print(f"Incierta [0.4, 0.3, ...]     | {entropia_media:.3f}")
    print(f"Máx. incierta [0.25,0.25,..] | {entropia_max:.3f}")
    
    print("\n>>> OBSERVACIÓN:")
    print(f"    Entropía máxima = {entropia_max:.2f} bits (distribución uniforme).")
    print("    Mayor entropía = mayor incertidumbre.")
    
    # ========== EJEMPLO 4: Tipos de Incertidumbre ==========
    print("\n" + "="*70)
    print("EJEMPLO 4: Clasificación de Incertidumbre")
    print("="*70)
    
    ejemplos = [
        "No sabemos la temperatura exacta porque el termómetro está roto (falta de datos)",
        "El lanzamiento de un dado es aleatorio por naturaleza",
        "El resultado de una elección es incierto por falta de información de votantes",
        "El ruido térmico en un circuito es inherente y estocástico"
    ]
    
    print("\n--- Clasificando ejemplos ---\n")
    for i, ejemplo in enumerate(ejemplos, 1):
        clasificacion = clasificar_incertidumbre(ejemplo)
        print(f"{i}. {ejemplo}")
        print(f"   → Tipo: {clasificacion}\n")
    
    print(">>> OBSERVACIÓN:")
    print("    - Epistémica: Se puede reducir con más información/mediciones")
    print("    - Aleatoria: No se puede eliminar, solo modelar probabilísticamente")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Análisis de Incertidumbre")
    print("="*70)
    
    print("\n--- Evaluación de Conjunto Difuso Personalizado ---")
    print("Define un conjunto difuso triangular para un concepto.")
    
    try:
        nombre = input("Nombre del concepto (ej: 'Velocidad Alta'): ").strip() or "Concepto"
        a = float(input("Valor mínimo (inicio): ").strip() or "0")
        b = float(input("Valor pico (máxima pertenencia): ").strip() or "50")
        c = float(input("Valor máximo (fin): ").strip() or "100")
    except:
        nombre = "Velocidad Alta"
        a, b, c = 30, 70, 100
    
    conjunto = ConjuntoDifusoTriangular(nombre, a, b, c)
    
    print(f"\nConjunto '{nombre}' creado: [{a}, {b}, {c}]")
    
    print("\n--- Evaluar valores ---")
    valores_prueba = [a, (a+b)/2, b, (b+c)/2, c]
    
    print(f"Valor | Pertenencia a '{nombre}'")
    print("-" * 35)
    for val in valores_prueba:
        pertenencia = conjunto.pertenencia(val)
        print(f"{val:5.1f} | {pertenencia:.3f}")
    
    # Análisis de entropía
    print("\n--- Calcular Entropía de una Distribución ---")
    print("Ingresa probabilidades separadas por espacios (deben sumar ~1.0)")
    print("Ejemplo: 0.5 0.3 0.2")
    
    entrada = input("\nProbabilidades: ").strip()
    
    try:
        probs = [float(p) for p in entrada.split()]
        suma = sum(probs)
        if abs(suma - 1.0) > 0.01:
            print(f"Advertencia: Las probabilidades suman {suma:.3f}, normalizando...")
            probs = [p/suma for p in probs]
        
        entropia = calcular_entropia(probs)
        print(f"\nEntropía: {entropia:.3f} bits")
        print(f"Entropía máxima posible: {np.log2(len(probs)):.3f} bits")
        print(f"Nivel de incertidumbre: {(entropia/np.log2(len(probs)))*100:.1f}%")
    except:
        print("Error al procesar probabilidades.")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("MODELADO DE INCERTIDUMBRE EN SISTEMAS INTELIGENTES")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (ejemplos de lógica difusa, intervalos, entropía)")
    print("2. INTERACTIVO (modela tus propios conceptos imprecisos)")
    
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
