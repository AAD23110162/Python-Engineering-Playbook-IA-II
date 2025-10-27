"""
031-E2-naive_bayes.py
--------------------------------
Este script implementa el clasificador Naïve Bayes:
- Asume independencia condicional de las características dado la clase.
- Soporta variantes multinomial/bernoulli/gaussiana a nivel conceptual.
- Calcula probabilidades a posteriori y realiza clasificación.
- Discute calibración y manejo de rareza con suavizado de Laplace.

El programa puede ejecutarse en dos modos:
1. DEMO: clasificación en datasets de juguete con desbalance.
2. INTERACTIVO: permite ingresar datos tabulares y evaluar predicciones.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Dict, Tuple
from collections import defaultdict

# ============================================================================
# Clasificador Naive Bayes Multinomial
# ============================================================================

class NaiveBayesMultinomial:
    """
    Clasificador Naïve Bayes para características discretas (multinomiales).
    Usa suavizado de Laplace para evitar probabilidades cero.
    """
    
    def __init__(self, alpha: float = 1.0):
        # Inicializa el parámetro de suavizado y estructuras para conteos
        """
        Inicializa el clasificador.
        
        Args:
            alpha: Parámetro de suavizado de Laplace (default=1.0)
        """
        # Parámetro de suavizado de Laplace
        self.alpha = alpha
        
        # Conteos aprendidos durante entrenamiento
        self.conteo_clases = defaultdict(int)           # N(clase)
        self.conteo_caracteristicas = defaultdict(lambda: defaultdict(int))  # N(caracteristica, valor, clase)
        self.vocabulario = set()  # Conjunto de todos los valores posibles
        
        # Probabilidades prior de las clases
        self.prob_clases = {}
        
        # Total de ejemplos
        self.n_ejemplos = 0
    
    def entrenar(self, X: List[List[str]], y: List[str]):
        # Entrenamiento: cuenta clases y características, calcula priors
        """
        Entrena el clasificador con datos de entrenamiento.
        
        Args:
            X: Lista de ejemplos, cada uno es una lista de valores de características
            y: Lista de etiquetas de clase
        """
        self.n_ejemplos = len(y)
        
        # Contar clases
        for clase in y:
            self.conteo_clases[clase] += 1
        
        # Contar características por clase
        for ejemplo, clase in zip(X, y):
            for i, valor in enumerate(ejemplo):
                # Registrar característica i con valor en la clase
                self.conteo_caracteristicas[(i, valor)][clase] += 1
                self.vocabulario.add((i, valor))
        
        # Calcular probabilidades prior P(clase)
        for clase, conteo in self.conteo_clases.items():
            self.prob_clases[clase] = conteo / self.n_ejemplos
    
    def predecir(self, x: List[str]) -> Tuple[str, Dict[str, float]]:
        # Predicción: calcula log-probabilidades y normaliza para obtener posterior
        """
        Predice la clase de un nuevo ejemplo.
        
        Args:
            x: Ejemplo a clasificar (lista de valores de características)
            
        Returns:
            (clase_predicha, probabilidades_posteriores)
        """
        # Calcular log-probabilidad posterior para cada clase
        log_probs = {}
        
        for clase in self.conteo_clases.keys():
            # Log prior: log P(clase)
            log_prob = math.log(self.prob_clases[clase])
            
            # Sumar log-likelihoods: log P(x_i | clase) para cada característica
            for i, valor in enumerate(x):
                # Contar cuántas veces apareció (i, valor) en la clase
                conteo = self.conteo_caracteristicas[(i, valor)][clase]
                
                # Total de ejemplos en la clase
                total_clase = self.conteo_clases[clase]
                
                # Tamaño del vocabulario para esta característica
                # (simplificación: usamos tamaño del vocabulario total)
                vocab_size = len(self.vocabulario)
                
                # Probabilidad con suavizado de Laplace:
                # P(x_i | clase) = (conteo + alpha) / (total_clase + alpha * vocab_size)
                prob = (conteo + self.alpha) / (total_clase + self.alpha * vocab_size)
                
                log_prob += math.log(prob)
            
            log_probs[clase] = log_prob
        
        # Normalizar para obtener probabilidades posteriores
        max_log_prob = max(log_probs.values())
        probs = {}
        suma = 0.0
        
        for clase, log_prob in log_probs.items():
            # Restar max para estabilidad numérica
            prob = math.exp(log_prob - max_log_prob)
            probs[clase] = prob
            suma += prob
        
        # Normalizar
        for clase in probs:
            probs[clase] /= suma
        
        # Seleccionar clase con mayor probabilidad
        clase_predicha = max(probs, key=probs.get)
        
        return clase_predicha, probs

# ============================================================================
# Clasificador Naive Bayes Gaussiano
# ============================================================================

class NaiveBayesGaussiano:
    """
    Clasificador Naïve Bayes para características continuas.
    Asume que cada característica sigue una distribución Gaussiana.
    """
    
    def __init__(self):
        # Inicializa estructuras para medias y varianzas por clase
        """Inicializa el clasificador."""
        # Estadísticas por clase y característica
        self.medias = defaultdict(dict)      # media[clase][i]
        self.varianzas = defaultdict(dict)   # varianza[clase][i]
        
        # Probabilidades prior
        self.prob_clases = {}
        self.clases = []
    
    def entrenar(self, X: List[List[float]], y: List[str]):
        # Entrenamiento: agrupa por clase, calcula medias y varianzas
        """
        Entrena el clasificador con datos de entrenamiento.
        
        Args:
            X: Lista de ejemplos, cada uno es una lista de valores numéricos
            y: Lista de etiquetas de clase
        """
        # Agrupar ejemplos por clase
        clases_dict = defaultdict(list)
        for ejemplo, clase in zip(X, y):
            clases_dict[clase].append(ejemplo)
        
        self.clases = list(clases_dict.keys())
        n_caracteristicas = len(X[0])
        
        # Calcular estadísticas para cada clase
        for clase, ejemplos in clases_dict.items():
            n_ejemplos = len(ejemplos)
            
            # Probabilidad prior P(clase)
            self.prob_clases[clase] = n_ejemplos / len(y)
            
            # Calcular media y varianza para cada característica
            for i in range(n_caracteristicas):
                valores = [ej[i] for ej in ejemplos]
                
                # Media
                media = sum(valores) / n_ejemplos
                self.medias[clase][i] = media
                
                # Varianza (con corrección de Bessel)
                varianza = sum((x - media) ** 2 for x in valores) / max(n_ejemplos - 1, 1)
                # Evitar varianza cero
                self.varianzas[clase][i] = max(varianza, 1e-6)
    
    def _pdf_gaussiana(self, x: float, media: float, varianza: float) -> float:
        # Calcula la densidad de la normal para una característica
        """Calcula la densidad de probabilidad Gaussiana."""
        # Evitar log de números muy pequeños
        exp_term = -0.5 * ((x - media) ** 2) / varianza
        coef = 1.0 / math.sqrt(2 * math.pi * varianza)
        return coef * math.exp(exp_term)
    
    def predecir(self, x: List[float]) -> Tuple[str, Dict[str, float]]:
        # Predicción: suma log-likelihoods y normaliza para obtener posterior
        """
        Predice la clase de un nuevo ejemplo.
        
        Args:
            x: Ejemplo a clasificar (lista de valores numéricos)
            
        Returns:
            (clase_predicha, probabilidades_posteriores)
        """
        log_probs = {}
        
        for clase in self.clases:
            # Log prior
            log_prob = math.log(self.prob_clases[clase])
            
            # Sumar log-likelihoods para cada característica
            for i, valor in enumerate(x):
                media = self.medias[clase][i]
                varianza = self.varianzas[clase][i]
                
                # Log P(x_i | clase)
                pdf = self._pdf_gaussiana(valor, media, varianza)
                log_prob += math.log(pdf + 1e-10)  # Evitar log(0)
            
            log_probs[clase] = log_prob
        
        # Normalizar
        max_log_prob = max(log_probs.values())
        probs = {}
        suma = 0.0
        
        for clase, log_prob in log_probs.items():
            prob = math.exp(log_prob - max_log_prob)
            probs[clase] = prob
            suma += prob
        
        for clase in probs:
            probs[clase] /= suma
        
        clase_predicha = max(probs, key=probs.get)
        
        return clase_predicha, probs

# ============================================================================
# Funciones de Evaluación
# ============================================================================

def calcular_exactitud(y_real: List[str], y_pred: List[str]) -> float:
    """Calcula la exactitud de las predicciones."""
    # Calcula la proporción de aciertos
    correctos = sum(1 for yr, yp in zip(y_real, y_pred) if yr == yp)
    return correctos / len(y_real)

def matriz_confusion(y_real: List[str], y_pred: List[str]) -> Dict:
    """Calcula la matriz de confusión."""
    # Construye la matriz de confusión para evaluar el desempeño
    clases = sorted(set(y_real + y_pred))
    matriz = {c: {c2: 0 for c2 in clases} for c in clases}
    
    for yr, yp in zip(y_real, y_pred):
        matriz[yr][yp] += 1
    
    return matriz

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra el clasificador Naïve Bayes con datasets de juguete."""
    print("MODO DEMO: Clasificador Naïve Bayes\n")
    
    # ========================================
    # Ejemplo 1: Clasificación de documentos (Multinomial)
    # ========================================
    print("=" * 60)
    print("Ejemplo 1: Clasificación de texto (Multinomial)")
    print("=" * 60)
    
    # Dataset: documentos representados por palabras (multinomial)
    # Clase: "spam" o "ham" (legítimo)
    X_train = [
        ["gratis", "dinero", "gana"],
        ["oferta", "gratis", "ahora"],
        ["reunion", "proyecto", "trabajo"],
        ["informe", "proyecto", "revision"],
        ["gratis", "oferta", "click"],
        ["trabajo", "equipo", "reunion"],
        ["gana", "premio", "gratis"],
        ["revision", "documento", "proyecto"]
    ]
    
    y_train = ["spam", "spam", "ham", "ham", "spam", "ham", "spam", "ham"]
    
    print("Datos de entrenamiento:")
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        print(f"  {i+1}. {x} -> {y}")
    print()
    
    # Entrenar clasificador
    # Entrenamiento del clasificador multinomial
    clf_multi = NaiveBayesMultinomial(alpha=1.0)
    clf_multi.entrenar(X_train, y_train)
    
    print(f"Probabilidades prior:")
    for clase, prob in clf_multi.prob_clases.items():
        print(f"  P({clase}) = {prob:.4f}")
    print()
    
    # Hacer predicciones
    # Pruebas de predicción con nuevos documentos
    X_test = [
        ["gratis", "premio", "ahora"],
        ["reunion", "informe", "revision"],
        ["oferta", "click", "gana"]
    ]
    
    print("Predicciones:")
    for x in X_test:
        clase_pred, probs = clf_multi.predecir(x)
        print(f"  Documento: {x}")
        print(f"    Clase predicha: {clase_pred}")
        print(f"    Probabilidades: {', '.join(f'{c}: {p:.4f}' for c, p in probs.items())}")
        print()
    
    # ========================================
    # Ejemplo 2: Clasificación con características continuas (Gaussiano)
    # ========================================
    print("=" * 60)
    print("Ejemplo 2: Clasificación con características continuas (Gaussiano)")
    print("=" * 60)
    
    # Dataset: flores con características [longitud_pétalo, ancho_pétalo] (gaussiano)
    random.seed(42)
    
    # Clase "setosa": pétalos pequeños
    X_setosa = [[random.gauss(1.5, 0.2), random.gauss(0.3, 0.1)] for _ in range(15)]
    y_setosa = ["setosa"] * 15
    
    # Clase "versicolor": pétalos medianos
    X_versicolor = [[random.gauss(4.0, 0.5), random.gauss(1.3, 0.2)] for _ in range(15)]
    y_versicolor = ["versicolor"] * 15
    
    # Combinar datos
    X_train_gauss = X_setosa + X_versicolor
    y_train_gauss = y_setosa + y_versicolor
    
    # Entrenar clasificador
    # Entrenamiento del clasificador gaussiano
    clf_gauss = NaiveBayesGaussiano()
    clf_gauss.entrenar(X_train_gauss, y_train_gauss)
    
    print(f"Probabilidades prior:")
    for clase, prob in clf_gauss.prob_clases.items():
        print(f"  P({clase}) = {prob:.4f}")
    print()
    
    print("Estadísticas aprendidas:")
    for clase in clf_gauss.clases:
        print(f"  Clase '{clase}':")
        for i in range(2):
            media = clf_gauss.medias[clase][i]
            varianza = clf_gauss.varianzas[clase][i]
            print(f"    Característica {i}: μ={media:.2f}, σ²={varianza:.2f}")
    print()
    
    # Hacer predicciones
    # Pruebas de predicción con ejemplos continuos
    X_test_gauss = [
        [1.4, 0.2],   # Debería ser setosa
        [4.5, 1.5],   # Debería ser versicolor
        [2.0, 0.8]    # Intermedio
    ]
    
    print("Predicciones:")
    for x in X_test_gauss:
        clase_pred, probs = clf_gauss.predecir(x)
        print(f"  Ejemplo: {[f'{v:.2f}' for v in x]}")
        print(f"    Clase predicha: {clase_pred}")
        print(f"    Probabilidades: {', '.join(f'{c}: {p:.4f}' for c, p in probs.items())}")
        print()
    
    # Evaluar en conjunto de entrenamiento
    # Evaluación de exactitud en el conjunto de entrenamiento
    y_pred = [clf_gauss.predecir(x)[0] for x in X_train_gauss]
    exactitud = calcular_exactitud(y_train_gauss, y_pred)
    print(f"Exactitud en entrenamiento: {exactitud:.2%}")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario ingresar datos y evaluar el clasificador."""
    print("MODO INTERACTIVO: Naïve Bayes\n")
    
    print("Seleccione el tipo de clasificador:")
    print("1. Multinomial (características discretas)")
    print("2. Gaussiano (características continuas)")
    
    tipo = input("Ingrese el número (default=1): ").strip()
    
    # Clasificador Gaussiano
    if tipo == "2":
        print("\nClasificador Gaussiano")
        print("Ingrese datos de entrenamiento en formato:")
        print("  clase valor1 valor2 ...")
        print("Escriba 'fin' cuando termine.\n")
        
        X_train = []
        y_train = []
        
        # Ingreso de datos de entrenamiento por el usuario
        while True:
            entrada = input("> ").strip()
            if entrada.lower() == "fin":
                break
            
            partes = entrada.split()
            if len(partes) >= 2:
                clase = partes[0]
                try:
                    valores = [float(v) for v in partes[1:]]
                except ValueError:
                    print("Valores inválidos: asegúrese de ingresar números para las características.")
                    continue
                y_train.append(clase)
                X_train.append(valores)
        
        if not X_train:
            print("No se ingresaron datos. Usando ejemplo por defecto.")
            return
        
        # Entrenamiento del modelo gaussiano
        clf = NaiveBayesGaussiano()
        clf.entrenar(X_train, y_train)
        
        print(f"\nModelo entrenado con {len(X_train)} ejemplos.")
        print(f"Clases: {clf.clases}")
        
        # Predecir
        print("\nIngrese un ejemplo para clasificar (valores separados por espacios):")
        entrada_test = input("> ").strip()
        
        if entrada_test:
            try:
                valores_test = [float(v) for v in entrada_test.split()]
            except ValueError:
                print("Entrada inválida: asegúrese de ingresar números separados por espacios.")
                return
            clase_pred, probs = clf.predecir(valores_test)
            
            print(f"\nClase predicha: {clase_pred}")
            print("Probabilidades posteriores:")
            for clase, prob in probs.items():
                print(f"  P({clase} | x) = {prob:.4f}")
    
    # Clasificador Multinomial (por defecto)
    else:
        print("\nClasificador Multinomial")
        print("Ingrese datos de entrenamiento en formato:")
        print("  clase caracteristica1 caracteristica2 ...")
        print("Escriba 'fin' cuando termine.\n")
        
        X_train = []
        y_train = []
        
        # Ingreso de datos de entrenamiento por el usuario
        while True:
            entrada = input("> ").strip()
            if entrada.lower() == "fin":
                break
            
            partes = entrada.split()
            if len(partes) >= 2:
                clase = partes[0]
                caracteristicas = partes[1:]
                y_train.append(clase)
                X_train.append(caracteristicas)
        
        if not X_train:
            print("No se ingresaron datos. Usando ejemplo por defecto.")
            return
        
        # Entrenamiento del modelo multinomial
        try:
            alpha = float(input("\nIngrese parámetro de suavizado alpha (default=1.0): ") or "1.0")
        except ValueError:
            print("Alpha inválido. Usando alpha=1.0")
            alpha = 1.0
        
        clf = NaiveBayesMultinomial(alpha=alpha)
        clf.entrenar(X_train, y_train)
        
        print(f"\nModelo entrenado con {len(X_train)} ejemplos.")
        
        # Predecir
        print("\nIngrese un ejemplo para clasificar (características separadas por espacios):")
        entrada_test = input("> ").strip()
        
        if entrada_test:
            caracteristicas_test = entrada_test.split()
            clase_pred, probs = clf.predecir(caracteristicas_test)
            
            print(f"\nClase predicha: {clase_pred}")
            print("Probabilidades posteriores:")
            for clase, prob in probs.items():
                print(f"  P({clase} | x) = {prob:.4f}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("031-E2: Clasificador Naïve Bayes")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Ejemplos de clasificación")
    print("2. INTERACTIVO: Ingresar datos propios")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
