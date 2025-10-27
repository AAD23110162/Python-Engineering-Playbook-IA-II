"""
035-E2-knn_kmedias_y_clustering.py
--------------------------------
Este script combina k-NN, k-Medias y nociones de Clustering:
- k-NN para clasificación basada en vecinos más cercanos.
- k-Medias para particionado en k clusters.
- Relación entre métricas de distancia y desempeño.
- Discute normalización de variables y elección de k.

El programa puede ejecutarse en dos modos:
1. DEMO: ejemplos con datos 2D y decisión de frontera.
2. INTERACTIVO: permite cargar datos, elegir k y métricas.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple, Dict
from collections import Counter, defaultdict

# ============================================================================
# Clasificador k-NN
# ============================================================================

class KNN:
    """
    Clasificador k-Nearest Neighbors (k-NN).
    """
    
    def __init__(self, k: int = 3, metrica: str = 'euclidiana'):
        """
        Inicializa el clasificador k-NN.
        
        Args:
            k: Número de vecinos más cercanos
            metrica: Métrica de distancia ('euclidiana' o 'manhattan')
        """
        self.k = k
        self.metrica = metrica
        
        # Datos de entrenamiento
        self.X_train = []
        self.y_train = []
    
    def _distancia(self, p1: List[float], p2: List[float]) -> float:
        """Calcula la distancia entre dos puntos según la métrica elegida."""
        if self.metrica == 'manhattan':
            # Distancia Manhattan: suma de diferencias absolutas
            return sum(abs(a - b) for a, b in zip(p1, p2))
        else:
            # Distancia Euclidiana: raíz cuadrada de suma de cuadrados
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def entrenar(self, X: List[List[float]], y: List[str]):
        """
        "Entrena" el clasificador: solo almacena los datos, no hay ajuste de parámetros.
        k-NN es un método basado en instancias (lazy learning).
        """
        self.X_train = [x[:] for x in X]
        self.y_train = y[:]
    
    def predecir(self, x: List[float]) -> Tuple[str, Dict[str, float]]:
        """
        Predice la clase de un nuevo punto usando voto mayoritario entre los k vecinos más cercanos.
        Devuelve la clase predicha y las probabilidades estimadas por frecuencia.
        """
        # Calcular distancias a todos los puntos de entrenamiento
        distancias = []
        for x_train, y_train in zip(self.X_train, self.y_train):
            dist = self._distancia(x, x_train)
            distancias.append((dist, y_train))
        # Ordenar por distancia y tomar los k más cercanos
        distancias.sort()
        k_vecinos = distancias[:self.k]
        # Votar por mayoría
        etiquetas_vecinos = [etiqueta for _, etiqueta in k_vecinos]
        conteo = Counter(etiquetas_vecinos)
        # Calcular probabilidades (proporción de cada clase entre los vecinos)
        total = len(etiquetas_vecinos)
        probabilidades = {clase: conteo.get(clase, 0) / total 
                         for clase in set(self.y_train)}
        # Clase con más votos
        clase_predicha = conteo.most_common(1)[0][0]
        return clase_predicha, probabilidades
    
    def predecir_batch(self, X: List[List[float]]) -> List[str]:
        """Predice las clases de múltiples puntos."""
        return [self.predecir(x)[0] for x in X]

# ============================================================================
# Algoritmo K-Medias (re-implementado para completitud)
# ============================================================================

class KMediasCluster:
    """
    Algoritmo de clustering K-Medias.
    """
    
    def __init__(self, k: int, max_iter: int = 100, metrica: str = 'euclidiana'):
        """
        Inicializa K-Medias.
        
        Args:
            k: Número de clusters
            max_iter: Número máximo de iteraciones
            metrica: Métrica de distancia
        """
        self.k = k
        self.max_iter = max_iter
        self.metrica = metrica
        
        self.centroides = []
        self.asignaciones = []
        self.inercia = 0.0
    
    def _distancia(self, p1: List[float], p2: List[float]) -> float:
        """Calcula la distancia entre dos puntos según la métrica elegida."""
        if self.metrica == 'manhattan':
            # Suma de diferencias absolutas
            return sum(abs(a - b) for a, b in zip(p1, p2))
        else:
            # Raíz cuadrada de suma de cuadrados
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def ajustar(self, datos: List[List[float]], verbose: bool = False) -> int:
        """
        Ajusta el modelo a los datos usando el algoritmo K-Medias clásico.
        Inicializa centroides aleatoriamente y alterna entre asignar puntos y actualizar centroides.
        """
        # Inicializar centroides (k puntos aleatorios del dataset)
        self.centroides = random.sample(datos, self.k)
        for iteracion in range(self.max_iter):
            # Paso 1: asignar cada punto al centroide más cercano
            nuevas_asignaciones = []
            for punto in datos:
                distancias = [self._distancia(punto, c) for c in self.centroides]
                cluster = distancias.index(min(distancias))
                nuevas_asignaciones.append(cluster)
            # Verificar convergencia: si no cambian las asignaciones, terminamos
            if nuevas_asignaciones == self.asignaciones:
                self.asignaciones = nuevas_asignaciones
                self._calcular_inercia(datos)
                if verbose:
                    print(f"Convergencia en iteración {iteracion + 1}")
                return iteracion + 1
            self.asignaciones = nuevas_asignaciones
            # Paso 2: actualizar centroides como la media de los puntos asignados
            dimension = len(datos[0])
            for j in range(self.k):
                puntos_cluster = [datos[i] for i, c in enumerate(self.asignaciones) if c == j]
                if puntos_cluster:
                    self.centroides[j] = [sum(p[d] for p in puntos_cluster) / len(puntos_cluster) 
                                         for d in range(dimension)]
        # Si no converge antes, calcular inercia final
        self._calcular_inercia(datos)
        if verbose:
            print(f"Máximo de iteraciones alcanzado ({self.max_iter})")
        return self.max_iter
    
    def _calcular_inercia(self, datos: List[List[float]]):
        """Calcula la inercia (suma de distancias cuadradas intra-cluster)."""
        self.inercia = 0.0
        for punto, cluster in zip(datos, self.asignaciones):
            dist = self._distancia(punto, self.centroides[cluster])
            self.inercia += dist ** 2
    
    def predecir(self, punto: List[float]) -> int:
        """Predice el cluster de un punto."""
        distancias = [self._distancia(punto, c) for c in self.centroides]
        return distancias.index(min(distancias))

# ============================================================================
# Funciones de Utilidad
# ============================================================================

def normalizar_datos(datos: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Normaliza datos usando z-score (media=0, desviación=1).
    
    Returns:
        (datos_normalizados, medias, desviaciones)
    """
    if not datos or not datos[0]:
        return datos, [], []
    # Normalización z-score: para cada dimensión, restamos la media y dividimos entre la desviación estándar
    dimension = len(datos[0])
    n = len(datos)
    # Calcular medias
    medias = [sum(datos[i][d] for i in range(n)) / n for d in range(dimension)]
    # Calcular desviaciones estándar (evitar división por cero)
    varianzas = [sum((datos[i][d] - medias[d]) ** 2 for i in range(n)) / n 
                for d in range(dimension)]
    desviaciones = [math.sqrt(v) if v > 0 else 1.0 for v in varianzas]
    # Normalizar cada punto
    datos_norm = []
    for punto in datos:
        punto_norm = [(punto[d] - medias[d]) / desviaciones[d] for d in range(dimension)]
        datos_norm.append(punto_norm)
    return datos_norm, medias, desviaciones

def calcular_exactitud(y_real: List[str], y_pred: List[str]) -> float:
    """Calcula la exactitud de las predicciones."""
    correctos = sum(1 for yr, yp in zip(y_real, y_pred) if yr == yp)
    return correctos / len(y_real) if y_real else 0.0

# ============================================================================
# Generación de Datos Sintéticos
# ============================================================================

def generar_datos_clasificacion_2d(n_por_clase: int = 50) -> Tuple[List[List[float]], List[str]]:
    """
    Genera datos sintéticos 2D para clasificación.
    
    Returns:
        (X, y): Datos y etiquetas
    """
    X = []
    y = []
    
    # Clase A: centrada en (2, 2)
    for _ in range(n_por_clase):
        x = [random.gauss(2, 0.5), random.gauss(2, 0.5)]
        X.append(x)
        y.append('A')
    
    # Clase B: centrada en (-2, -2)
    for _ in range(n_por_clase):
        x = [random.gauss(-2, 0.5), random.gauss(-2, 0.5)]
        X.append(x)
        y.append('B')
    
    # Clase C: centrada en (2, -2)
    for _ in range(n_por_clase):
        x = [random.gauss(2, 0.5), random.gauss(-2, 0.5)]
        X.append(x)
        y.append('C')
    
    # Mezclar
    indices = list(range(len(X)))
    random.shuffle(indices)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    
    return X, y

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra k-NN y K-Medias."""
    print("MODO DEMO: k-NN, K-Medias y Clustering\n")
    
    random.seed(42)
    
    # ========================================
    # Parte 1: k-NN para clasificación
    # ========================================
    print("=" * 60)
    print("Parte 1: Clasificación con k-NN")
    print("=" * 60)
    
    # Generar datos sintéticos 2D para tres clases bien separadas
    X, y = generar_datos_clasificacion_2d(n_por_clase=40)
    print(f"Generados {len(X)} puntos en 3 clases")
    print(f"Primeros 5: {[f'[{p[0]:.2f}, {p[1]:.2f}]->{c}' for p, c in zip(X[:5], y[:5])]}")
    print()
    # Dividir en entrenamiento (70%) y prueba (30%)
    n_train = int(0.7 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    print(f"Conjunto de entrenamiento: {len(X_train)} puntos")
    print(f"Conjunto de prueba: {len(X_test)} puntos")
    print()
    
    # Entrenar k-NN con diferentes valores de k
    print("Evaluando k-NN con diferentes valores de k:\n")
    
    # Probar k-NN con diferentes valores de k para ver el efecto en exactitud
    for k in [1, 3, 5, 7]:
        knn = KNN(k=k, metrica='euclidiana')
        knn.entrenar(X_train, y_train)
        y_pred = knn.predecir_batch(X_test)
        exactitud = calcular_exactitud(y_test, y_pred)
        print(f"k={k}: Exactitud = {exactitud:.2%}")
    
    print()
    
    # ========================================
    # Parte 2: K-Medias para clustering
    # ========================================
    print("=" * 60)
    print("Parte 2: Clustering con K-Medias")
    print("=" * 60)
    
    # Generar datos no etiquetados (solo X)
    print(f"\nUsando los mismos {len(X)} puntos para clustering no supervisado")
    print()
    
    # Probar con diferentes valores de k
    print("Evaluando K-Medias con diferentes valores de k:\n")
    
    # Probar K-Medias con diferentes valores de k para ver el efecto en la inercia
    for k in [2, 3, 4, 5]:
        kmeans = KMediasCluster(k=k, max_iter=50, metrica='euclidiana')
        iteraciones = kmeans.ajustar(X, verbose=False)
        print(f"k={k}: Inercia = {kmeans.inercia:.2f}, Iteraciones = {iteraciones}")
    
    print()
    
    # ========================================
    # Parte 3: Comparación de métricas
    # ========================================
    print("=" * 60)
    print("Parte 3: Comparación de métricas de distancia")
    print("=" * 60)
    
    print("\nk-NN (k=3) con diferentes métricas:\n")
    
    # Comparar el efecto de la métrica de distancia en k-NN
    for metrica in ['euclidiana', 'manhattan']:
        knn = KNN(k=3, metrica=metrica)
        knn.entrenar(X_train, y_train)
        y_pred = knn.predecir_batch(X_test)
        exactitud = calcular_exactitud(y_test, y_pred)
        print(f"Métrica {metrica}: Exactitud = {exactitud:.2%}")
    
    print()
    
    # ========================================
    # Parte 4: Efecto de la normalización
    # ========================================
    print("=" * 60)
    print("Parte 4: Efecto de la normalización")
    print("=" * 60)
    
    # Crear datos con escalas diferentes
    # Simular datos con escalas desbalanceadas para mostrar la importancia de la normalización
    X_desbalanceado = [[x[0] * 10, x[1]] for x in X]
    X_train_desb = [[x[0] * 10, x[1]] for x in X_train]
    X_test_desb = [[x[0] * 10, x[1]] for x in X_test]
    print("\nDatos con escala desbalanceada (x0 * 10):")
    # Sin normalización: la dimensión x domina la distancia
    knn = KNN(k=3, metrica='euclidiana')
    knn.entrenar(X_train_desb, y_train)
    y_pred = knn.predecir_batch(X_test_desb)
    exactitud_sin_norm = calcular_exactitud(y_test, y_pred)
    print(f"  Sin normalización: Exactitud = {exactitud_sin_norm:.2%}")
    # Con normalización: ambas dimensiones tienen igual peso
    X_train_norm, medias, desv = normalizar_datos(X_train_desb)
    X_test_norm = [[(x[d] - medias[d]) / desv[d] for d in range(len(x))] 
                   for x in X_test_desb]
    knn_norm = KNN(k=3, metrica='euclidiana')
    knn_norm.entrenar(X_train_norm, y_train)
    y_pred_norm = knn_norm.predecir_batch(X_test_norm)
    exactitud_con_norm = calcular_exactitud(y_test, y_pred_norm)
    print(f"  Con normalización: Exactitud = {exactitud_con_norm:.2%}")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario experimentar con k-NN y K-Medias."""
    print("MODO INTERACTIVO: k-NN y K-Medias\n")
    
    print("Seleccione el algoritmo:")
    print("1. k-NN (clasificación)")
    print("2. K-Medias (clustering)")
    
    opcion = input("Ingrese el número (default=1): ").strip()
    
    if opcion == "2":
        # K-Medias
        print("\nK-Medias (Clustering)")
        
        k = int(input("Ingrese el número de clusters (default=3): ") or "3")
        
        print("\n¿Desea generar datos sintéticos 2D? (s/n, default=s):")
        gen = input("> ").strip().lower()
        
        if gen == "n":
            print("\nIngrese puntos 2D (x y) uno por línea. Escriba 'fin' al terminar.\n")
            datos = []
            while True:
                entrada = input("> ").strip()
                if entrada.lower() == "fin":
                    break
                partes = entrada.split()
                if len(partes) >= 2:
                    datos.append([float(partes[0]), float(partes[1])])
        else:
            n_puntos = int(input("\nIngrese el número de puntos a generar (default=100): ") or "100")
            n_clusters_real = int(input("Clusters reales en los datos (default=3): ") or "3")
            
            # Generar datos
            datos = []
            for i in range(n_clusters_real):
                cx = random.uniform(-5, 5)
                cy = random.uniform(-5, 5)
                for _ in range(n_puntos // n_clusters_real):
                    x = cx + random.gauss(0, 0.5)
                    y = cy + random.gauss(0, 0.5)
                    datos.append([x, y])
        
        if not datos:
            print("No hay datos.")
            return
        
        print(f"\nEjecutando K-Medias con k={k} en {len(datos)} puntos...")
        
        kmeans = KMediasCluster(k=k, max_iter=100, metrica='euclidiana')
        iteraciones = kmeans.ajustar(datos, verbose=True)
        
        print(f"\nCentroides encontrados:")
        for i, c in enumerate(kmeans.centroides):
            print(f"  Cluster {i+1}: [{c[0]:.2f}, {c[1]:.2f}]")
        print(f"Inercia: {kmeans.inercia:.2f}")
    
    else:
        # k-NN
        print("\nk-NN (Clasificación)")
        
        k = int(input("Ingrese el valor de k (default=3): ") or "3")
        
        print("¿Desea generar datos sintéticos? (s/n, default=s):")
        gen = input("> ").strip().lower()
        
        if gen == "n":
            print("\nIngrese datos de entrenamiento (x y clase) uno por línea.")
            print("Escriba 'fin' al terminar.\n")
            
            X_train = []
            y_train = []
            
            while True:
                entrada = input("> ").strip()
                if entrada.lower() == "fin":
                    break
                partes = entrada.split()
                if len(partes) >= 3:
                    X_train.append([float(partes[0]), float(partes[1])])
                    y_train.append(partes[2])
            
            if not X_train:
                print("No hay datos.")
                return
            
            # Entrenar
            knn = KNN(k=k, metrica='euclidiana')
            knn.entrenar(X_train, y_train)
            
            print(f"\nModelo entrenado con {len(X_train)} puntos.")
            
            # Predecir
            print("\nIngrese un punto para clasificar (x y):")
            entrada = input("> ").strip()
            partes = entrada.split()
            
            if len(partes) >= 2:
                punto = [float(partes[0]), float(partes[1])]
                clase, probs = knn.predecir(punto)
                
                print(f"\nClase predicha: {clase}")
                print(f"Probabilidades: {probs}")
        else:
            # Generar datos sintéticos
            X, y = generar_datos_clasificacion_2d(n_por_clase=30)
            
            # Dividir
            n_train = int(0.7 * len(X))
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            
            # Entrenar
            knn = KNN(k=k, metrica='euclidiana')
            knn.entrenar(X_train, y_train)
            
            # Evaluar
            y_pred = knn.predecir_batch(X_test)
            exactitud = calcular_exactitud(y_test, y_pred)
            
            print(f"\nModelo entrenado con {len(X_train)} puntos.")
            print(f"Evaluado con {len(X_test)} puntos.")
            print(f"Exactitud: {exactitud:.2%}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("035-E2: k-NN, K-Medias y Clustering")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Ejemplos con datos 2D")
    print("2. INTERACTIVO: Experimentar con algoritmos")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
