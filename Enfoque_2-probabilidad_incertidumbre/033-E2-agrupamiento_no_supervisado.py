"""
033-E2-agrupamiento_no_supervisado.py
--------------------------------
Este script presenta técnicas de agrupamiento no supervisado:
- Introduce k-medias, GMM (EM) y clustering jerárquico a nivel conceptual.
- Mide calidad de clusters con inercia y silueta.
- Discute selección de k y validación cruzada no supervisada.

El programa puede ejecutarse en dos modos:
1. DEMO: clustering de datos 2D sintéticos.
2. INTERACTIVO: permite cargar datos y ajustar hiperparámetros de clustering.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple, Dict
from collections import defaultdict

# ============================================================================
# Algoritmo K-Medias
# ============================================================================

class KMedias:
    """
    Algoritmo de clustering K-Medias (K-Means).
    """
    
    def __init__(self, k: int, max_iter: int = 100):
        """
        Inicializa el algoritmo K-Medias.
        
        Args:
            k: Número de clusters
            max_iter: Número máximo de iteraciones
        """
        self.k = k  # Número de clusters
        self.max_iter = max_iter  # Máximo de iteraciones
        
        self.centroides = []  # Centroides de los clusters
        self.asignaciones = []  # Asignación de cada punto a un cluster
        self.inercia = 0.0  # Suma de distancias cuadradas intra-cluster
    
    def _distancia_euclidiana(self, p1: List[float], p2: List[float]) -> float:
        """Calcula la distancia euclidiana entre dos puntos."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def _inicializar_centroides(self, datos: List[List[float]]):
        """Inicializa centroides seleccionando k puntos aleatorios."""
        # K-means++: inicialización mejorada
        # Primer centroide aleatorio
        self.centroides = [random.choice(datos)[:]]
        
        # Seleccionar k-1 centroides restantes
        for _ in range(self.k - 1):
            # Calcular distancias al centroide más cercano
            distancias = []
            for punto in datos:
                min_dist = min(self._distancia_euclidiana(punto, c) 
                              for c in self.centroides)
                distancias.append(min_dist ** 2)
            
            # Seleccionar siguiente centroide con probabilidad proporcional a distancia²
            suma_dist = sum(distancias)
            if suma_dist > 0:
                probs = [d / suma_dist for d in distancias]
                # Muestreo ponderado
                r = random.random()
                acum = 0.0
                for i, p in enumerate(probs):
                    acum += p
                    if r <= acum:
                        self.centroides.append(datos[i][:])
                        break
            else:
                self.centroides.append(random.choice(datos)[:])
    
    def _asignar_clusters(self, datos: List[List[float]]) -> bool:
        """
        Asigna cada punto al cluster más cercano.
        
        Returns:
            True si hubo cambios en las asignaciones
        """
        nuevas_asignaciones = []
        cambios = False
        
        for i, punto in enumerate(datos):
            # Encontrar centroide más cercano
            distancias = [self._distancia_euclidiana(punto, c) 
                         for c in self.centroides]
            cluster = distancias.index(min(distancias))
            nuevas_asignaciones.append(cluster)
            
            if i >= len(self.asignaciones) or cluster != self.asignaciones[i]:
                cambios = True
        
        self.asignaciones = nuevas_asignaciones
        return cambios
    
    def _actualizar_centroides(self, datos: List[List[float]]):
        """Actualiza centroides como la media de los puntos en cada cluster."""
        dimension = len(datos[0])
        
        for j in range(self.k):
            # Puntos asignados al cluster j
            puntos_cluster = [datos[i] for i, c in enumerate(self.asignaciones) if c == j]
            
            if puntos_cluster:
                # Calcular media en cada dimensión
                nuevo_centroide = []
                for d in range(dimension):
                    media_d = sum(p[d] for p in puntos_cluster) / len(puntos_cluster)
                    nuevo_centroide.append(media_d)
                self.centroides[j] = nuevo_centroide
    
    def _calcular_inercia(self, datos: List[List[float]]):
        """Calcula la inercia (suma de distancias cuadradas intra-cluster)."""
        self.inercia = 0.0
        
        for punto, cluster in zip(datos, self.asignaciones):
            dist = self._distancia_euclidiana(punto, self.centroides[cluster])
            self.inercia += dist ** 2
    
    def ajustar(self, datos: List[List[float]], verbose: bool = True) -> int:
        """
        Ajusta el modelo a los datos.
        
        Args:
            datos: Lista de puntos (cada punto es una lista de coordenadas)
            verbose: Si imprimir progreso
            
        Returns:
            Número de iteraciones realizadas
        """
        # Inicializar centroides
        self._inicializar_centroides(datos)
        
        # Iterar hasta convergencia o máximo de iteraciones
        for iteracion in range(self.max_iter):
            # Asignar puntos a clusters
            cambios = self._asignar_clusters(datos)
            
            # Actualizar centroides
            self._actualizar_centroides(datos)
            
            if verbose and (iteracion + 1) % 10 == 0:
                self._calcular_inercia(datos)
                print(f"Iteración {iteracion + 1}: Inercia = {self.inercia:.4f}")
            
            # Verificar convergencia
            if not cambios:
                self._calcular_inercia(datos)
                if verbose:
                    print(f"Convergencia alcanzada en iteración {iteracion + 1}")
                    print(f"Inercia final: {self.inercia:.4f}")
                return iteracion + 1
        
        self._calcular_inercia(datos)
        if verbose:
            print(f"Máximo de iteraciones alcanzado ({self.max_iter})")
            print(f"Inercia final: {self.inercia:.4f}")
        
        return self.max_iter
    
    def predecir(self, punto: List[float]) -> int:
        """Predice el cluster de un nuevo punto."""
        distancias = [self._distancia_euclidiana(punto, c) 
                     for c in self.centroides]
        return distancias.index(min(distancias))

# ============================================================================
# Métricas de Evaluación
# ============================================================================

def calcular_silueta(datos: List[List[float]], asignaciones: List[int], 
                    centroides: List[List[float]]) -> float:
    """
    Calcula el coeficiente de silueta promedio.
    Mide qué tan bien está cada punto asignado a su cluster.
    Valores cercanos a 1 indican buenos clusters.
    """
    def distancia(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    n = len(datos)
    if n == 0:
        return 0.0
    
    # Agrupar puntos por cluster
    clusters = defaultdict(list)
    for i, c in enumerate(asignaciones):
        clusters[c].append(i)
    
    siluetas = []
    
    for i in range(n):
        punto = datos[i]
        cluster_actual = asignaciones[i]
        
        # a(i): distancia promedio a puntos del mismo cluster
        puntos_mismo_cluster = [j for j in clusters[cluster_actual] if j != i]
        if puntos_mismo_cluster:
            a = sum(distancia(punto, datos[j]) for j in puntos_mismo_cluster) / len(puntos_mismo_cluster)
        else:
            a = 0.0
        
        # b(i): distancia promedio mínima a puntos de otros clusters
        b = float('inf')
        for otro_cluster in clusters:
            if otro_cluster != cluster_actual:
                puntos_otro_cluster = clusters[otro_cluster]
                if puntos_otro_cluster:
                    dist_promedio = sum(distancia(punto, datos[j]) 
                                       for j in puntos_otro_cluster) / len(puntos_otro_cluster)
                    b = min(b, dist_promedio)
        
        if b == float('inf'):
            b = 0.0
        
        # Silueta: (b - a) / max(a, b)
        if max(a, b) > 0:
            silueta = (b - a) / max(a, b)
        else:
            silueta = 0.0
        
        siluetas.append(silueta)
    
    return sum(siluetas) / n if siluetas else 0.0

# ============================================================================
# Funciones de Generación de Datos
# ============================================================================

def generar_clusters_2d(k: int, n_por_cluster: int, 
                        separacion: float = 3.0) -> List[List[float]]:
    """
    Genera datos 2D sintéticos con k clusters separados.
    
    Args:
        k: Número de clusters
        n_por_cluster: Puntos por cluster
        separacion: Distancia entre centroides de clusters
        
    Returns:
        Lista de puntos 2D
    """
    datos = []
    
    # Generar k clusters en círculo
    for i in range(k):
        # Ángulo para distribuir clusters uniformemente
        angulo = 2 * math.pi * i / k
        
        # Centro del cluster
        cx = separacion * math.cos(angulo)
        cy = separacion * math.sin(angulo)
        
        # Generar puntos alrededor del centro
        for _ in range(n_por_cluster):
            x = cx + random.gauss(0, 0.5)
            y = cy + random.gauss(0, 0.5)
            datos.append([x, y])
    
    # Mezclar aleatoriamente
    random.shuffle(datos)
    return datos

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """Demuestra clustering con K-Medias."""
    print("MODO DEMO: Agrupamiento no Supervisado (K-Medias)\n")
    
    # Configurar semilla
    random.seed(42)
    
    # ========================================
    # Generar datos sintéticos
    # ========================================
    print("=" * 60)
    print("Generando datos sintéticos")
    print("=" * 60)
    
    k_real = 4
    n_por_cluster = 30
    datos = generar_clusters_2d(k_real, n_por_cluster, separacion=5.0)
    
    print(f"Generados {len(datos)} puntos en {k_real} clusters")
    print(f"Primeros 5 puntos: {[f'[{p[0]:.2f}, {p[1]:.2f}]' for p in datos[:5]]}")
    print()
    
    # ========================================
    # Aplicar K-Medias
    # ========================================
    print("=" * 60)
    print("Aplicando K-Medias")
    print("=" * 60)
    
    modelo = KMedias(k=k_real, max_iter=50)
    iteraciones = modelo.ajustar(datos, verbose=True)
    print()
    
    # ========================================
    # Mostrar resultados
    # ========================================
    print("=" * 60)
    print("Resultados")
    print("=" * 60)
    
    print(f"Centroides encontrados:")
    for i, c in enumerate(modelo.centroides):
        print(f"  Cluster {i+1}: [{c[0]:.2f}, {c[1]:.2f}]")
    print()
    
    # Contar puntos por cluster
    conteos = defaultdict(int)
    for asig in modelo.asignaciones:
        conteos[asig] += 1
    
    print(f"Distribución de puntos:")
    for i in range(k_real):
        print(f"  Cluster {i+1}: {conteos[i]} puntos")
    print()
    
    # Calcular silueta
    silueta = calcular_silueta(datos, modelo.asignaciones, modelo.centroides)
    print(f"Coeficiente de silueta: {silueta:.4f}")
    print(f"Inercia: {modelo.inercia:.4f}")
    print()
    
    # ========================================
    # Método del codo (Elbow Method)
    # ========================================
    print("=" * 60)
    print("Método del Codo (selección de k)")
    print("=" * 60)
    
    print("Probando diferentes valores de k...\n")
    
    inercias = []
    valores_k = range(2, 8)
    
    for k_prueba in valores_k:
        modelo_prueba = KMedias(k=k_prueba, max_iter=50)
        modelo_prueba.ajustar(datos, verbose=False)
        inercias.append(modelo_prueba.inercia)
        print(f"k={k_prueba}: Inercia={modelo_prueba.inercia:.2f}")
    
    print(f"\nEl 'codo' sugiere k óptimo (observar caída de inercia)")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario configurar y ejecutar clustering."""
    print("MODO INTERACTIVO: Agrupamiento no Supervisado\n")
    
    # Configurar número de clusters
    k = int(input("Ingrese el número de clusters (default=3): ") or "3")
    
    # Generar o ingresar datos
    print("\n¿Desea generar datos sintéticos 2D? (s/n, default=s):")
    opcion = input("> ").strip().lower()
    
    if opcion == "n":
        # Ingresar datos manualmente
        print("\nIngrese puntos 2D en formato 'x y', uno por línea.")
        print("Escriba 'fin' cuando termine.\n")
        
        datos = []
        while True:
            entrada = input("> ").strip()
            if entrada.lower() == "fin":
                break
            
            partes = entrada.split()
            if len(partes) >= 2:
                x, y = float(partes[0]), float(partes[1])
                datos.append([x, y])
    else:
        # Generar datos sintéticos
        n_clusters_real = int(input("\nIngrese el número real de clusters a generar (default=3): ") or "3")
        n_por_cluster = int(input("Ingrese puntos por cluster (default=30): ") or "30")
        separacion = float(input("Ingrese separación entre clusters (default=3.0): ") or "3.0")
        
        print(f"\nGenerando {n_clusters_real * n_por_cluster} puntos...")
        datos = generar_clusters_2d(n_clusters_real, n_por_cluster, separacion)
    
    if not datos:
        print("No hay datos para clustering.")
        return
    
    print(f"\nTotal de puntos: {len(datos)}")
    
    # Configurar parámetros
    max_iter = int(input("Ingrese el número máximo de iteraciones (default=100): ") or "100")
    
    print("\nEjecutando K-Medias...\n")
    
    # Ajustar modelo
    modelo = KMedias(k=k, max_iter=max_iter)
    iteraciones = modelo.ajustar(datos, verbose=True)
    
    # Mostrar resultados
    print(f"\nCentroides encontrados:")
    for i, c in enumerate(modelo.centroides):
        coords = ', '.join(f'{v:.2f}' for v in c)
        print(f"  Cluster {i+1}: [{coords}]")
    
    # Calcular silueta
    silueta = calcular_silueta(datos, modelo.asignaciones, modelo.centroides)
    print(f"\nCoeficiente de silueta: {silueta:.4f}")
    print(f"Inercia: {modelo.inercia:.4f}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("=" * 60)
    print("033-E2: Agrupamiento no Supervisado")
    print("=" * 60)
    print("Seleccione el modo de ejecución:")
    print("1. DEMO: Clustering de datos 2D")
    print("2. INTERACTIVO: Configurar clustering personalizado")
    print("=" * 60)
    
    opcion = input("Ingrese su opción (1 o 2, default=1): ").strip()
    
    if opcion == "2":
        modo_interactivo()
    else:
        modo_demo()

if __name__ == "__main__":
    main()
