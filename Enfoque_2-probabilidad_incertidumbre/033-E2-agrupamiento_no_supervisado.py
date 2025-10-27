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
        # K-means++: inicialización mejorada que evita centroides agrupados
        # Primer centroide aleatorio
        self.centroides = [random.choice(datos)[:]]
        
        # Seleccionar k-1 centroides restantes
        for _ in range(self.k - 1):
            # Calcular distancias al centroide más cercano para cada punto
            # Esto identifica qué puntos están más lejos de centroides existentes
            distancias = []
            for punto in datos:
                min_dist = min(self._distancia_euclidiana(punto, c) 
                              for c in self.centroides)
                # Elevamos al cuadrado para dar más peso a puntos lejanos
                distancias.append(min_dist ** 2)
            
            # Seleccionar siguiente centroide con probabilidad proporcional a distancia²
            # Puntos lejanos tienen más probabilidad de ser elegidos como centroides
            suma_dist = sum(distancias)
            if suma_dist > 0:
                # Normalizamos para obtener distribución de probabilidad
                probs = [d / suma_dist for d in distancias]
                # Muestreo ponderado: simulamos ruleta con acumulación
                r = random.random()
                acum = 0.0
                for i, p in enumerate(probs):
                    acum += p
                    if r <= acum:
                        # Copiamos el punto seleccionado como nuevo centroide
                        self.centroides.append(datos[i][:])
                        break
            else:
                # Caso degenerado: todos los puntos están en los mismos lugares
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
            # Encontrar centroide más cercano calculando todas las distancias
            distancias = [self._distancia_euclidiana(punto, c) 
                         for c in self.centroides]
            # Asignamos el punto al cluster del centroide más cercano
            cluster = distancias.index(min(distancias))
            nuevas_asignaciones.append(cluster)
            
            # Detectar si esta asignación cambió respecto a la iteración anterior
            if i >= len(self.asignaciones) or cluster != self.asignaciones[i]:
                cambios = True
        
        # Actualizamos las asignaciones globales
        self.asignaciones = nuevas_asignaciones
        # Retornamos True si al menos un punto cambió de cluster
        return cambios
    
    def _actualizar_centroides(self, datos: List[List[float]]):
        """Actualiza centroides como la media de los puntos en cada cluster."""
        dimension = len(datos[0])
        
        for j in range(self.k):
            # Recolectamos todos los puntos asignados al cluster j
            puntos_cluster = [datos[i] for i, c in enumerate(self.asignaciones) if c == j]
            
            if puntos_cluster:
                # Calcular media en cada dimensión (centroide = promedio de coordenadas)
                nuevo_centroide = []
                for d in range(dimension):
                    # Promediamos la coordenada d de todos los puntos del cluster
                    media_d = sum(p[d] for p in puntos_cluster) / len(puntos_cluster)
                    nuevo_centroide.append(media_d)
                # Reemplazamos el centroide anterior con el nuevo promedio
                self.centroides[j] = nuevo_centroide
    
    def _calcular_inercia(self, datos: List[List[float]]):
        """Calcula la inercia (suma de distancias cuadradas intra-cluster)."""
        self.inercia = 0.0
        
        # La inercia mide qué tan compactos son los clusters
        # Menor inercia = puntos más cercanos a sus centroides
        for punto, cluster in zip(datos, self.asignaciones):
            dist = self._distancia_euclidiana(punto, self.centroides[cluster])
            # Sumamos distancias al cuadrado para penalizar puntos lejanos
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
        # Inicializar centroides con K-means++
        self._inicializar_centroides(datos)
        
        # Iterar hasta convergencia o máximo de iteraciones
        # Cada iteración: asignar → actualizar centroides → verificar convergencia
        for iteracion in range(self.max_iter):
            # Paso 1: Asignar cada punto al centroide más cercano
            cambios = self._asignar_clusters(datos)
            
            # Paso 2: Actualizar centroides como media de puntos asignados
            self._actualizar_centroides(datos)
            
            # Mostrar progreso cada 10 iteraciones
            if verbose and (iteracion + 1) % 10 == 0:
                self._calcular_inercia(datos)
                print(f"Iteración {iteracion + 1}: Inercia = {self.inercia:.4f}")
            
            # Verificar convergencia: si ningún punto cambió de cluster, terminamos
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
    Valores cercanos a 1 indican buenos clusters, cercanos a -1 indican mala asignación.
    """
    def distancia(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    n = len(datos)
    if n == 0:
        return 0.0
    
    # Agrupar índices de puntos por su cluster asignado
    # Esto facilita calcular distancias intra e inter-cluster
    clusters = defaultdict(list)
    for i, c in enumerate(asignaciones):
        clusters[c].append(i)
    
    siluetas = []
    
    for i in range(n):
        punto = datos[i]
        cluster_actual = asignaciones[i]
        
        # a(i): cohesión intra-cluster (distancia promedio a puntos del mismo cluster)
        # Menor a(i) = más compacto el cluster
        puntos_mismo_cluster = [j for j in clusters[cluster_actual] if j != i]
        if puntos_mismo_cluster:
            a = sum(distancia(punto, datos[j]) for j in puntos_mismo_cluster) / len(puntos_mismo_cluster)
        else:
            # Punto único en su cluster
            a = 0.0
        
        # b(i): separación inter-cluster (distancia promedio mínima al cluster vecino más cercano)
        # Mayor b(i) = más separado está de otros clusters
        b = float('inf')
        for otro_cluster in clusters:
            if otro_cluster != cluster_actual:
                puntos_otro_cluster = clusters[otro_cluster]
                if puntos_otro_cluster:
                    # Calculamos distancia promedio a este cluster
                    dist_promedio = sum(distancia(punto, datos[j]) 
                                       for j in puntos_otro_cluster) / len(puntos_otro_cluster)
                    # Nos quedamos con el cluster vecino más cercano
                    b = min(b, dist_promedio)
        
        if b == float('inf'):
            # Caso con un solo cluster
            b = 0.0
        
        # Silueta: (b - a) / max(a, b)
        # Si b >> a (bien separado): s ≈ 1
        # Si a >> b (mal asignado): s ≈ -1
        # Si a ≈ b (en el límite): s ≈ 0
        if max(a, b) > 0:
            silueta = (b - a) / max(a, b)
        else:
            silueta = 0.0
        
        siluetas.append(silueta)
    
    # Retornamos el promedio de siluetas de todos los puntos
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
    
    # Generar k clusters distribuidos uniformemente en círculo
    # Esto evita superposiciones y crea clusters claramente separados
    for i in range(k):
        # Ángulo para distribuir clusters uniformemente en 360°
        angulo = 2 * math.pi * i / k
        
        # Centro del cluster i (en coordenadas polares convertidas a cartesianas)
        cx = separacion * math.cos(angulo)
        cy = separacion * math.sin(angulo)
        
        # Generar puntos alrededor del centro con ruido gaussiano
        # Desviación estándar de 0.5 mantiene los puntos relativamente compactos
        for _ in range(n_por_cluster):
            x = cx + random.gauss(0, 0.5)
            y = cy + random.gauss(0, 0.5)
            datos.append([x, y])
    
    # Mezclar aleatoriamente para simular datos no ordenados
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
    
    # Configuramos el problema: 4 clusters bien separados
    k_real = 4
    n_por_cluster = 30
    # Generamos 120 puntos distribuidos en 4 grupos
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
    
    # Mostramos las coordenadas de los centroides encontrados
    print(f"Centroides encontrados:")
    for i, c in enumerate(modelo.centroides):
        print(f"  Cluster {i+1}: [{c[0]:.2f}, {c[1]:.2f}]")
    print()
    
    # Contar cuántos puntos fueron asignados a cada cluster
    # Esto verifica el balance de la asignación
    conteos = defaultdict(int)
    for asig in modelo.asignaciones:
        conteos[asig] += 1
    
    print(f"Distribución de puntos:")
    for i in range(k_real):
        print(f"  Cluster {i+1}: {conteos[i]} puntos")
    print()
    
    # Calcular métricas de calidad del clustering
    silueta = calcular_silueta(datos, modelo.asignaciones, modelo.centroides)
    print(f"Coeficiente de silueta: {silueta:.4f}")
    print(f"  (Valores cercanos a 1 = buenos clusters)")
    print(f"Inercia: {modelo.inercia:.4f}")
    print(f"  (Menor inercia = clusters más compactos)")
    print()
    
    # ========================================
    # Método del codo (Elbow Method)
    # ========================================
    print("=" * 60)
    print("Método del Codo (selección de k)")
    print("=" * 60)
    
    print("Probando diferentes valores de k...\n")
    
    # El método del codo ayuda a identificar el k óptimo
    # Buscamos el punto donde la inercia deja de disminuir significativamente
    inercias = []
    valores_k = range(2, 8)
    
    for k_prueba in valores_k:
        modelo_prueba = KMedias(k=k_prueba, max_iter=50)
        modelo_prueba.ajustar(datos, verbose=False)
        inercias.append(modelo_prueba.inercia)
        print(f"k={k_prueba}: Inercia={modelo_prueba.inercia:.2f}")
    
    print(f"\nEl 'codo' sugiere k óptimo (observar caída de inercia)")
    print(f"Buscamos donde la curva forma un 'codo' (cambio de pendiente)")
    print()

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite al usuario configurar y ejecutar clustering."""
    print("MODO INTERACTIVO: Agrupamiento no Supervisado\n")
    
    # Configurar número de clusters k que queremos encontrar
    k = int(input("Ingrese el número de clusters (default=3): ") or "3")
    
    # Generar o ingresar datos manualmente
    print("\n¿Desea generar datos sintéticos 2D? (s/n, default=s):")
    opcion = input("> ").strip().lower()
    
    if opcion == "n":
        # Ingresar datos manualmente (útil para probar con datos personalizados)
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
        # Generar datos sintéticos con parámetros configurables
        n_clusters_real = int(input("\nIngrese el número real de clusters a generar (default=3): ") or "3")
        n_por_cluster = int(input("Ingrese puntos por cluster (default=30): ") or "30")
        separacion = float(input("Ingrese separación entre clusters (default=3.0): ") or "3.0")
        
        print(f"\nGenerando {n_clusters_real * n_por_cluster} puntos...")
        datos = generar_clusters_2d(n_clusters_real, n_por_cluster, separacion)
    
    if not datos:
        print("No hay datos para clustering.")
        return
    
    print(f"\nTotal de puntos: {len(datos)}")
    
    # Configurar parámetros del algoritmo
    max_iter = int(input("Ingrese el número máximo de iteraciones (default=100): ") or "100")
    
    print("\nEjecutando K-Medias...\n")
    
    # Ajustar modelo K-Medias con los datos y parámetros configurados
    modelo = KMedias(k=k, max_iter=max_iter)
    iteraciones = modelo.ajustar(datos, verbose=True)
    
    # Mostrar resultados finales del clustering
    print(f"\nCentroides encontrados:")
    for i, c in enumerate(modelo.centroides):
        coords = ', '.join(f'{v:.2f}' for v in c)
        print(f"  Cluster {i+1}: [{coords}]")
    
    # Calcular y mostrar métricas de calidad
    silueta = calcular_silueta(datos, modelo.asignaciones, modelo.centroides)
    print(f"\nCoeficiente de silueta: {silueta:.4f}")
    print(f"  (Rango: -1 a 1, valores altos indican buen clustering)")
    print(f"Inercia: {modelo.inercia:.4f}")
    print(f"  (Suma de distancias² intra-cluster, menor es mejor)")

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
