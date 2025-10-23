# Python-Engineering-Playbook-IA-II


**Autor:** Alejandro Aguirre Díaz.  
**Descripción:** Continuación directa del repositorio Python-Engineering-Playbook-IA, pero esta vez centrado en algoritmos de búsqueda, toma de decisiones y aprendizaje automático.    
**Ultima modificación**: Jueves 23 de octubre del 2025.

---

## 📚 Contenido del Repositorio

Este repositorio contiene implementaciones educativas de algoritmos fundamentales de Inteligencia Artificial, organizados en el directorio `Enfoque_1-busqueda_de_grafos/`. Cada script está documentado en español e incluye dos modos de ejecución:
- **MODO DEMO**: Ejecuta automáticamente un ejemplo predefinido
- **MODO INTERACTIVO**: Permite al usuario ingresar datos o seleccionar entre datasets precargados

---

## 🔍 Algoritmos Implementados del enfoque 1

### **Búsqueda No Informada (Scripts 001-007)**

#### **001 - Búsqueda en Anchura (BFS)**
Implementa el algoritmo Breadth-First Search que explora el grafo nivel por nivel usando una cola FIFO. Encuentra el camino más corto en términos de número de aristas.

#### **002 - Búsqueda de Costo Uniforme (UCS)**
Extiende BFS considerando el costo de las aristas. Utiliza una cola de prioridad (heap) para expandir el nodo con menor costo acumulado, garantizando el camino óptimo en grafos ponderados.

#### **003 - Búsqueda en Profundidad (DFS)**
Implementación recursiva de Depth-First Search que explora cada rama hasta el final antes de retroceder. Útil para exploración exhaustiva pero no garantiza el camino más corto.

#### **004 - Búsqueda en Profundidad Limitada (DLS)**
Variante de DFS que impone un límite máximo de profundidad para evitar ciclos infinitos y controlar el espacio de búsqueda en grafos profundos.

#### **005 - Búsqueda en Profundidad Iterativa (IDDFS)**
Combina las ventajas de BFS y DFS ejecutando múltiples búsquedas DLS con límites incrementales. Encuentra el camino más corto con uso eficiente de memoria.

#### **006 - Búsqueda Bidireccional**
Ejecuta dos búsquedas simultáneas: una desde el origen hacia el destino y otra inversa. Cuando ambas se encuentran, combina los caminos. Reduce el espacio de búsqueda significativamente.

#### **007 - Búsqueda en Grafos (Base Genérica)**
Implementación genérica de exploración de grafos que puede adaptarse para diferentes estrategias. Retorna todos los nodos alcanzables y el árbol de expansión.

---

### **Búsqueda Informada (Scripts 008-010)**

#### **008 - Funciones Heurísticas**
Define y demuestra el uso de funciones heurísticas h(n) que estiman el costo restante hasta el objetivo. Incluye ejemplos de heurísticas para diferentes problemas.

#### **009 - Búsqueda Voraz Primero el Mejor (Greedy Best-First)**
Expande siempre el nodo con menor valor heurístico h(n). Rápido pero no garantiza optimalidad al ignorar el costo acumulado g(n).

#### **010 - Búsquedas A* y AO***
- **A***: Usa f(n) = g(n) + h(n) para encontrar el camino óptimo (con heurística admisible)
- **AO***: Variante para grafos AND-OR con decisiones múltiples
Muestra paso a paso: nodos abiertos/cerrados, valores f/g/h, y selección de nodos.

---

### **Búsqueda Local y Optimización (Scripts 011-015)**

#### **011 - Ascenso de Colinas (Hill Climbing)**
Algoritmo de búsqueda local que selecciona iterativamente el vecino con mejor valor objetivo. Termina en óptimos locales. Implementa función objetivo f(x) = -(x-5)² + 10.

#### **012 - Búsqueda Tabú**
Mejora Hill Climbing usando una lista tabú para evitar soluciones recientes. Incorpora mecanismos de diversificación y memoria adaptativa para escapar de óptimos locales.

#### **013 - Temple Simulado (Simulated Annealing)**
Acepta empeoramientos con probabilidad decreciente según una "temperatura". Permite escapar de óptimos locales al inicio y converge gradualmente a la solución final.

#### **014 - Búsqueda por Haz Local (Beam Search)**
Mantiene k estados candidatos simultáneamente. En cada iteración, genera sucesores y selecciona los k mejores. Balance entre amplitud de exploración y eficiencia.

#### **015 - Algoritmos Genéticos**
Implementa evolución de población con selección, cruce y mutación. Incluye:
- Generación de individuos (cromosomas)
- Función de aptitud (fitness)
- Selección por torneo
- Cruce de un punto
- Mutación aleatoria

---

### **Búsqueda Online (Script 016)**

#### **016 - Búsqueda Online**
El agente explora un grafo desconocido descubriendo nodos adyacentes conforme avanza. Toma decisiones basadas en información parcial y actualiza su conocimiento dinámicamente.

---

### **Satisfacción de Restricciones - CSP (Scripts 017-022)**

#### **017 - Satisfacción de Restricciones (CSP)**
Framework base para CSP con:
- Variables con dominios de valores
- Restricciones binarias
- Backtracking con forward checking
- Estadísticas de nodos explorados

#### **018 - Búsqueda de Vuelta Atrás (Backtracking)**
Construye soluciones incrementalmente y retrocede al encontrar inconsistencias. Se adapta automáticamente al tamaño del problema según los datos ingresados.

#### **019 - Comprobación hacia Adelante (Forward Checking)**
Extiende backtracking reduciendo dominios de variables no asignadas después de cada asignación. Detecta fallos temprano reduciendo el espacio de búsqueda.

#### **020 - Propagación de Restricciones**
Implementa consistencia de arcos (Arc Consistency) para propagar efectos de asignaciones a través del grafo de restricciones. Reduce dominios de múltiples variables simultáneamente.

#### **021 - Salto Atrás Dirigido por Conflictos (CBJ)**
Cuando encuentra un callejón sin salida, analiza conjuntos de conflicto para saltar directamente a la variable que causó el problema, evitando exploración innecesaria.

#### **022 - Mínimos Conflictos**
Algoritmo de reparación iterativa que:
- Inicia con asignación completa (posiblemente inconsistente)
- Selecciona variables en conflicto
- Reasigna valores que minimizan conflictos
Especialmente eficiente para problemas grandes como N-Reinas.

---

### **Búsqueda Adversaria (Script 023)**

#### **023 - Acondicionamiento y Corte (Minimax con Alfa-Beta)**
Implementa búsqueda adversaria para juegos de dos jugadores:
- Minimax: maximiza la mejor peor opción
- Poda Alfa-Beta: elimina ramas que no afectan la decisión
- Profundidad de corte (cutoff) con función de evaluación heurística
Incluye implementación completa de Tres en Raya (Tic-Tac-Toe) con visualización de valores alfa/beta y momentos de poda.

---

### **Teoría de Decisiones (Scripts 024-026)**

#### **024 - Teoría de la Utilidad**
Implementa cálculo de utilidad esperada con múltiples funciones:
- **Lineal**: u(x) = x
- **Logarítmica**: u(x) = k·log(x) (aversión al riesgo decreciente)
- **Exponencial**: u(x) = 1 - e^(-k·x) (aversión al riesgo constante)
Muestra paso a paso: probabilidades, valores, utilidades y contribuciones.

#### **025 - Redes de Decisión**
Implementa Influence Diagrams con:
- Nodos de azar (chance nodes): variables inciertas
- Nodos de decisión: acciones posibles
- Nodos de utilidad: función objetivo
Calcula utilidades esperadas para cada decisión y recomienda la acción óptima.

#### **026 - Valor de la Información (VEI)**
Calcula cuánto vale obtener información adicional antes de decidir:
1. Utilidad esperada sin información adicional
2. Utilidad esperada con información perfecta
3. VEI = diferencia entre ambas
Incluye interpretación visual con símbolos (✓ vale la pena, ✗ no vale, ○ neutral).

---

### **Procesos de Decisión de Márkov - MDP (Scripts 027-029)**

#### **027 - Iteración de Valores**
Implementa Value Iteration para encontrar política óptima:
- Actualiza V(s) usando ecuación de Bellman óptima
- V'(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γ·V(s')]
- Deriva política óptima π*(s) = argmax_a
Muestra convergencia iteración por iteración.

#### **028 - Iteración de Políticas**
Alterna entre:
- **Evaluación de política**: calcula V^π para política actual
- **Mejora de política**: actualiza π para ser greedy respecto a V^π
Converge en menos iteraciones que Value Iteration pero con evaluaciones más costosas.

#### **029 - Proceso de Decisión de Márkov (MDP)**
Implementación base de MDP con:
- Estados (S): situaciones del sistema
- Acciones (A): opciones disponibles
- Probabilidades de transición P(s,a,s')
- Recompensas R(s,a,s')
- Factor de descuento γ
Calcula valor de política V^π(s) mediante aproximación iterativa.

Incluye 3 datasets precargados:
1. MDP pequeño (3 estados, 2 acciones)
2. Mantenimiento de máquina (4 estados: Idle, Working, Broken, Repairing)
3. Modo personalizado (definición manual completa)

---

## 🎯 Características Comunes

### **Estructura de Código**
- ✅ Variables y funciones en español
- ✅ Comentarios intermedios explicativos
- ✅ Documentación de funciones (docstrings)
- ✅ Modo DEMO y modo INTERACTIVO
- ✅ Validación de sintaxis

### **Visualización**
- 📊 Progreso paso a paso por terminal
- 📈 Estadísticas (nodos explorados, iteraciones, convergencia)
- 🎨 Formato claro con separadores y encabezados
- 💡 Explicaciones didácticas del proceso

### **Datasets Precargados**
Los scripts recientes (026-029) incluyen datasets predefinidos para simplificar la interacción:
- Opción 1: Dataset pequeño/ejemplo clásico
- Opción 2: Dataset mediano/problema real
- Opción 3: Modo personalizado (ingreso manual)

---

## 🚀 Uso

Cada script se ejecuta independientemente:

```bash
# Ejemplo: ejecutar búsqueda en anchura
python3 Enfoque_1-busqueda_de_grafos/001-E1-busqueda_anchura.py

# Ejemplo: ejecutar iteración de valores con datasets
python3 Enfoque_1-busqueda_de_grafos/027-E1-iteracion-valores.py
```

Al ejecutar, selecciona:
1. **Modo DEMO**: Ver ejemplo predefinido con explicaciones
2. **Modo INTERACTIVO**: Ingresar datos o elegir dataset

---

## 📖 Conceptos Clave

### **Búsqueda en Grafos**
- **No informada**: No usa información del objetivo (BFS, DFS, UCS)
- **Informada**: Usa heurísticas para guiar la búsqueda (A*, Greedy)
- **Local**: Busca en vecindario sin explorar todo el espacio

### **CSP (Problema de Satisfacción de Restricciones)**
- Variables con dominios discretos
- Restricciones que limitan combinaciones válidas
- Técnicas: backtracking, forward checking, arc consistency

### **MDP (Proceso de Decisión de Márkov)**
- Modelado de decisiones secuenciales bajo incertidumbre
- Política: mapeo de estados a acciones
- Valor: utilidad esperada a largo plazo
- Algoritmos: Value Iteration, Policy Iteration

### **Teoría de Decisiones**
- Utilidad: preferencias sobre resultados
- Información: valor de reducir incertidumbre
- Redes de decisión: representación gráfica de problemas de decisión

---

## 📝 Notas de Implementación

- Los grafos se representan como diccionarios de listas de adyacencia
- Los MDPs usan tuplas (estado, acción, estado') como claves
- Las restricciones CSP son funciones booleanas binarias
- Factor de descuento γ típicamente entre 0.9 y 0.99
- Criterios de convergencia con tolerancia theta (ej: 1e-6)

---

## 🔧 Requisitos

- Python 3.6+
- Bibliotecas estándar: `collections`, `heapq`, `random`, `math`, `itertools`

---

