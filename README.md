# Python Engineering Playbook - Inteligencia Artificial II

**Autor:** Alejandro Aguirre Díaz  
**Descripción:** Continuación directa del repositorio Python-Engineering-Playbook-IA, pero esta vez centrado en algoritmos de búsqueda, toma de decisiones y aprendizaje automático para algoritmos de Inteligencia Artificial que cubren búsqueda, planificación, probabilidad, razonamiento bayesiano, aprendizaje automático y percepción.  
**Última modificación:** Martes 28 de octubre del 2025.

---

## 📚 Contenido del Repositorio

Este repositorio contiene implementaciones educativas de algoritmos fundamentales de Inteligencia Artificial, organizados en dos enfoques:

- **Enfoque 1: Búsqueda y Planificación** - Algoritmos de búsqueda, satisfacción de restricciones, teoría de decisiones y aprendizaje por refuerzo
- **Enfoque 2: Probabilidad e Incertidumbre** - Razonamiento probabilístico, modelos temporales, aprendizaje bayesiano, redes neuronales y procesamiento del lenguaje

Cada script incluye:
- ✅ **Comentarios detallados** en español
- ✅ **Dos modos de ejecución**: DEMO (ejemplos predefinidos) e INTERACTIVO (configuración personalizada)
- ✅ **Código educativo** que prioriza claridad sobre eficiencia

---

## 🔍 Enfoque 1: Búsqueda y Planificación

Este enfoque cubre algoritmos de búsqueda (no informada e informada), satisfacción de restricciones, utilidad y toma de decisiones, teoría de juegos y aprendizaje por refuerzo.

### **1.1 Búsqueda No Informada (001-007)**

Scripts que implementan algoritmos de búsqueda sin heurísticas:

- **001-E1-busqueda_anchura.py** - Búsqueda en anchura (BFS): exploración nivel por nivel, garantiza solución más cercana
- **002_E1_busqueda_costo_uniforme.py** - Búsqueda de costo uniforme (UCS): expansión por costo acumulado mínimo
- **003-E1-busqueda_profundidad.py** - Búsqueda en profundidad (DFS): exploración hasta el fondo, menor uso de memoria
- **004-E1-busqueda_profundidad_limitada.py** - DFS con límite de profundidad para evitar búsquedas infinitas
- **005_E1_busqueda_profundidad_iterativa.py** - Búsqueda en profundidad iterativa: combina ventajas de BFS y DFS
- **006-E1-busqueda_bidireccional.py** - Búsqueda bidireccional: desde inicio y objetivo simultáneamente
- **007-E1-busqueda_en_grafos.py** - Búsqueda general en grafos con manejo de estados repetidos

### **1.2 Búsqueda Informada - Heurística (008-016)**

Algoritmos que utilizan heurísticas y funciones de evaluación:

- **008-E1-heuristicas.py** - Diseño y evaluación de funciones heurísticas (admisibles, consistentes)
- **009-E1-busqueda_voraz_primero_mejor.py** - Búsqueda voraz que selecciona según h(n), rápida pero no óptima
- **010-E1-busquedasA_AO.py** - Algoritmo A* con f(n) = g(n) + h(n), óptimo con heurística admisible
- **011-E1-asscension_colinas.py** - Ascenso de colinas: búsqueda local siguiendo gradiente
- **012-E1-busqueda_tabu.py** - Búsqueda tabú: memoria de corto plazo para evitar ciclos
- **013-E1-temple_simulado.py** - Temple simulado: acepta movimientos sub-óptimos con probabilidad decreciente
- **014-E1-haz_local.py** - Búsqueda de haz local: mantiene múltiples estados candidatos (k beams)
- **015-E1-algoritmos_geneticos.py** - Algoritmos genéticos: evolución mediante selección, cruce y mutación
- **016-E1-busqueda_online.py** - Búsqueda online: descubre entorno en tiempo real con conocimiento parcial

### **1.3 Satisfacción de Restricciones (017-023)**

Problemas CSP y técnicas de resolución:

- **017-E1-satisfaccion_restricciones.py** - Introducción a CSP: variables, dominios y restricciones
- **018-E1-busqueda_vuelta_atras.py** - Backtracking: asignación incremental con retroceso
- **019-E1-comprobacion_hacia_delante.py** - Forward checking: reduce dominios tras cada asignación
- **020-E1-propagacion_restricciones.py** - AC-3: consistencia de arcos para reducir dominios
- **021-E1-salto_atras_conflictos.py** - Conflict-directed backjumping: retrocede a variable causante
- **022-E1-minimos_conflictos.py** - Búsqueda local que minimiza restricciones violadas
- **023-E1-acondicionamiento_corte.py** - Minimax con poda alfa-beta para juegos adversarios

### **1.4 Utilidad y Toma de Decisiones (024-037)**

Teoría de utilidad, decisiones bajo incertidumbre, MDP y aprendizaje por refuerzo:

- **024-E1-teoria_utilidad.py** - Funciones de utilidad y maximización de utilidad esperada
- **025_E1-redes_decision.py** - Redes de decisión: nodos de azar, decisión y utilidad (MEU)
- **026-E1-valor_informacion.py** - Valor de información perfecta (VPI)
- **027-E1-iteracion-valores.py** - Value Iteration: programación dinámica para MDP
- **028-E1-iteracion-politicas.py** - Policy Iteration: alterna evaluación y mejora de política
- **029-E1-proceso_decision_markov.py** - MDP: estados, acciones, transiciones, recompensas
- **030-E1-POMDP.py** - POMDP: MDP con observaciones parciales y belief states
- **031-E1-red-bayesiana-dinamica.py** - DBN: dependencias temporales entre variables de estado
- **032-E1-teoria_juegos.py** - Teoría de juegos: matriz de pagos, equilibrio de Nash
- **033-E1-refuerzo_pasivo.py** - Aprendizaje por refuerzo pasivo: estimación de valores con Monte Carlo
- **034-E1-refuerzo_activo.py** - Aprendizaje por refuerzo activo: Q-Learning con exploración ε-greedy
- **035-E1-QLearning.py** - Q-Learning clásico: actualización temporal off-policy
- **036-E1-exploracion_vs_explotacion.py** - Estrategias de exploración: ε-greedy, UCB, softmax, multi-armed bandits
- **037-E1-busqueda_politica.py** - Policy Search: Hill Climbing, REINFORCE, Cross-Entropy, Evolution Strategies

---

## 🎲 Enfoque 2: Probabilidad e Incertidumbre

Este enfoque cubre fundamentos de probabilidad, razonamiento probabilístico, modelos temporales, aprendizaje bayesiano, redes neuronales, procesamiento del lenguaje y percepción.

### **2.1 Probabilidad (001-013)**

Fundamentos de probabilidad e incertidumbre:

- **001-E2-incertidumbre_probabilidad.py** - Eventos aleatorios, espacios muestrales, operaciones básicas
- **002-E2-RP_red-bayesiana.py** - Introducción a representación con redes bayesianas
- **003-E2-RPT_reconocimiento_habla.py** - Introducción a razonamiento probabilístico temporal
- **004-E2-aprendizaje_profundo.py** - Introducción a aprendizaje profundo y arquitecturas
- **005-E2-redes_neuronales.py** - Fundamentos de redes neuronales artificiales
- **006-E2-tratamiento_probabilistico_lenguaje.py** - Introducción a modelos probabilísticos del lenguaje
- **007-E2-percepcion.py** - Introducción a percepción y visión por computadora
- **008-E2-incertidumbre.py** - Manejo de incertidumbre en sistemas inteligentes
- **009-E2-probabilidad_a_priori.py** - Probabilidad a priori: distribuciones sin evidencia
- **010-E2-Probabilidad_condicionada_normalizacion.py** - Probabilidad condicionada P(A|B) y normalización
- **011-E2-distribucion_probabilidad.py** - Distribuciones discretas y continuas
- **012-E2-independencia_condicional.py** - Independencia condicional: P(X,Y|Z) = P(X|Z)·P(Y|Z)
- **013-E2-regla_de_bayes.py** - Regla de Bayes: actualización de creencias con evidencia

### **2.2 Razonamiento Probabilístico (014-021)**

Redes bayesianas, inferencia exacta y aproximada:

- **014-E2-red_bayesiana.py** - Redes bayesianas completas: DAG, CPT, d-separación
- **015-E2-regla_cadena.py** - Regla de la cadena: factorización en redes bayesianas
- **016-E2-manto_de_markov.py** - Markov Blanket: conjunto mínimo para independencia condicional
- **017-E2-inferencia_por_enumeracion.py** - Inferencia exacta por enumeración exhaustiva
- **018-E2-eliminacion_de_variables.py** - Eliminación de variables: inferencia exacta eficiente
- **019-E2-muestreo_directo_y_por_rechazo.py** - Inferencia aproximada mediante muestreo
- **020-E2-ponderacion_de_verosimilitud.py** - Likelihood weighting: muestreo con evidencia fija
- **021-E2-monte_carlo_para_cadenas_de_markov.py** - MCMC: Gibbs sampling, Metropolis-Hastings

### **2.3 Razonamiento Probabilístico en el Tiempo (022-029)**

Modelos temporales y series de tiempo:

- **022-E2-procesos_estacionarios.py** - Procesos estocásticos estacionarios: simulación AR(1)
- **023-E2-hipotesis_de_markov_procesos_de_markov.py** - Propiedad de Markov y cadenas de Markov
- **024-E2-filtrado_prediccion_suavizado_explicacion.py** - Filtrado, predicción y suavizado en modelos temporales
- **025-E2-algoritmo_hacia_delante_atras.py** - Forward-Backward: inferencia en HMM
- **026-E2-modelos_ocultos_de_markov.py** - HMM: estados ocultos, evaluación, Viterbi, aprendizaje
- **027-E2-filtros_de_kalman.py** - Filtros de Kalman: sistemas lineales con ruido gaussiano
- **028-E2-red_bayes_dinamica_filtrado_de_particulas.py** - DBN y filtrado de partículas
- **029-E2-reconocimiento_del_habla.py** - Aplicación de HMM a reconocimiento de voz

### **2.4 Aprendizaje Probabilístico (030-037)**

Clasificadores bayesianos, aprendizaje no supervisado y métodos avanzados:

- **030-E2-aprendizaje_bayesiano.py** - Actualización de distribuciones sobre parámetros, modelos conjugados
- **031-E2-naive_bayes.py** - Clasificador Naïve Bayes: multinomial, Bernoulli, gaussiana
- **032-E2-algoritmo_em.py** - Expectation-Maximization: aprendizaje con variables latentes
- **033-E2-agrupamiento_no_supervisado.py** - Clustering: k-means, jerárquico, métricas
- **034-E2-modelos_de_markov_ocultos.py** - HMM avanzado: Viterbi, Baum-Welch
- **035-E2-knn_kmedias_y_clustering.py** - k-NN para clasificación, comparación de clustering
- **036-E2-maquinas_de_vectores_soporte_nucleo.py** - SVM con kernels: lineal, polinomial, RBF
- **037-E2-aprendizaje_profundo.py** - Redes profundas: retropropagación, optimización, regularización

### **2.5 Redes Neuronales (038-045)**

Fundamentos de computación neuronal y arquitecturas:

- **038-E2-computacion_neuronal.py** - Neuronas umbral, funciones lógicas (AND, OR, NOT, XOR)
- **039-E2-funciones_de_activacion.py** - Funciones de activación: escalón, sigmoide, tanh, ReLU, softmax
- **040-E2-perceptron_adaline_madaline.py** - Modelos históricos: perceptrón, ADALINE, MADALINE
- **041-E2-separabilidad_lineal.py** - Límites de clasificadores lineales, problema XOR
- **042-E2-redes_multicapa.py** - Perceptrón multicapa (MLP): aproximación universal
- **043-E2-retropropagacion_del_error.py** - Backpropagation: regla de la cadena para gradientes
- **044-E2-mapas_autoorganizados_de_kohonen.py** - SOM: organización topológica no supervisada
- **045-E2-hamming_hopfield_hebb_boltzmann.py** - Redes clásicas: memorias asociativas, máquinas de Boltzmann

### **2.6 Tratamiento Probabilístico del Lenguaje (046-051)**

Modelos del lenguaje, gramáticas, recuperación de información y traducción:

- **046-E2-modelo_probabilistico_del_lenguaje_corpus.py** - Modelos n-gramas, generación de texto, perplejidad
- **047-E2-gramaticas_probabilisticas_independientes_del_contexto.py** - PCFG con algoritmo CKY
- **048-E2-gramaticas_probabilisticas_lexicalizadas.py** - PCFG lexicalizadas con head words
- **049-E2-recuperacion_de_datos.py** - Information Retrieval: TF-IDF, similitud coseno
- **050-E2-extraccion_de_informacion.py** - NER y extracción de relaciones con regex
- **051-E2-traduccion_automatica_estadistica.py** - IBM Model 1, alineamiento, EM

### **2.7 Percepción (052)**

Visión por computadora y procesamiento de imágenes:

- **052-E2-percepcion.py** - Pipeline completo de visión: filtros, bordes (Canny), segmentación (K-means), texturas (LBP), template matching, k-NN para dígitos, Hough para líneas, flujo óptico

---

## 🚀 Instalación y Uso

### **Requisitos**

- Python 3.6 o superior
- Bibliotecas estándar: `random`, `math`, `collections`, `itertools`, `typing`
- Bibliotecas opcionales (según script):
  - OpenCV: `pip install opencv-python-headless`
  - scikit-image: `pip install scikit-image`
  - scikit-learn: `pip install scikit-learn`

### **Ejecución**

Cada script puede ejecutarse directamente:

```bash
# Enfoque 1: Búsqueda
python Enfoque_1-busqueda_de_grafos/001-E1-busqueda_anchura.py
python Enfoque_1-busqueda_de_grafos/027-E1-iteracion-valores.py

# Enfoque 2: Probabilidad
python Enfoque_2-probabilidad_incertidumbre/014-E2-red_bayesiana.py
python Enfoque_2-probabilidad_incertidumbre/052-E2-percepcion.py
```

Al ejecutar, selecciona el modo:
- **DEMO**: Ejecuta ejemplos predefinidos automáticamente
- **INTERACTIVO**: Permite configurar parámetros y experimentar

---

## 📖 Glosario Técnico

### **Algoritmos de Búsqueda**

**Búsqueda No Informada (Ciega)**
- Explora el espacio de estados sin información adicional sobre el objetivo
- **BFS (Breadth-First Search)**: Explora nivel por nivel, garantiza el camino más corto en grafos no ponderados
- **DFS (Depth-First Search)**: Explora en profundidad primero, usa menos memoria pero puede no encontrar la solución óptima
- **UCS (Uniform Cost Search)**: Expande nodos por costo acumulado mínimo, óptimo para grafos ponderados

**Búsqueda Informada (Heurística)**
- Utiliza funciones heurísticas h(n) que estiman la distancia al objetivo
- **A* (A-estrella)**: Combina costo real g(n) y heurística h(n): f(n) = g(n) + h(n). Óptimo si h es admisible
- **Greedy Best-First**: Solo usa h(n), más rápido pero no garantiza optimalidad
- **Heurística admisible**: Nunca sobreestima el costo real al objetivo (h(n) ≤ costo real)
- **Heurística consistente**: h(n) ≤ costo(n,n') + h(n') para todo sucesor n' de n

**Búsqueda Local**
- Trabaja con un estado actual y explora vecinos cercanos
- **Hill Climbing**: Sube siempre por la pendiente más empinada, puede quedar atrapado en máximos locales
- **Simulated Annealing**: Acepta movimientos malos con probabilidad decreciente ("temperatura")
- **Búsqueda Tabú**: Mantiene memoria de estados visitados para evitar ciclos

### **CSP (Constraint Satisfaction Problems)**

**Definición**: Problemas con variables que deben satisfacer restricciones simultáneamente

**Componentes**
- **Variables**: Elementos a los que se asignan valores (ej: colores en grafos, posiciones en N-reinas)
- **Dominios**: Conjunto de valores posibles para cada variable (ej: {rojo, verde, azul})
- **Restricciones**: Relaciones que limitan combinaciones válidas (ej: nodos adyacentes con distinto color)

**Técnicas de Resolución**
- **Backtracking**: Asigna valores incrementalmente y retrocede al encontrar inconsistencias
- **Forward Checking**: Tras asignar una variable, elimina valores inconsistentes de variables futuras
- **Arc Consistency (AC-3)**: Propaga restricciones para reducir dominios antes de la búsqueda
- **Conflict-Directed Backjumping**: Salta directamente a la variable causante del conflicto

**Heurísticas**
- **MRV (Minimum Remaining Values)**: Elige la variable con menos valores legales restantes
- **Grado**: Elige la variable involucrada en más restricciones con variables no asignadas
- **LCV (Least Constraining Value)**: Prefiere valores que dejan más opciones a otras variables

### **MDP (Markov Decision Process)**

**Definición**: Modelo matemático para decisiones secuenciales bajo incertidumbre

**Componentes**
- **Estados (S)**: Situaciones posibles del sistema
- **Acciones (A)**: Decisiones disponibles en cada estado
- **Transiciones P(s'|s,a)**: Probabilidad de llegar a s' desde s tomando acción a
- **Recompensas R(s,a,s')**: Beneficio inmediato por la transición
- **Factor de descuento γ (gamma)**: Peso de recompensas futuras (0 < γ < 1, típicamente 0.9-0.99)

**Política y Valor**
- **Política π**: Función que mapea estados a acciones (qué hacer en cada situación)
- **Valor V^π(s)**: Utilidad esperada a largo plazo siguiendo política π desde estado s
- **Política óptima π***: Política que maximiza el valor en todos los estados

**Algoritmos**
- **Value Iteration**: Actualiza valores iterativamente usando ecuación de Bellman hasta convergencia
- **Policy Iteration**: Alterna evaluación de política y mejora hasta encontrar π*

**POMDP (Partially Observable MDP)**
- Extensión de MDP donde el agente no observa directamente el estado
- Mantiene un **belief state** (distribución de probabilidad sobre estados posibles)
- Actualiza creencias con observaciones ruidosas

### **Redes Bayesianas**

**Definición**: Grafo acíclico dirigido (DAG) que representa dependencias probabilísticas entre variables

**Componentes**
- **Nodos**: Variables aleatorias
- **Arcos**: Dependencias probabilísticas directas (padre → hijo)
- **CPT (Conditional Probability Tables)**: P(Variable|Padres) para cada nodo
- **DAG (Directed Acyclic Graph)**: Grafo sin ciclos que define estructura causal

**Independencia Condicional**
- Variable X es independiente de Y dado Z: P(X,Y|Z) = P(X|Z)·P(Y|Z)
- **Markov Blanket**: Padres + hijos + padres de los hijos de un nodo
- **D-separación**: Criterio gráfico para determinar independencias

**Inferencia**
- **Exacta**: Cálculo preciso de probabilidades
  - Enumeración: Suma sobre todas las combinaciones (exponencial)
  - Eliminación de variables: Más eficiente, orden de eliminación importa
- **Aproximada**: Estimación mediante muestras
  - Muestreo directo: Genera muestras de la distribución conjunta
  - Muestreo por rechazo: Descarta muestras inconsistentes con evidencia
  - Likelihood weighting: Pondera muestras según evidencia
  - MCMC: Gibbs sampling, Metropolis-Hastings para distribuciones complejas

### **Aprendizaje Automático**

**Aprendizaje Supervisado**
- Aprende de datos etiquetados (pares entrada-salida)
- **Clasificación**: Predice categorías discretas (ej: spam/no spam, dígitos 0-9)
- **Regresión**: Predice valores continuos (ej: precio de casa, temperatura)
- Ejemplos: Naïve Bayes, k-NN, SVM, redes neuronales

**Aprendizaje No Supervisado**
- Descubre patrones en datos sin etiquetas
- **Clustering**: Agrupa datos similares (k-means, jerárquico)
- **Reducción de dimensionalidad**: Simplifica datos preservando información (PCA, autoencoders)
- **Detección de anomalías**: Identifica datos atípicos

**Aprendizaje por Refuerzo**
- Agente aprende mediante interacción y recompensas
- **Exploración vs Explotación**: Dilema entre probar nuevas acciones o usar las mejores conocidas
- **Q-Learning**: Aprende función Q(s,a) = valor de tomar acción a en estado s
- **Policy Gradient**: Optimiza directamente la política mediante gradientes
- **Estrategias**: ε-greedy (explora con probabilidad ε), UCB, softmax

### **Modelos Temporales**

**HMM (Hidden Markov Model)**
- Modelo con estados ocultos que emiten observaciones ruidosas
- **Filtrado**: Estimar estado actual dado el pasado
- **Predicción**: Estimar estados futuros
- **Suavizado**: Estimar estados pasados con toda la evidencia
- **Algoritmo Viterbi**: Encuentra secuencia de estados más probable
- **Baum-Welch**: Aprende parámetros del HMM (EM para HMM)

**Filtros de Kalman**
- Caso especial de HMM para sistemas lineales con ruido gaussiano
- Tracking óptimo de estado en tiempo real
- Usado en navegación, control, predicción

**DBN (Dynamic Bayesian Networks)**
- Generalización de HMM a múltiples variables temporales
- Red bayesiana que evoluciona en el tiempo

### **Redes Neuronales**

**Componentes Básicos**
- **Neurona**: Unidad que computa suma ponderada + función de activación
- **Pesos**: Parámetros aprendibles que conectan neuronas
- **Bias**: Término independiente que permite desplazar la función
- **Capa**: Conjunto de neuronas que procesan simultáneamente

**Funciones de Activación**
- **Escalón**: Salida binaria 0/1
- **Sigmoide**: σ(x) = 1/(1+e^-x), salida en (0,1), usada en clasificación binaria
- **Tanh**: salida en (-1,1), centrada en cero
- **ReLU**: max(0,x), rápida y efectiva, estándar en capas ocultas
- **Softmax**: Normaliza a distribución de probabilidad, usada en clasificación multiclase

**Arquitecturas**
- **Perceptrón**: Neurona simple, solo problemas linealmente separables
- **MLP (Multilayer Perceptron)**: Red multicapa, aproximador universal de funciones
- **CNN (Convolutional)**: Capas convolucionales para imágenes
- **RNN (Recurrent)**: Conexiones recurrentes para secuencias temporales
- **Autoencoder**: Codifica y reconstruye datos, reducción de dimensionalidad

**Entrenamiento**
- **Backpropagation**: Algoritmo que propaga errores hacia atrás usando regla de la cadena
- **Gradiente descendente**: Actualiza pesos en dirección opuesta al gradiente: w ← w - α·∇L
- **Learning rate α**: Tamaño del paso en la actualización
- **Epoch**: Pasada completa por todos los datos de entrenamiento
- **Batch**: Subconjunto de datos usado en cada actualización

### **Métricas y Evaluación**

**Clasificación**
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: De las predicciones positivas, cuántas son correctas
- **Recall**: De los positivos reales, cuántos se detectan
- **F1-score**: Media armónica de precision y recall

**Clustering**
- **Inercia**: Suma de distancias al centroide más cercano (menor es mejor)
- **Silhouette**: Cohesión vs separación (-1 a 1, mayor es mejor)

**Probabilísticos**
- **Perplejidad**: Inverso de probabilidad geométrica, mide calidad de modelo de lenguaje
- **Log-likelihood**: Logaritmo de probabilidad de los datos bajo el modelo




