"""
042-E2-redes_multicapa.py
--------------------------------
Este script explica Redes Multicapa (MLP - Multilayer Perceptron):
- Arquitecturas con varias capas ocultas para problemas no lineales.
- Retropropagación del error (backpropagation) para ajustar pesos.
- Capacidad de aproximación universal con funciones de activación no lineales.
- Discute inicialización de pesos, tasa de aprendizaje y convergencia.

El programa puede ejecutarse en dos modos:
1. DEMO: entrenamiento para clasificación XOR y aproximación de funciones.
2. INTERACTIVO: configurar arquitectura, tasa de aprendizaje y épocas.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple, Callable

# ============================================================================
# Funciones de Activación
# ============================================================================

def sigmoide(x: float) -> float:
    """
    Función sigmoide (logística): σ(x) = 1 / (1 + e^(-x))
    Salida en rango [0, 1], útil para clasificación binaria.
    """
    # Prevenir overflow en exponencial
    if x < -500:
        return 0.0
    elif x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def derivada_sigmoide(y: float) -> float:
    """
    Derivada de sigmoide: σ'(x) = σ(x) * (1 - σ(x))
    Se recibe y = σ(x) ya calculado para eficiencia.
    """
    return y * (1.0 - y)

def tanh(x: float) -> float:
    """
    Función tangente hiperbólica: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Salida en rango [-1, 1], centrada en cero.
    """
    # Prevenir overflow
    if x < -500:
        return -1.0
    elif x > 500:
        return 1.0
    return math.tanh(x)

def derivada_tanh(y: float) -> float:
    """
    Derivada de tanh: tanh'(x) = 1 - tanh²(x)
    Se recibe y = tanh(x) ya calculado.
    """
    return 1.0 - y * y

def relu(x: float) -> float:
    """
    Función ReLU (Rectified Linear Unit): max(0, x)
    Muy popular en redes profundas por su simplicidad.
    """
    return max(0.0, x)

def derivada_relu(x: float) -> float:
    """
    Derivada de ReLU: 1 si x > 0, 0 en otro caso.
    """
    return 1.0 if x > 0 else 0.0

# ============================================================================
# Clase Red Neuronal Multicapa
# ============================================================================

class RedMulticapa:
    """
    Red Neuronal Multicapa (MLP) con retropropagación.
    Arquitectura: n_entradas -> [capas_ocultas] -> n_salidas
    """
    
    def __init__(self, arquitectura: List[int], 
                 funcion_activacion: str = 'sigmoide',
                 tasa_aprendizaje: float = 0.1):
        """
        Inicializa la red neuronal multicapa.
        
        Args:
            arquitectura: Lista con número de neuronas por capa [entrada, oculta1, ..., salida]
            funcion_activacion: 'sigmoide', 'tanh' o 'relu'
            tasa_aprendizaje: Tasa de aprendizaje (η) para el descenso de gradiente
        """
        self.arquitectura = arquitectura
        self.num_capas = len(arquitectura)
        self.tasa_aprendizaje = tasa_aprendizaje
        
        # Seleccionar función de activación y su derivada
        if funcion_activacion == 'tanh':
            self.activacion = tanh
            self.derivada_activacion = derivada_tanh
        elif funcion_activacion == 'relu':
            self.activacion = relu
            self.derivada_activacion = derivada_relu
        else:
            self.activacion = sigmoide
            self.derivada_activacion = derivada_sigmoide
        
        # Inicializar pesos y sesgos
        # pesos[i] conecta la capa i con la capa i+1
        self.pesos = []
        self.sesgos = []
        
        # Inicialización Xavier/Glorot para mejor convergencia
        # Evita problemas de gradientes que desaparecen o explotan
        for i in range(self.num_capas - 1):
            n_entrada = arquitectura[i]
            n_salida = arquitectura[i + 1]
            
            # Límite para inicialización uniforme: sqrt(6 / (n_in + n_out))
            # Este valor mantiene la varianza de activaciones constante entre capas
            limite = math.sqrt(6.0 / (n_entrada + n_salida))
            
            # Matriz de pesos: n_salida × n_entrada
            # Cada peso se inicializa aleatoriamente en el rango [-limite, +limite]
            pesos_capa = [[random.uniform(-limite, limite) for _ in range(n_entrada)]
                         for _ in range(n_salida)]
            self.pesos.append(pesos_capa)
            
            # Vector de sesgos: n_salida × 1
            # También se inicializan aleatoriamente en el mismo rango
            sesgos_capa = [random.uniform(-limite, limite) for _ in range(n_salida)]
            self.sesgos.append(sesgos_capa)
        
        # Variables para almacenar activaciones durante forward pass
        self.activaciones = []
    
    def propagar_adelante(self, entradas: List[float]) -> List[float]:
        """
        Propagación hacia adelante (forward pass).
        Calcula la salida de la red dadas las entradas.
        
        Args:
            entradas: Vector de entrada
            
        Returns:
            Vector de salida de la red
        """
        # La primera activación es la entrada misma
        # Guardamos una copia para no modificar el original
        self.activaciones = [entradas[:]]
        
        # Propagar a través de todas las capas (desde entrada hasta salida)
        for capa in range(self.num_capas - 1):
            # Obtener las activaciones de la capa anterior
            activacion_anterior = self.activaciones[-1]
            activacion_nueva = []
            
            # Para cada neurona en la capa actual
            for neurona in range(len(self.pesos[capa])):
                # Inicializar suma con el sesgo (bias) de esta neurona
                # Calcular suma ponderada: Σ(w_ij * x_j) + b_i
                suma = self.sesgos[capa][neurona]
                
                # Sumar cada peso multiplicado por su entrada correspondiente
                for j in range(len(activacion_anterior)):
                    suma += self.pesos[capa][neurona][j] * activacion_anterior[j]
                
                # Aplicar función de activación (sigmoide, tanh, relu)
                # Esto introduce no linealidad en la red
                activacion_nueva.append(self.activacion(suma))
            
            # Guardar las activaciones de esta capa para usarlas en backprop
            self.activaciones.append(activacion_nueva)
        
        # Retornar activación de la capa de salida (última capa)
        return self.activaciones[-1]
    
    def propagar_atras(self, salida_deseada: List[float]) -> float:
        """
        Propagación hacia atrás (backpropagation).
        Calcula gradientes y actualiza pesos mediante descenso de gradiente.
        
        Args:
            salida_deseada: Salida esperada (target)
            
        Returns:
            Error cuadrático medio de esta muestra
        """
        # Obtener la salida que produjo la red en el forward pass
        salida_red = self.activaciones[-1]
        errores_capas = []
        
        # PASO 1: Calcular error en la capa de salida
        # δ_salida = (y - ŷ) * σ'(z)
        # donde y = salida deseada, ŷ = salida de la red
        error_salida = []
        for i in range(len(salida_red)):
            # Calcular diferencia entre lo esperado y lo obtenido
            error = salida_deseada[i] - salida_red[i]
            
            # Multiplicar por derivada de la función de activación
            # Esto nos da la "dirección" en la que ajustar los pesos
            delta = error * self.derivada_activacion(salida_red[i])
            error_salida.append(delta)
        
        # Guardar el error de la capa de salida
        errores_capas.append(error_salida)
        
        # PASO 2: Retropropagar el error a capas ocultas
        # Recorremos desde la penúltima capa hacia atrás (hacia la entrada)
        for capa in range(self.num_capas - 2, 0, -1):
            error_capa = []
            
            # Para cada neurona en la capa actual
            for neurona in range(len(self.activaciones[capa])):
                error = 0.0
                
                # Sumar errores ponderados de la capa siguiente
                # δ_j = Σ(δ_k * w_kj) * σ'(z_j)
                # El error de cada neurona depende de cómo contribuyó al error total
                for k in range(len(errores_capas[-1])):
                    # Error de neurona k en capa siguiente * peso que conecta j con k
                    error += errores_capas[-1][k] * self.pesos[capa][k][neurona]
                
                # Multiplicar por derivada de activación de esta neurona
                delta = error * self.derivada_activacion(self.activaciones[capa][neurona])
                error_capa.append(delta)
            
            # Guardar errores de esta capa
            errores_capas.append(error_capa)
        
        # Invertir para tener orden correcto (de entrada a salida)
        # Ahora errores_capas[i] corresponde a la capa i
        errores_capas.reverse()
        
        # PASO 3: Actualizar pesos y sesgos usando gradiente descendente
        # Regla de actualización: w_ij(nuevo) = w_ij(viejo) + η * δ_i * a_j
        # donde η = tasa de aprendizaje, δ_i = error de neurona i, a_j = activación de neurona j
        for capa in range(self.num_capas - 1):
            for neurona in range(len(self.pesos[capa])):
                # Actualizar sesgos (bias)
                # El sesgo se actualiza solo con el error y la tasa de aprendizaje
                self.sesgos[capa][neurona] += self.tasa_aprendizaje * errores_capas[capa][neurona]
                
                # Actualizar pesos de todas las conexiones de esta neurona
                for entrada_idx in range(len(self.pesos[capa][neurona])):
                    # Calcular ajuste: η * error * activación_entrada
                    # Si el error y la activación tienen el mismo signo, incrementamos el peso
                    # Si tienen signos opuestos, decrementamos el peso
                    ajuste = self.tasa_aprendizaje * errores_capas[capa][neurona] * \
                            self.activaciones[capa][entrada_idx]
                    self.pesos[capa][neurona][entrada_idx] += ajuste
        
        # Calcular error cuadrático medio (MSE) para esta muestra
        # Esto nos permite monitorear el progreso del entrenamiento
        ecm = sum((salida_deseada[i] - salida_red[i]) ** 2 
                  for i in range(len(salida_red))) / len(salida_red)
        
        return ecm
    
    def entrenar(self, datos_entrenamiento: List[Tuple[List[float], List[float]]], 
                 epocas: int = 1000,
                 verbose: bool = False) -> List[float]:
        """
        Entrena la red usando el conjunto de datos.
        
        Args:
            datos_entrenamiento: Lista de tuplas (entrada, salida_deseada)
            epocas: Número de iteraciones completas sobre el dataset
            verbose: Si mostrar progreso cada 100 épocas
            
        Returns:
            Lista con el error promedio por época
        """
        historial_error = []
        
        # Iterar por el número de épocas especificado
        # Una época = un pase completo por todos los datos de entrenamiento
        for epoca in range(epocas):
            error_total = 0.0
            
            # Presentar cada patrón de entrenamiento a la red
            # Esto es aprendizaje "por lotes" (batch learning)
            for entrada, salida_deseada in datos_entrenamiento:
                # Forward pass: calcular la salida de la red
                self.propagar_adelante(entrada)
                
                # Backward pass: calcular gradientes y actualizar pesos
                error = self.propagar_atras(salida_deseada)
                
                # Acumular el error de este patrón
                error_total += error
            
            # Calcular error promedio en esta época
            # Esto nos da una métrica de qué tan bien está aprendiendo la red
            error_promedio = error_total / len(datos_entrenamiento)
            historial_error.append(error_promedio)
            
            # Mostrar progreso cada 100 épocas si verbose=True
            if verbose and (epoca + 1) % 100 == 0:
                print(f"Época {epoca + 1}/{epocas} - Error: {error_promedio:.6f}")
        
        return historial_error
    
    def predecir(self, entrada: List[float]) -> List[float]:
        """
        Realiza una predicción para una entrada dada.
        
        Args:
            entrada: Vector de entrada
            
        Returns:
            Vector de salida predicho
        """
        return self.propagar_adelante(entrada)

# ============================================================================
# Funciones de Utilidad
# ============================================================================

def normalizar_datos(datos: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Normaliza datos usando min-max scaling a rango [0, 1].
    Importante para mejorar convergencia en redes neuronales.
    
    Returns:
        (datos_normalizados, minimos, maximos)
    """
    # Verificar que hay datos válidos
    if not datos or not datos[0]:
        return datos, [], []
    
    # Obtener número de dimensiones (características) de los datos
    dimension = len(datos[0])
    
    # Calcular mínimos y máximos por dimensión
    # Esto nos permite normalizar cada característica independientemente
    minimos = [min(dato[d] for dato in datos) for d in range(dimension)]
    maximos = [max(dato[d] for dato in datos) for d in range(dimension)]
    
    # Normalizar cada dato
    datos_norm = []
    for dato in datos:
        dato_norm = []
        # Normalizar cada dimensión del dato
        for d in range(dimension):
            rango = maximos[d] - minimos[d]
            if rango > 0:
                # Escalar a [0, 1] usando: (x - min) / (max - min)
                valor_norm = (dato[d] - minimos[d]) / rango
            else:
                # Si no hay variación en esta dimensión, usar 0.5
                # Esto evita división por cero
                valor_norm = 0.5
            dato_norm.append(valor_norm)
        datos_norm.append(dato_norm)
    
    # Retornar datos normalizados y los valores min/max para posible desnormalización
    return datos_norm, minimos, maximos

def calcular_precision(red: RedMulticapa, 
                       datos_prueba: List[Tuple[List[float], List[float]]],
                       umbral: float = 0.5) -> float:
    """
    Calcula la precisión de clasificación binaria.
    
    Args:
        red: Red neuronal entrenada
        datos_prueba: Datos de prueba (entrada, salida_esperada)
        umbral: Umbral para clasificación binaria
        
    Returns:
        Precisión (porcentaje de aciertos)
    """
    correctos = 0
    total = len(datos_prueba)
    
    # Evaluar cada patrón de prueba
    for entrada, salida_esperada in datos_prueba:
        # Obtener predicción de la red
        salida_red = red.predecir(entrada)
        
        # Clasificar según umbral (convertir valores continuos a binarios)
        # Si salida >= 0.5 → clase 1, si no → clase 0
        prediccion = [1.0 if s >= umbral else 0.0 for s in salida_red]
        
        # Verificar si la predicción coincide exactamente con la salida esperada
        if prediccion == salida_esperada:
            correctos += 1
    
    # Calcular porcentaje de aciertos
    return correctos / total if total > 0 else 0.0

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """
    Demuestra el funcionamiento de una red multicapa con el problema XOR
    y aproximación de funciones no lineales.
    """
    print("\n" + "="*70)
    print("MODO DEMO: Redes Neuronales Multicapa (MLP)")
    print("="*70)
    
    # ========================================
    # Parte 1: Problema XOR (no linealmente separable)
    # ========================================
    print("\n--- Parte 1: Clasificación XOR ---")
    print("\nEl problema XOR no es linealmente separable.")
    print("Requiere al menos una capa oculta para resolverse.\n")
    
    # Datos de entrenamiento XOR
    # Entrada: [x1, x2], Salida: [xor(x1, x2)]
    datos_xor = [
        ([0.0, 0.0], [0.0]),  # 0 XOR 0 = 0
        ([0.0, 1.0], [1.0]),  # 0 XOR 1 = 1
        ([1.0, 0.0], [1.0]),  # 1 XOR 0 = 1
        ([1.0, 1.0], [0.0])   # 1 XOR 1 = 0
    ]
    
    print("Datos de entrenamiento:")
    for entrada, salida in datos_xor:
        print(f"  {entrada} -> {salida}")
    
    # Crear red: 2 entradas, 4 neuronas ocultas, 1 salida
    print("\nArquitectura: 2 -> [4] -> 1")
    print("Función de activación: Sigmoide")
    print("Tasa de aprendizaje: 0.5\n")
    
    red_xor = RedMulticapa(
        arquitectura=[2, 4, 1],
        funcion_activacion='sigmoide',
        tasa_aprendizaje=0.5
    )
    
    # Entrenar
    print("Entrenando...")
    historial = red_xor.entrenar(datos_xor, epocas=5000, verbose=True)
    
    # Probar red entrenada
    print("\n--- Resultados después del entrenamiento ---")
    print(f"Error final: {historial[-1]:.6f}\n")
    
    print("Predicciones:")
    for entrada, salida_esperada in datos_xor:
        prediccion = red_xor.predecir(entrada)
        print(f"  Entrada: {entrada} -> Predicción: {prediccion[0]:.4f} (Esperado: {salida_esperada[0]})")
    
    # Calcular precisión
    precision = calcular_precision(red_xor, datos_xor, umbral=0.5)
    print(f"\nPrecisión: {precision * 100:.2f}%")
    
    # ========================================
    # Parte 2: Aproximación de función sinusoidal
    # ========================================
    print("\n" + "="*70)
    print("--- Parte 2: Aproximación de Función (sen(x)) ---")
    print("="*70)
    
    print("\nTeorema de Aproximación Universal:")
    print("Una red con al menos una capa oculta puede aproximar")
    print("cualquier función continua con precisión arbitraria.\n")
    
    # Generar datos de entrenamiento: f(x) = sin(x)
    datos_seno = []
    n_muestras = 20
    
    for i in range(n_muestras):
        # Generar valores de x en el rango [0, 2π]
        x = (i / n_muestras) * 2 * math.pi  # x en [0, 2π]
        y = math.sin(x)  # Calcular sin(x)
        
        # Normalizar entrada a [0, 1] para mejorar aprendizaje
        x_norm = x / (2 * math.pi)
        
        # Normalizar salida a [0, 1] desde [-1, 1]
        # sin(x) está en [-1, 1], lo escalamos a [0, 1]
        y_norm = (y + 1) / 2
        
        # Guardar par (entrada_normalizada, salida_normalizada)
        datos_seno.append(([x_norm], [y_norm]))
    
    print(f"Generadas {n_muestras} muestras de sin(x) en [0, 2π]")
    
    # Crear red para aproximación
    print("\nArquitectura: 1 -> [8] -> 1")
    print("Función de activación: Sigmoide")
    print("Tasa de aprendizaje: 0.3\n")
    
    red_seno = RedMulticapa(
        arquitectura=[1, 8, 1],
        funcion_activacion='sigmoide',
        tasa_aprendizaje=0.3
    )
    
    # Entrenar
    print("Entrenando...")
    historial_seno = red_seno.entrenar(datos_seno, epocas=2000, verbose=True)
    
    print(f"\nError final: {historial_seno[-1]:.6f}")
    
    # Mostrar algunas predicciones
    print("\nAlgunas predicciones:")
    for i in [0, n_muestras//4, n_muestras//2, 3*n_muestras//4]:
        entrada, salida_esperada = datos_seno[i]
        prediccion = red_seno.predecir(entrada)
        
        # Desnormalizar valores para mostrarlos en escala original
        # Revertir normalización de x: [0,1] → [0, 2π]
        x_original = entrada[0] * 2 * math.pi
        
        # Revertir normalización de y: [0,1] → [-1, 1]
        y_esperado = salida_esperada[0] * 2 - 1
        y_predicho = prediccion[0] * 2 - 1
        
        # Mostrar comparación entre valor esperado y predicho
        print(f"  x={x_original:.4f}: sin(x)={y_esperado:.4f}, red={y_predicho:.4f}, error={abs(y_esperado-y_predicho):.4f}")
    
    # ========================================
    # Parte 3: Efecto de la profundidad
    # ========================================
    print("\n" + "="*70)
    print("--- Parte 3: Comparación de Arquitecturas ---")
    print("="*70)
    
    print("\nComparando diferentes arquitecturas en XOR:\n")
    
    # Definir diferentes arquitecturas para comparar
    # Formato: ([arquitectura], "descripción")
    arquitecturas_prueba = [
        ([2, 2, 1], "Poco profunda (2 neuronas ocultas)"),
        ([2, 4, 1], "Moderada (4 neuronas ocultas)"),
        ([2, 6, 1], "Ancha (6 neuronas ocultas)"),
        ([2, 3, 3, 1], "Profunda (dos capas ocultas)")
    ]
    
    # Entrenar y evaluar cada arquitectura
    for arq, descripcion in arquitecturas_prueba:
        # Crear red con la arquitectura actual
        red = RedMulticapa(
            arquitectura=arq,
            funcion_activacion='sigmoide',
            tasa_aprendizaje=0.5
        )
        
        # Entrenar sin mostrar progreso (verbose=False)
        historial = red.entrenar(datos_xor, epocas=3000, verbose=False)
        
        # Calcular precisión en el conjunto de entrenamiento
        precision = calcular_precision(red, datos_xor)
        
        # Mostrar resultados de esta arquitectura
        print(f"{descripcion}:")
        print(f"  Arquitectura: {arq}")
        print(f"  Error final: {historial[-1]:.6f}")
        print(f"  Precisión: {precision * 100:.2f}%\n")

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """
    Permite al usuario configurar y entrenar una red multicapa personalizada.
    """
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Configuración de Red Multicapa")
    print("="*70)
    
    print("\nSeleccione el problema:")
    print("1) XOR (clasificación binaria)")
    print("2) Aproximación de función personalizada")
    print("3) Configuración manual completa")
    
    opcion = input("\nIngrese opción (1-3, default=1): ").strip()
    
    if opcion == "2":
        # Aproximación de función
        print("\n--- Aproximación de Función ---")
        
        print("\nSeleccione función objetivo:")
        print("1) sin(x)")
        print("2) x²")
        print("3) e^x")
        
        func_opcion = input("Ingrese opción (1-3, default=1): ").strip()
        
        # Definir función objetivo
        if func_opcion == "2":
            func = lambda x: x ** 2
            func_nombre = "x²"
            rango = (0, 2)
        elif func_opcion == "3":
            func = lambda x: math.exp(x)
            func_nombre = "e^x"
            rango = (0, 1)
        else:
            func = lambda x: math.sin(x)
            func_nombre = "sin(x)"
            rango = (0, 2 * math.pi)
        
        # Generar datos
        n_muestras = int(input(f"\nNúmero de muestras (default=20): ").strip() or "20")
        
        datos = []
        for i in range(n_muestras):
            x = rango[0] + (i / n_muestras) * (rango[1] - rango[0])
            y = func(x)
            
            # Normalizar
            x_norm = (x - rango[0]) / (rango[1] - rango[0])
            y_min, y_max = min(func(rango[0]), func(rango[1])), max(func(rango[0]), func(rango[1]))
            y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            
            datos.append(([x_norm], [y_norm]))
        
        print(f"\nGeneradas {n_muestras} muestras de {func_nombre}")
        
        # Configurar red
        n_ocultas = int(input("Neuronas en capa oculta (default=8): ").strip() or "8")
        tasa = float(input("Tasa de aprendizaje (default=0.3): ").strip() or "0.3")
        epocas = int(input("Épocas de entrenamiento (default=2000): ").strip() or "2000")
        
        red = RedMulticapa(
            arquitectura=[1, n_ocultas, 1],
            funcion_activacion='sigmoide',
            tasa_aprendizaje=tasa
        )
        
        print(f"\nArquitectura: 1 -> [{n_ocultas}] -> 1")
        print("Entrenando...\n")
        
        historial = red.entrenar(datos, epocas=epocas, verbose=True)
        
        print(f"\nError final: {historial[-1]:.6f}")
        
        # Mostrar predicciones
        print("\nAlgunas predicciones:")
        indices = [0, n_muestras//4, n_muestras//2, 3*n_muestras//4, n_muestras-1]
        for i in indices:
            if i < len(datos):
                entrada, salida_esperada = datos[i]
                prediccion = red.predecir(entrada)
                
                x_real = rango[0] + entrada[0] * (rango[1] - rango[0])
                print(f"  x={x_real:.4f}: esperado={salida_esperada[0]:.4f}, predicho={prediccion[0]:.4f}")
    
    elif opcion == "3":
        # Configuración manual completa
        print("\n--- Configuración Manual ---")
        
        n_entradas = int(input("Número de entradas (default=2): ").strip() or "2")
        
        # Configurar capas ocultas
        print("\nCapas ocultas (ingrese número de neuronas por capa, 'fin' para terminar):")
        capas_ocultas = []
        i = 1
        while True:
            entrada = input(f"  Capa oculta {i} (o 'fin'): ").strip()
            if entrada.lower() == 'fin':
                break
            try:
                n_neuronas = int(entrada)
                if n_neuronas > 0:
                    capas_ocultas.append(n_neuronas)
                    i += 1
            except ValueError:
                print("    Entrada inválida, intente de nuevo.")
        
        if not capas_ocultas:
            capas_ocultas = [4]  # Default
        
        n_salidas = int(input("Número de salidas (default=1): ").strip() or "1")
        
        arquitectura = [n_entradas] + capas_ocultas + [n_salidas]
        
        print("\nFunción de activación:")
        print("1) Sigmoide")
        print("2) Tanh")
        print("3) ReLU")
        
        act_opcion = input("Ingrese opción (1-3, default=1): ").strip()
        activacion = 'sigmoide'
        if act_opcion == '2':
            activacion = 'tanh'
        elif act_opcion == '3':
            activacion = 'relu'
        
        tasa = float(input("\nTasa de aprendizaje (default=0.1): ").strip() or "0.1")
        
        red = RedMulticapa(
            arquitectura=arquitectura,
            funcion_activacion=activacion,
            tasa_aprendizaje=tasa
        )
        
        print(f"\nRed creada: {arquitectura}")
        print(f"Activación: {activacion}")
        print(f"Tasa de aprendizaje: {tasa}")
        
        print("\nPara entrenar necesitará proporcionar datos de entrenamiento.")
        print("(Esta es una demostración, ejecute modo DEMO para ver ejemplos completos)")
    
    else:
        # XOR por defecto
        print("\n--- Problema XOR ---")
        
        datos_xor = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0])
        ]
        
        n_ocultas = int(input("Neuronas en capa oculta (default=4): ").strip() or "4")
        tasa = float(input("Tasa de aprendizaje (default=0.5): ").strip() or "0.5")
        epocas = int(input("Épocas de entrenamiento (default=5000): ").strip() or "5000")
        
        red = RedMulticapa(
            arquitectura=[2, n_ocultas, 1],
            funcion_activacion='sigmoide',
            tasa_aprendizaje=tasa
        )
        
        print(f"\nArquitectura: 2 -> [{n_ocultas}] -> 1")
        print("Entrenando...\n")
        
        historial = red.entrenar(datos_xor, epocas=epocas, verbose=True)
        
        print(f"\nError final: {historial[-1]:.6f}")
        
        print("\nPredicciones:")
        for entrada, salida_esperada in datos_xor:
            prediccion = red.predecir(entrada)
            print(f"  {entrada} -> {prediccion[0]:.4f} (esperado: {salida_esperada[0]})")
        
        precision = calcular_precision(red, datos_xor)
        print(f"\nPrecisión: {precision * 100:.2f}%")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("\n" + "="*70)
    print("042-E2: Redes Neuronales Multicapa (MLP)")
    print("="*70)
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (ejemplos predefinidos)")
    print("2) Modo INTERACTIVO (configuración personalizada)")
    
    opcion = input("\nIngrese opción (1 o 2, default=1): ").strip()
    
    if opcion == '2':
        modo_interactivo()
    else:
        modo_demo()
    
    print("\n" + "="*70)
    print("Programa finalizado")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
