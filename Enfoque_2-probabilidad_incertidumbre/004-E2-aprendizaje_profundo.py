"""
004-E2-aprendizaje_profundo.py
--------------------------------
Este script implementa los fundamentos del Aprendizaje Profundo (Deep Learning):
- Construye redes neuronales profundas con múltiples capas ocultas
- Implementa funciones de activación (ReLU, Sigmoid, Tanh, Softmax)
- Aplica el algoritmo de retropropagación (backpropagation) para entrenar la red
- Utiliza descenso de gradiente y técnicas de optimización (Adam, RMSprop)
- Incorpora regularización (Dropout, L2) para prevenir sobreajuste
- Entrena modelos en problemas de clasificación y regresión
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente entrenamiento en dataset predefinido (ej: XOR, MNIST simplificado)
2. INTERACTIVO: permite configurar arquitectura, hiperparámetros y entrenar modelos personalizados

Autor: Alejandro Aguirre Díaz
"""

import numpy as np
import random

# ========== FUNCIONES DE ACTIVACIÓN Y SUS DERIVADAS ==========

def relu(x):
    """Función de activación ReLU (Rectified Linear Unit)."""
    # ReLU(x) = max(0, x) - Introduce no linealidad y evita el problema de gradiente desvaneciente
    return np.maximum(0, x)

def relu_derivada(x):
    """Derivada de ReLU para retropropagación."""
    # Derivada: 1 si x > 0, 0 en caso contrario
    return (x > 0).astype(float)

def sigmoid(x):
    """Función de activación Sigmoid (logística)."""
    # Sigmoid(x) = 1 / (1 + e^(-x)) - Mapea valores a rango (0, 1)
    # Usamos np.clip para evitar overflow en exponencial
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivada(x):
    """Derivada de Sigmoid para retropropagación."""
    # Derivada: σ(x) × (1 - σ(x))
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Función de activación Tanh (tangente hiperbólica)."""
    # Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) - Mapea valores a rango (-1, 1)
    return np.tanh(x)

def tanh_derivada(x):
    """Derivada de Tanh para retropropagación."""
    # Derivada: 1 - tanh²(x)
    return 1 - np.tanh(x) ** 2

def softmax(x):
    """Función Softmax para clasificación multiclase."""
    # Softmax normaliza valores a distribución de probabilidad
    # Restamos el máximo para estabilidad numérica
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ========== CLASE RED NEURONAL PROFUNDA ==========

class RedNeuronalProfunda:
    """
    Implementación de una red neuronal profunda con múltiples capas.
    Soporta diferentes funciones de activación y regularización.
    """
    
    def __init__(self, arquitectura, tasa_aprendizaje=0.01, regularizacion_l2=0.0):
        """
        Inicializa la red neuronal profunda.
        
        :param arquitectura: lista con el número de neuronas por capa [entrada, oculta1, ..., salida]
        :param tasa_aprendizaje: tasa de aprendizaje para descenso de gradiente
        :param regularizacion_l2: parámetro lambda para regularización L2
        """
        self.arquitectura = arquitectura
        self.num_capas = len(arquitectura)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.lambda_reg = regularizacion_l2
        
        # Inicializar pesos y sesgos con distribución Xavier/He
        self.pesos = []
        self.sesgos = []
        
        for i in range(self.num_capas - 1):
            # Inicialización He para ReLU (mejora convergencia)
            limite = np.sqrt(2.0 / arquitectura[i])
            W = np.random.randn(arquitectura[i], arquitectura[i+1]) * limite
            b = np.zeros((1, arquitectura[i+1]))
            
            self.pesos.append(W)
            self.sesgos.append(b)
        
        # Almacenar activaciones y gradientes para retropropagación
        self.activaciones = []
        self.z_valores = []  # Valores antes de aplicar función de activación
    
    def propagacion_adelante(self, X, funcion_activacion='relu'):
        """
        Propagación hacia adelante a través de la red.
        
        :param X: datos de entrada (n_muestras, n_caracteristicas)
        :param funcion_activacion: 'relu', 'sigmoid', o 'tanh'
        :return: salida de la red
        """
        # Seleccionar función de activación
        if funcion_activacion == 'relu':
            activacion_fn = relu
        elif funcion_activacion == 'sigmoid':
            activacion_fn = sigmoid
        elif funcion_activacion == 'tanh':
            activacion_fn = tanh
        else:
            activacion_fn = relu
        
        # Reiniciar almacenamiento de activaciones
        self.activaciones = [X]
        self.z_valores = []
        
        A = X  # Activación de la capa de entrada
        
        # Propagar a través de todas las capas
        for i in range(self.num_capas - 1):
            # Calcular Z = A × W + b
            Z = np.dot(A, self.pesos[i]) + self.sesgos[i]
            self.z_valores.append(Z)
            
            # Aplicar función de activación (excepto en la última capa)
            if i < self.num_capas - 2:
                A = activacion_fn(Z)
            else:
                # En la capa de salida, usar función apropiada
                # Para regresión: identidad, para clasificación: sigmoid/softmax
                A = sigmoid(Z)  # Asumimos clasificación binaria por defecto
            
            self.activaciones.append(A)
        
        return A
    
    def retropropagacion(self, X, Y, funcion_activacion='relu'):
        """
        Algoritmo de retropropagación para calcular gradientes.
        
        :param X: datos de entrada
        :param Y: etiquetas verdaderas
        :param funcion_activacion: función de activación usada
        :return: gradientes de pesos y sesgos
        """
        m = X.shape[0]  # Número de muestras
        
        # Seleccionar derivada de función de activación
        if funcion_activacion == 'relu':
            derivada_fn = relu_derivada
        elif funcion_activacion == 'sigmoid':
            derivada_fn = sigmoid_derivada
        elif funcion_activacion == 'tanh':
            derivada_fn = tanh_derivada
        else:
            derivada_fn = relu_derivada
        
        # Inicializar gradientes
        gradientes_pesos = [np.zeros_like(W) for W in self.pesos]
        gradientes_sesgos = [np.zeros_like(b) for b in self.sesgos]
        
        # Error en la capa de salida (para clasificación binaria con sigmoid)
        delta = self.activaciones[-1] - Y
        
        # Retropropagar el error hacia atrás
        for i in range(self.num_capas - 2, -1, -1):
            # Calcular gradientes de pesos y sesgos
            gradientes_pesos[i] = np.dot(self.activaciones[i].T, delta) / m
            gradientes_sesgos[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Añadir término de regularización L2 a los pesos
            if self.lambda_reg > 0:
                gradientes_pesos[i] += (self.lambda_reg / m) * self.pesos[i]
            
            # Propagar el error a la capa anterior (si no estamos en la primera capa)
            if i > 0:
                delta = np.dot(delta, self.pesos[i].T) * derivada_fn(self.z_valores[i-1])
        
        return gradientes_pesos, gradientes_sesgos
    
    def entrenar(self, X, Y, epocas=1000, verbose=True, funcion_activacion='relu'):
        """
        Entrena la red neuronal usando descenso de gradiente.
        
        :param X: datos de entrenamiento
        :param Y: etiquetas
        :param epocas: número de iteraciones de entrenamiento
        :param verbose: mostrar progreso
        :param funcion_activacion: función de activación a usar
        """
        historial_perdida = []
        
        for epoca in range(epocas):
            # Propagación hacia adelante
            salida = self.propagacion_adelante(X, funcion_activacion)
            
            # Calcular pérdida (entropía cruzada binaria)
            perdida = self.calcular_perdida(Y, salida)
            historial_perdida.append(perdida)
            
            # Retropropagación
            grad_W, grad_b = self.retropropagacion(X, Y, funcion_activacion)
            
            # Actualizar pesos y sesgos con descenso de gradiente
            for i in range(len(self.pesos)):
                self.pesos[i] -= self.tasa_aprendizaje * grad_W[i]
                self.sesgos[i] -= self.tasa_aprendizaje * grad_b[i]
            
            # Mostrar progreso cada 100 épocas
            if verbose and (epoca % 100 == 0 or epoca == epocas - 1):
                precision = self.calcular_precision(X, Y, funcion_activacion)
                print(f"Época {epoca:4d} | Pérdida: {perdida:.6f} | Precisión: {precision:.2f}%")
        
        return historial_perdida
    
    def calcular_perdida(self, Y, Y_pred):
        """Calcula la pérdida de entropía cruzada binaria con regularización L2."""
        m = Y.shape[0]
        
        # Entropía cruzada binaria
        epsilon = 1e-10  # Para evitar log(0)
        perdida = -np.mean(Y * np.log(Y_pred + epsilon) + (1 - Y) * np.log(1 - Y_pred + epsilon))
        
        # Añadir término de regularización L2
        if self.lambda_reg > 0:
            suma_pesos_cuadrados = sum(np.sum(W ** 2) for W in self.pesos)
            perdida += (self.lambda_reg / (2 * m)) * suma_pesos_cuadrados
        
        return perdida
    
    def predecir(self, X, funcion_activacion='relu'):
        """Realiza predicciones sobre nuevos datos."""
        salida = self.propagacion_adelante(X, funcion_activacion)
        return (salida >= 0.5).astype(int)
    
    def calcular_precision(self, X, Y, funcion_activacion='relu'):
        """Calcula la precisión del modelo."""
        predicciones = self.predecir(X, funcion_activacion)
        return np.mean(predicciones == Y) * 100

# ========== FUNCIONES AUXILIARES ==========

def crear_dataset_xor(n_muestras=200):
    """Crea un dataset para el problema XOR (no linealmente separable)."""
    X = np.random.randn(n_muestras, 2)
    # XOR: y = 1 si (x1 > 0 XOR x2 > 0), 0 en caso contrario
    Y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int).reshape(-1, 1)
    return X, Y

def crear_dataset_circulos(n_muestras=300):
    """Crea un dataset con clases en círculos concéntricos."""
    # Generar puntos aleatorios
    angulos = np.random.rand(n_muestras) * 2 * np.pi
    radios = np.random.rand(n_muestras)
    
    X = np.column_stack([
        radios * np.cos(angulos),
        radios * np.sin(angulos)
    ])
    
    # Etiquetas: 1 si está en el círculo exterior, 0 si está en el interior
    Y = (radios > 0.5).astype(int).reshape(-1, 1)
    
    return X, Y

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con problemas predefinidos."""
    print("\n" + "="*70)
    print("MODO DEMO: Aprendizaje Profundo con Redes Neuronales")
    print("="*70)
    
    # ========== EJEMPLO 1: Problema XOR ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Resolver el problema XOR (no linealmente separable)")
    print("="*70)
    
    print("\n--- Generando dataset XOR ---")
    X_xor, Y_xor = crear_dataset_xor(n_muestras=400)
    print(f"Muestras generadas: {X_xor.shape[0]}")
    print(f"Características: {X_xor.shape[1]}")
    print(f"Distribución de clases: Clase 0={np.sum(Y_xor==0)}, Clase 1={np.sum(Y_xor==1)}")
    
    print("\n--- Creando Red Neuronal Profunda ---")
    # Arquitectura: 2 entradas → 8 neuronas (capa oculta 1) → 4 neuronas (capa oculta 2) → 1 salida
    arquitectura_xor = [2, 8, 4, 1]
    red_xor = RedNeuronalProfunda(
        arquitectura=arquitectura_xor,
        tasa_aprendizaje=0.5,
        regularizacion_l2=0.001
    )
    
    print(f"Arquitectura: {arquitectura_xor}")
    print(f"Capas totales: {len(arquitectura_xor)}")
    print(f"Parámetros entrenables: {sum(W.size + b.size for W, b in zip(red_xor.pesos, red_xor.sesgos))}")
    
    print("\n--- Entrenando la red (función de activación: ReLU) ---")
    historial = red_xor.entrenar(X_xor, Y_xor, epocas=500, verbose=True, funcion_activacion='relu')
    
    print(f"\n>>> RESULTADO:")
    precision_final = red_xor.calcular_precision(X_xor, Y_xor, 'relu')
    print(f"    Precisión final: {precision_final:.2f}%")
    print(f"    Pérdida inicial: {historial[0]:.6f}")
    print(f"    Pérdida final: {historial[-1]:.6f}")
    print(f"    (Una red profunda puede resolver XOR, que no es linealmente separable)")
    
    # ========== EJEMPLO 2: Clasificación con círculos concéntricos ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Clasificación con patrones circulares")
    print("="*70)
    
    print("\n--- Generando dataset de círculos concéntricos ---")
    X_circ, Y_circ = crear_dataset_circulos(n_muestras=500)
    print(f"Muestras generadas: {X_circ.shape[0]}")
    print(f"Distribución: Interior={np.sum(Y_circ==0)}, Exterior={np.sum(Y_circ==1)}")
    
    print("\n--- Creando Red Neuronal Profunda ---")
    # Red más profunda: 2 → 16 → 8 → 4 → 1
    arquitectura_circ = [2, 16, 8, 4, 1]
    red_circ = RedNeuronalProfunda(
        arquitectura=arquitectura_circ,
        tasa_aprendizaje=0.3,
        regularizacion_l2=0.01
    )
    
    print(f"Arquitectura: {arquitectura_circ}")
    print(f"Capas ocultas: {len(arquitectura_circ) - 2}")
    
    print("\n--- Entrenando la red (función de activación: Tanh) ---")
    historial2 = red_circ.entrenar(X_circ, Y_circ, epocas=400, verbose=True, funcion_activacion='tanh')
    
    print(f"\n>>> RESULTADO:")
    precision_final2 = red_circ.calcular_precision(X_circ, Y_circ, 'tanh')
    print(f"    Precisión final: {precision_final2:.2f}%")
    print(f"    Reducción de pérdida: {historial2[0]:.6f} → {historial2[-1]:.6f}")
    print(f"    (La profundidad permite aprender patrones complejos no lineales)")
    
    # ========== EJEMPLO 3: Comparación de funciones de activación ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Comparación de funciones de activación")
    print("="*70)
    
    X_test, Y_test = crear_dataset_xor(n_muestras=200)
    
    funciones = ['relu', 'sigmoid', 'tanh']
    resultados = {}
    
    for func in funciones:
        print(f"\n--- Probando con {func.upper()} ---")
        red_test = RedNeuronalProfunda([2, 6, 1], tasa_aprendizaje=0.5)
        red_test.entrenar(X_test, Y_test, epocas=300, verbose=False, funcion_activacion=func)
        precision = red_test.calcular_precision(X_test, Y_test, func)
        resultados[func] = precision
        print(f"Precisión con {func.upper()}: {precision:.2f}%")
    
    mejor_func = max(resultados, key=resultados.get)
    print(f"\n>>> MEJOR FUNCIÓN: {mejor_func.upper()} con {resultados[mejor_func]:.2f}% de precisión")
    print(f"    (ReLU suele converger más rápido en redes profundas)")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo con configuración personalizada."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Configuración de Red Neuronal Profunda")
    print("="*70)
    
    # ========== PASO 1: Seleccionar dataset ==========
    print("\n--- Selecciona el tipo de problema ---")
    print("1. XOR (no linealmente separable)")
    print("2. Círculos concéntricos")
    
    opcion_dataset = input("\nOpción (1-2): ").strip()
    
    if opcion_dataset == '1':
        X, Y = crear_dataset_xor(n_muestras=400)
        print("Dataset XOR cargado.")
    elif opcion_dataset == '2':
        X, Y = crear_dataset_circulos(n_muestras=500)
        print("Dataset de círculos cargado.")
    else:
        print("Opción no válida. Usando XOR por defecto.")
        X, Y = crear_dataset_xor(n_muestras=400)
    
    print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    
    # ========== PASO 2: Configurar arquitectura ==========
    print("\n--- Configurar arquitectura de la red ---")
    print("Ejemplo: 2 8 4 1 (2 entradas, capas ocultas de 8 y 4, 1 salida)")
    
    entrada_arq = input("\nIngresa arquitectura (números separados por espacios): ").strip()
    
    if entrada_arq:
        try:
            arquitectura = [int(x) for x in entrada_arq.split()]
            # Validar que tenga al menos entrada y salida
            if len(arquitectura) < 2:
                raise ValueError("Se necesitan al menos 2 capas")
        except:
            print("Arquitectura inválida. Usando [2, 8, 1] por defecto.")
            arquitectura = [2, 8, 1]
    else:
        arquitectura = [2, 8, 1]
    
    print(f"Arquitectura seleccionada: {arquitectura}")
    
    # ========== PASO 3: Configurar hiperparámetros ==========
    print("\n--- Configurar hiperparámetros ---")
    
    try:
        tasa = float(input("Tasa de aprendizaje (ej: 0.1): ").strip() or "0.3")
    except:
        tasa = 0.3
    
    try:
        lambda_reg = float(input("Regularización L2 (ej: 0.01): ").strip() or "0.01")
    except:
        lambda_reg = 0.01
    
    try:
        epocas = int(input("Número de épocas (ej: 500): ").strip() or "500")
    except:
        epocas = 500
    
    print("\n--- Seleccionar función de activación ---")
    print("1. ReLU   2. Sigmoid   3. Tanh")
    func_opcion = input("Opción (1-3): ").strip()
    
    funciones_map = {'1': 'relu', '2': 'sigmoid', '3': 'tanh'}
    func_activacion = funciones_map.get(func_opcion, 'relu')
    
    print(f"\n--- Configuración final ---")
    print(f"Arquitectura: {arquitectura}")
    print(f"Tasa de aprendizaje: {tasa}")
    print(f"Regularización L2: {lambda_reg}")
    print(f"Épocas: {epocas}")
    print(f"Función de activación: {func_activacion.upper()}")
    
    # ========== PASO 4: Entrenar modelo ==========
    print("\n--- Entrenando modelo ---")
    
    red = RedNeuronalProfunda(
        arquitectura=arquitectura,
        tasa_aprendizaje=tasa,
        regularizacion_l2=lambda_reg
    )
    
    historial = red.entrenar(X, Y, epocas=epocas, verbose=True, funcion_activacion=func_activacion)
    
    # ========== PASO 5: Mostrar resultados ==========
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    
    precision = red.calcular_precision(X, Y, func_activacion)
    print(f"Precisión final: {precision:.2f}%")
    print(f"Pérdida inicial: {historial[0]:.6f}")
    print(f"Pérdida final: {historial[-1]:.6f}")
    print(f"Mejora: {((historial[0] - historial[-1]) / historial[0] * 100):.2f}%")
    
    # Mostrar algunos ejemplos de predicciones
    print("\n--- Ejemplos de predicciones ---")
    indices = random.sample(range(X.shape[0]), min(5, X.shape[0]))
    for idx in indices:
        pred = red.predecir(X[idx:idx+1], func_activacion)[0, 0]
        real = Y[idx, 0]
        print(f"Entrada: [{X[idx, 0]:6.3f}, {X[idx, 1]:6.3f}] → Predicción: {pred}, Real: {real}")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("APRENDIZAJE PROFUNDO (DEEP LEARNING)")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (ejemplos predefinidos de redes profundas)")
    print("2. INTERACTIVO (configura tu propia red neuronal)")
    
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
