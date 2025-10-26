"""
005-E2-redes_neuronales.py
--------------------------------
Este script implementa Redes Neuronales Artificiales básicas (perceptrón multicapa):
- Construye redes neuronales con capas de entrada, ocultas y de salida
- Implementa la propagación hacia adelante (forward propagation)
- Calcula funciones de activación y sus derivadas
- Aplica retropropagación del error para ajustar pesos
- Entrena la red mediante descenso de gradiente
- Evalúa el rendimiento con métricas de error y precisión
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente entrenamiento en problemas clásicos (AND, OR, XOR)
2. INTERACTIVO: permite diseñar arquitecturas personalizadas y entrenar con datos propios

Autor: Alejandro Aguirre Díaz
"""

import numpy as np

# ========== FUNCIONES DE ACTIVACIÓN ==========

def sigmoide(x):
    """
    Función de activación sigmoide.
    Mapea cualquier valor de entrada a un rango entre 0 y 1.
    """
    # Usar clip para evitar overflow en exp()
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def derivada_sigmoide(x):
    """
    Derivada de la función sigmoide.
    Útil para la retropropagación del error.
    """
    # Derivada: f'(x) = f(x) * (1 - f(x))
    s = sigmoide(x)
    return s * (1 - s)

def escalon(x):
    """
    Función escalón (umbral binario).
    Usada en perceptrones simples.
    """
    return (x >= 0).astype(int)

# ========== CLASE RED NEURONAL ==========

class RedNeuronal:
    """
    Implementación de una red neuronal artificial con una capa oculta.
    Estructura: capa de entrada → capa oculta → capa de salida
    """
    
    def __init__(self, neuronas_entrada, neuronas_ocultas, neuronas_salida, tasa_aprendizaje=0.5):
        """
        Inicializa la arquitectura de la red neuronal.
        
        :param neuronas_entrada: número de neuronas en la capa de entrada
        :param neuronas_ocultas: número de neuronas en la capa oculta
        :param neuronas_salida: número de neuronas en la capa de salida
        :param tasa_aprendizaje: velocidad de ajuste de los pesos (learning rate)
        """
        self.neuronas_entrada = neuronas_entrada
        self.neuronas_ocultas = neuronas_ocultas
        self.neuronas_salida = neuronas_salida
        self.tasa_aprendizaje = tasa_aprendizaje
        
        # ========== INICIALIZACIÓN DE PESOS Y SESGOS ==========
        # Pesos de la capa de entrada a la capa oculta (matriz de conexiones)
        # Inicialización aleatoria pequeña para romper simetría
        self.pesos_entrada_oculta = np.random.randn(self.neuronas_entrada, self.neuronas_ocultas) * 0.5
        self.sesgo_oculta = np.zeros((1, self.neuronas_ocultas))
        
        # Pesos de la capa oculta a la capa de salida
        self.pesos_oculta_salida = np.random.randn(self.neuronas_ocultas, self.neuronas_salida) * 0.5
        self.sesgo_salida = np.zeros((1, self.neuronas_salida))
        
        # Variables para almacenar activaciones durante propagación hacia adelante
        self.activacion_entrada = None
        self.suma_oculta = None  # Entrada neta de la capa oculta (antes de activación)
        self.activacion_oculta = None
        self.suma_salida = None  # Entrada neta de la capa de salida
        self.activacion_salida = None
    
    def propagacion_adelante(self, X):
        """
        Propagación hacia adelante: calcula la salida de la red para una entrada dada.
        
        :param X: datos de entrada (puede ser un vector o matriz de muestras)
        :return: salida de la red neuronal
        """
        # ========== CAPA DE ENTRADA → CAPA OCULTA ==========
        # Guardar la activación de entrada (son las entradas mismas)
        self.activacion_entrada = X
        
        # Calcular entrada neta de la capa oculta: Z_oculta = X × W1 + b1
        self.suma_oculta = np.dot(X, self.pesos_entrada_oculta) + self.sesgo_oculta
        
        # Aplicar función de activación (sigmoide) a la capa oculta
        self.activacion_oculta = sigmoide(self.suma_oculta)
        
        # ========== CAPA OCULTA → CAPA DE SALIDA ==========
        # Calcular entrada neta de la capa de salida: Z_salida = A_oculta × W2 + b2
        self.suma_salida = np.dot(self.activacion_oculta, self.pesos_oculta_salida) + self.sesgo_salida
        
        # Aplicar función de activación (sigmoide) a la capa de salida
        self.activacion_salida = sigmoide(self.suma_salida)
        
        return self.activacion_salida
    
    def retropropagacion(self, X, Y):
        """
        Retropropagación del error: ajusta los pesos para minimizar el error.
        
        :param X: datos de entrada
        :param Y: etiquetas verdaderas (valores esperados)
        """
        m = X.shape[0]  # Número de muestras
        
        # ========== CALCULAR ERROR EN LA CAPA DE SALIDA ==========
        # Error = Salida predicha - Salida esperada
        error_salida = self.activacion_salida - Y
        
        # Gradiente de la capa de salida (aplicando la regla de la cadena)
        # δ_salida = error × derivada_sigmoide(Z_salida)
        delta_salida = error_salida * derivada_sigmoide(self.suma_salida)
        
        # ========== CALCULAR ERROR EN LA CAPA OCULTA ==========
        # Propagar el error hacia atrás: error_oculta = δ_salida × W2^T
        error_oculta = np.dot(delta_salida, self.pesos_oculta_salida.T)
        
        # Gradiente de la capa oculta
        # δ_oculta = error_oculta × derivada_sigmoide(Z_oculta)
        delta_oculta = error_oculta * derivada_sigmoide(self.suma_oculta)
        
        # ========== ACTUALIZAR PESOS Y SESGOS ==========
        # Gradientes de los pesos (capa oculta → salida)
        gradiente_pesos_oculta_salida = np.dot(self.activacion_oculta.T, delta_salida) / m
        gradiente_sesgo_salida = np.sum(delta_salida, axis=0, keepdims=True) / m
        
        # Gradientes de los pesos (entrada → oculta)
        gradiente_pesos_entrada_oculta = np.dot(self.activacion_entrada.T, delta_oculta) / m
        gradiente_sesgo_oculta = np.sum(delta_oculta, axis=0, keepdims=True) / m
        
        # Actualizar pesos usando descenso de gradiente: W_nuevo = W_viejo - α × gradiente
        self.pesos_oculta_salida -= self.tasa_aprendizaje * gradiente_pesos_oculta_salida
        self.sesgo_salida -= self.tasa_aprendizaje * gradiente_sesgo_salida
        
        self.pesos_entrada_oculta -= self.tasa_aprendizaje * gradiente_pesos_entrada_oculta
        self.sesgo_oculta -= self.tasa_aprendizaje * gradiente_sesgo_oculta
    
    def entrenar(self, X, Y, epocas=1000, verbose=True):
        """
        Entrena la red neuronal usando el algoritmo de retropropagación.
        
        :param X: datos de entrenamiento
        :param Y: etiquetas de entrenamiento
        :param epocas: número de iteraciones de entrenamiento
        :param verbose: si True, muestra el progreso del entrenamiento
        """
        historial_error = []
        
        for epoca in range(epocas):
            # Propagación hacia adelante para calcular salida
            salida = self.propagacion_adelante(X)
            
            # Calcular error cuadrático medio (MSE)
            error = np.mean((Y - salida) ** 2)
            historial_error.append(error)
            
            # Retropropagación para ajustar pesos
            self.retropropagacion(X, Y)
            
            # Mostrar progreso cada 100 épocas
            if verbose and (epoca % 100 == 0 or epoca == epocas - 1):
                print(f"Época {epoca:4d} | Error MSE: {error:.6f}")
        
        return historial_error
    
    def predecir(self, X, umbral=0.5):
        """
        Realiza predicciones sobre nuevos datos.
        
        :param X: datos de entrada
        :param umbral: umbral de decisión para clasificación binaria
        :return: predicciones binarias (0 o 1)
        """
        # Obtener salida de la red
        salida = self.propagacion_adelante(X)
        
        # Aplicar umbral para clasificación binaria
        return (salida >= umbral).astype(int)
    
    def calcular_precision(self, X, Y):
        """
        Calcula la precisión de las predicciones.
        
        :param X: datos de prueba
        :param Y: etiquetas verdaderas
        :return: porcentaje de precisión
        """
        predicciones = self.predecir(X)
        return np.mean(predicciones == Y) * 100

# ========== FUNCIONES AUXILIARES ==========

def crear_datos_and():
    """Crea el dataset para la compuerta lógica AND."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [0], [0], [1]])
    return X, Y

def crear_datos_or():
    """Crea el dataset para la compuerta lógica OR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [1]])
    return X, Y

def crear_datos_xor():
    """Crea el dataset para la compuerta lógica XOR (no linealmente separable)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    return X, Y

def mostrar_tabla_verdad(X, Y, predicciones, nombre_operacion):
    """
    Muestra una tabla de verdad comparando predicciones con valores reales.
    
    :param X: entradas
    :param Y: salidas esperadas
    :param predicciones: salidas predichas por la red
    :param nombre_operacion: nombre de la operación lógica
    """
    print(f"\n--- Tabla de Verdad: {nombre_operacion} ---")
    print("X1  X2  | Esperado | Predicho | ¿Correcto?")
    print("-" * 45)
    
    for i in range(len(X)):
        x1, x2 = X[i]
        esperado = Y[i, 0]
        predicho = predicciones[i, 0]
        correcto = "✓" if esperado == predicho else "✗"
        print(f" {x1}   {x2}  |    {esperado}     |    {predicho}     |    {correcto}")

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con problemas lógicos clásicos."""
    print("\n" + "="*70)
    print("MODO DEMO: Redes Neuronales - Compuertas Lógicas")
    print("="*70)
    
    # ========== EJEMPLO 1: Compuerta AND ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Aprender la compuerta lógica AND")
    print("="*70)
    
    X_and, Y_and = crear_datos_and()
    print("\n--- Dataset AND ---")
    print("Entradas (X):\n", X_and)
    print("Salidas (Y):\n", Y_and.T)
    
    print("\n--- Creando y entrenando red neuronal ---")
    print("Arquitectura: 2 entradas → 2 neuronas ocultas → 1 salida")
    
    red_and = RedNeuronal(neuronas_entrada=2, neuronas_ocultas=2, neuronas_salida=1, tasa_aprendizaje=0.5)
    historial_and = red_and.entrenar(X_and, Y_and, epocas=1000, verbose=True)
    
    predicciones_and = red_and.predecir(X_and)
    precision_and = red_and.calcular_precision(X_and, Y_and)
    
    mostrar_tabla_verdad(X_and, Y_and, predicciones_and, "AND")
    print(f"\n>>> PRECISIÓN: {precision_and:.2f}%")
    print(f"    Error inicial: {historial_and[0]:.6f}")
    print(f"    Error final: {historial_and[-1]:.6f}")
    
    # ========== EJEMPLO 2: Compuerta OR ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Aprender la compuerta lógica OR")
    print("="*70)
    
    X_or, Y_or = crear_datos_or()
    print("\n--- Dataset OR ---")
    print("Entradas (X):\n", X_or)
    print("Salidas (Y):\n", Y_or.T)
    
    print("\n--- Entrenando red neuronal ---")
    red_or = RedNeuronal(neuronas_entrada=2, neuronas_ocultas=2, neuronas_salida=1, tasa_aprendizaje=0.5)
    historial_or = red_or.entrenar(X_or, Y_or, epocas=1000, verbose=True)
    
    predicciones_or = red_or.predecir(X_or)
    precision_or = red_or.calcular_precision(X_or, Y_or)
    
    mostrar_tabla_verdad(X_or, Y_or, predicciones_or, "OR")
    print(f"\n>>> PRECISIÓN: {precision_or:.2f}%")
    print(f"    (La red aprendió correctamente la función OR)")
    
    # ========== EJEMPLO 3: Compuerta XOR (problema clásico) ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Aprender la compuerta lógica XOR (no linealmente separable)")
    print("="*70)
    
    X_xor, Y_xor = crear_datos_xor()
    print("\n--- Dataset XOR ---")
    print("Entradas (X):\n", X_xor)
    print("Salidas (Y):\n", Y_xor.T)
    print("\nNOTA: XOR no es linealmente separable, requiere una capa oculta.")
    
    print("\n--- Entrenando red neuronal con 4 neuronas ocultas ---")
    red_xor = RedNeuronal(neuronas_entrada=2, neuronas_ocultas=4, neuronas_salida=1, tasa_aprendizaje=1.0)
    historial_xor = red_xor.entrenar(X_xor, Y_xor, epocas=2000, verbose=True)
    
    predicciones_xor = red_xor.predecir(X_xor)
    precision_xor = red_xor.calcular_precision(X_xor, Y_xor)
    
    mostrar_tabla_verdad(X_xor, Y_xor, predicciones_xor, "XOR")
    print(f"\n>>> PRECISIÓN: {precision_xor:.2f}%")
    print(f"    Error inicial: {historial_xor[0]:.6f}")
    print(f"    Error final: {historial_xor[-1]:.6f}")
    print(f"    (¡La red multicapa resolvió XOR exitosamente!)")
    
    # ========== DEMOSTRACIÓN: Importancia de la capa oculta ==========
    print("\n" + "="*70)
    print("DEMOSTRACIÓN: ¿Por qué es importante la capa oculta?")
    print("="*70)
    
    print("\n--- Probando diferentes cantidades de neuronas ocultas en XOR ---")
    configuraciones = [1, 2, 3, 4, 6, 8]
    
    for n_ocultas in configuraciones:
        red_test = RedNeuronal(neuronas_entrada=2, neuronas_ocultas=n_ocultas, neuronas_salida=1, tasa_aprendizaje=1.0)
        red_test.entrenar(X_xor, Y_xor, epocas=2000, verbose=False)
        precision = red_test.calcular_precision(X_xor, Y_xor)
        print(f"Neuronas ocultas: {n_ocultas} → Precisión: {precision:.2f}%")
    
    print("\n>>> CONCLUSIÓN: Más neuronas ocultas permiten representar funciones más complejas")
    print("    pero también pueden causar sobreajuste en datasets pequeños.")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo con configuración personalizada."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Diseña tu propia Red Neuronal")
    print("="*70)
    
    # ========== PASO 1: Seleccionar problema ==========
    print("\n--- Selecciona el problema a resolver ---")
    print("1. AND (compuerta lógica AND)")
    print("2. OR (compuerta lógica OR)")
    print("3. XOR (compuerta lógica XOR - no linealmente separable)")
    
    opcion_problema = input("\nOpción (1-3): ").strip()
    
    if opcion_problema == '1':
        X, Y = crear_datos_and()
        nombre_problema = "AND"
    elif opcion_problema == '2':
        X, Y = crear_datos_or()
        nombre_problema = "OR"
    elif opcion_problema == '3':
        X, Y = crear_datos_xor()
        nombre_problema = "XOR"
    else:
        print("Opción no válida. Usando XOR por defecto.")
        X, Y = crear_datos_xor()
        nombre_problema = "XOR"
    
    print(f"\nProblema seleccionado: {nombre_problema}")
    print("Datos de entrenamiento:")
    print("X =\n", X)
    print("Y =\n", Y.T)
    
    # ========== PASO 2: Configurar arquitectura ==========
    print("\n--- Configurar arquitectura de la red ---")
    
    try:
        n_ocultas = int(input(f"Número de neuronas en la capa oculta (recomendado: 2-8): ").strip() or "4")
    except:
        n_ocultas = 4
    
    try:
        tasa = float(input("Tasa de aprendizaje (recomendado: 0.1-1.0): ").strip() or "0.5")
    except:
        tasa = 0.5
    
    try:
        epocas = int(input("Número de épocas de entrenamiento (ej: 1000-5000): ").strip() or "2000")
    except:
        epocas = 2000
    
    print(f"\n--- Configuración ---")
    print(f"Problema: {nombre_problema}")
    print(f"Arquitectura: 2 → {n_ocultas} → 1")
    print(f"Tasa de aprendizaje: {tasa}")
    print(f"Épocas: {epocas}")
    
    # ========== PASO 3: Entrenar red ==========
    print("\n--- Entrenando red neuronal ---")
    
    red = RedNeuronal(neuronas_entrada=2, neuronas_ocultas=n_ocultas, neuronas_salida=1, tasa_aprendizaje=tasa)
    historial = red.entrenar(X, Y, epocas=epocas, verbose=True)
    
    # ========== PASO 4: Evaluar resultados ==========
    print("\n" + "="*70)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("="*70)
    
    predicciones = red.predecir(X)
    precision = red.calcular_precision(X, Y)
    
    mostrar_tabla_verdad(X, Y, predicciones, nombre_problema)
    
    print(f"\n--- Métricas de rendimiento ---")
    print(f"Precisión: {precision:.2f}%")
    print(f"Error MSE inicial: {historial[0]:.6f}")
    print(f"Error MSE final: {historial[-1]:.6f}")
    print(f"Reducción de error: {((historial[0] - historial[-1]) / historial[0] * 100):.2f}%")
    
    # Mostrar salidas continuas (antes del umbral)
    print("\n--- Salidas de la red (valores continuos) ---")
    salidas = red.propagacion_adelante(X)
    for i in range(len(X)):
        print(f"Entrada: {X[i]} → Salida: {salidas[i, 0]:.4f} → Predicción: {predicciones[i, 0]}")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("REDES NEURONALES ARTIFICIALES (Perceptrón Multicapa)")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (compuertas lógicas: AND, OR, XOR)")
    print("2. INTERACTIVO (diseña tu propia red neuronal)")
    
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
