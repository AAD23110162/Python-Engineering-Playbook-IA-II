"""
043-E2-retropropagacion_del_error.py
--------------------------------
Este script presenta la Retropropagación del Error (Backpropagation):
- Deriva gradientes capa a capa usando la regla de la cadena.
- Implementa actualización de pesos con descenso de gradiente y variantes (SGD, Momentum, Adam).
- Discute problemas de vanishing/exploding gradients y técnicas de mitigación.
- Muestra cómo los gradientes se propagan desde la salida hasta la entrada.

El programa puede ejecutarse en dos modos:
1. DEMO: entrenamiento de MLP con diferentes optimizadores y análisis de gradientes.
2. INTERACTIVO: ajuste de tasas de aprendizaje, optimizadores y tamaños de lote.

Autor: Alejandro Aguirre Díaz
"""

import random
import math
from typing import List, Tuple, Dict, Callable

# ============================================================================
# Funciones de Activación y sus Derivadas
# ============================================================================

def sigmoide(x: float) -> float:
    """
    Función sigmoide: σ(x) = 1 / (1 + e^(-x))
    Rango de salida: [0, 1]
    """
    # Prevenir overflow numérico
    if x < -500:
        return 0.0
    elif x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def derivada_sigmoide(y: float) -> float:
    """
    Derivada de sigmoide: σ'(x) = σ(x) * (1 - σ(x))
    Recibe y = σ(x) ya calculado.
    
    PROBLEMA: Para valores cercanos a 0 o 1, la derivada es muy pequeña
    → Gradiente que desaparece (vanishing gradient)
    """
    return y * (1.0 - y)

def tanh(x: float) -> float:
    """
    Tangente hiperbólica: tanh(x)
    Rango de salida: [-1, 1]
    """
    if x < -500:
        return -1.0
    elif x > 500:
        return 1.0
    return math.tanh(x)

def derivada_tanh(y: float) -> float:
    """
    Derivada de tanh: tanh'(x) = 1 - tanh²(x)
    Mejor que sigmoide pero aún sufre de vanishing gradient.
    """
    return 1.0 - y * y

def relu(x: float) -> float:
    """
    ReLU: max(0, x)
    Ventajas: No sufre vanishing gradient, computacionalmente eficiente.
    """
    return max(0.0, x)

def derivada_relu(x: float) -> float:
    """
    Derivada de ReLU: 1 si x > 0, 0 en otro caso.
    Problema: "Dying ReLU" cuando x <= 0.
    """
    return 1.0 if x > 0 else 0.0

# ============================================================================
# Clase Red Neuronal con Retropropagación Detallada
# ============================================================================

class RedNeuronalBackprop:
    """
    Red neuronal que implementa retropropagación del error con diferentes
    optimizadores y registro detallado de gradientes.
    """
    
    def __init__(self, 
                 arquitectura: List[int],
                 funcion_activacion: str = 'sigmoide',
                 optimizador: str = 'sgd',
                 tasa_aprendizaje: float = 0.1,
                 momento: float = 0.9,
                 beta1: float = 0.9,
                 beta2: float = 0.999):
        """
        Inicializa la red neuronal.
        
        Args:
            arquitectura: [n_entrada, n_oculta1, ..., n_salida]
            funcion_activacion: 'sigmoide', 'tanh' o 'relu'
            optimizador: 'sgd', 'momentum' o 'adam'
            tasa_aprendizaje: Tasa de aprendizaje (η o α)
            momento: Factor de momento para momentum (β)
            beta1, beta2: Parámetros para Adam
        """
        self.arquitectura = arquitectura
        self.num_capas = len(arquitectura)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.optimizador = optimizador
        
        # Seleccionar función de activación
        if funcion_activacion == 'tanh':
            self.activacion = tanh
            self.derivada_activacion = derivada_tanh
        elif funcion_activacion == 'relu':
            self.activacion = relu
            self.derivada_activacion = derivada_relu
        else:
            self.activacion = sigmoide
            self.derivada_activacion = derivada_sigmoide
        
        # Inicializar pesos y sesgos con Xavier/Glorot
        self.pesos = []
        self.sesgos = []
        
        for i in range(self.num_capas - 1):
            n_in = arquitectura[i]
            n_out = arquitectura[i + 1]
            
            # Xavier: límite = sqrt(6 / (n_in + n_out))
            limite = math.sqrt(6.0 / (n_in + n_out))
            
            # Pesos: matriz n_out × n_in
            pesos_capa = [[random.uniform(-limite, limite) for _ in range(n_in)]
                         for _ in range(n_out)]
            self.pesos.append(pesos_capa)
            
            # Sesgos: vector n_out
            sesgos_capa = [random.uniform(-limite, limite) for _ in range(n_out)]
            self.sesgos.append(sesgos_capa)
        
        # Variables para almacenar durante forward/backward pass
        self.activaciones = []  # Salidas de cada capa
        self.z_valores = []     # Valores pre-activación (z = Wx + b)
        self.gradientes_pesos = []  # Gradientes de los pesos
        self.gradientes_sesgos = []  # Gradientes de los sesgos
        
        # Variables para optimizadores con momento
        self.momento_pesos = None
        self.momento_sesgos = None
        
        # Variables para Adam
        self.m_pesos = None  # Primer momento (media)
        self.v_pesos = None  # Segundo momento (varianza)
        self.m_sesgos = None
        self.v_sesgos = None
        self.t = 0  # Contador de iteraciones
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        
        # Parámetro de momento
        self.momento_factor = momento
        
        # Inicializar estructuras para optimizadores
        self._inicializar_optimizador()
    
    def _inicializar_optimizador(self):
        """Inicializa estructuras según el optimizador elegido."""
        if self.optimizador == 'momentum':
            # Inicializar velocidades para momento
            self.momento_pesos = []
            self.momento_sesgos = []
            
            for i in range(self.num_capas - 1):
                # Velocidades iniciales en cero
                n_out = len(self.pesos[i])
                n_in = len(self.pesos[i][0])
                
                vel_pesos = [[0.0 for _ in range(n_in)] for _ in range(n_out)]
                vel_sesgos = [0.0 for _ in range(n_out)]
                
                self.momento_pesos.append(vel_pesos)
                self.momento_sesgos.append(vel_sesgos)
        
        elif self.optimizador == 'adam':
            # Inicializar momentos para Adam
            self.m_pesos = []
            self.v_pesos = []
            self.m_sesgos = []
            self.v_sesgos = []
            
            for i in range(self.num_capas - 1):
                n_out = len(self.pesos[i])
                n_in = len(self.pesos[i][0])
                
                # Primeros momentos (media de gradientes)
                m_p = [[0.0 for _ in range(n_in)] for _ in range(n_out)]
                m_b = [0.0 for _ in range(n_out)]
                
                # Segundos momentos (varianza de gradientes)
                v_p = [[0.0 for _ in range(n_in)] for _ in range(n_out)]
                v_b = [0.0 for _ in range(n_out)]
                
                self.m_pesos.append(m_p)
                self.v_pesos.append(v_p)
                self.m_sesgos.append(m_b)
                self.v_sesgos.append(v_b)
    
    def forward(self, entradas: List[float]) -> List[float]:
        """
        Propagación hacia adelante.
        Guarda valores intermedios necesarios para backpropagation.
        
        Returns:
            Salida de la red
        """
        # Guardar entrada como primera activación
        self.activaciones = [entradas[:]]
        self.z_valores = []
        
        # Propagar a través de las capas
        for capa in range(self.num_capas - 1):
            act_anterior = self.activaciones[-1]
            z_capa = []  # Valores pre-activación
            act_capa = []  # Valores post-activación
            
            # Para cada neurona en esta capa
            for neurona in range(len(self.pesos[capa])):
                # Calcular z = Σ(w * x) + b
                z = self.sesgos[capa][neurona]
                for j in range(len(act_anterior)):
                    z += self.pesos[capa][neurona][j] * act_anterior[j]
                
                z_capa.append(z)
                
                # Aplicar función de activación: a = σ(z)
                a = self.activacion(z)
                act_capa.append(a)
            
            # Guardar para backprop
            self.z_valores.append(z_capa)
            self.activaciones.append(act_capa)
        
        return self.activaciones[-1]
    
    def backward(self, salida_deseada: List[float]) -> float:
        """
        Retropropagación del error.
        Calcula gradientes usando la regla de la cadena.
        
        ALGORITMO DE BACKPROPAGATION:
        1. Calcular error en capa de salida: δ^L = (a^L - y) ⊙ σ'(z^L)
        2. Propagar error hacia atrás: δ^l = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^l)
        3. Calcular gradientes: ∂C/∂W^l = δ^l (a^(l-1))^T
        4. Calcular gradientes de sesgos: ∂C/∂b^l = δ^l
        
        Returns:
            Error cuadrático medio
        """
        # Obtener salida de la red (lo que predijo en forward)
        salida_red = self.activaciones[-1]
        
        # Lista para almacenar deltas (errores) de cada capa
        # Los deltas representan cuánto contribuye cada neurona al error total
        deltas = []
        
        # ============================================================
        # PASO 1: Calcular delta de la capa de salida
        # ============================================================
        # δ^L = (a^L - y) ⊙ σ'(z^L)
        # donde ⊙ es producto elemento a elemento (Hadamard)
        delta_salida = []
        for i in range(len(salida_red)):
            # Calcular diferencia entre salida esperada y obtenida
            # Error positivo: la red predijo menos de lo esperado
            # Error negativo: la red predijo más de lo esperado
            error = salida_deseada[i] - salida_red[i]
            
            # Calcular gradiente de la función de activación
            # Indica qué tan sensible es la salida a cambios en z
            grad_act = self.derivada_activacion(salida_red[i])
            
            # Delta = -error * gradiente (negativo porque minimizamos)
            # Nota: usamos -error porque derivamos la función de pérdida L = 1/2(y - ŷ)²
            # La derivada de L respecto a ŷ es -(y - ŷ)
            delta = -error * grad_act
            delta_salida.append(delta)
        
        # Guardar deltas de la capa de salida
        deltas.append(delta_salida)
        
        # ============================================================
        # PASO 2: Retropropagar deltas a capas ocultas
        # ============================================================
        # Aplicamos la REGLA DE LA CADENA para propagar el error hacia atrás
        # δ^l = ((W^(l+1))^T δ^(l+1)) ⊙ σ'(z^l)
        # Recorremos desde la penúltima capa hacia atrás (hacia la entrada)
        for capa in range(self.num_capas - 2, 0, -1):
            delta_capa = []
            
            # Para cada neurona en esta capa
            for j in range(len(self.activaciones[capa])):
                # Sumar contribución de todas las neuronas de la capa siguiente
                # Esto implementa (W^(l+1))^T δ^(l+1)
                # El error de cada neurona depende de:
                # 1. Los errores de las neuronas conectadas en la capa siguiente
                # 2. Los pesos de esas conexiones
                suma_delta = 0.0
                for k in range(len(deltas[-1])):
                    # delta de neurona k en capa siguiente * peso que conecta j con k
                    # Si el peso es grande, esta neurona contribuye más al error
                    suma_delta += self.pesos[capa][k][j] * deltas[-1][k]
                
                # Multiplicar por derivada de activación de esta neurona
                # Esto implementa ⊙ σ'(z^l)
                # Si la derivada es pequeña (gradiente saturado), el error se "desvanece"
                grad_act = self.derivada_activacion(self.activaciones[capa][j])
                delta = suma_delta * grad_act
                
                delta_capa.append(delta)
            
            # Guardar deltas de esta capa
            deltas.append(delta_capa)
        
        # Invertir deltas para tener orden correcto (de entrada a salida)
        # Ahora deltas[i] corresponde a la capa i
        deltas.reverse()
        
        # PASO 3: Calcular gradientes de pesos y sesgos
        # ∂C/∂w^l_jk = δ^l_j * a^(l-1)_k
        # ∂C/∂b^l_j = δ^l_j
        self.gradientes_pesos = []
        self.gradientes_sesgos = []
        
        for capa in range(self.num_capas - 1):
            grad_pesos_capa = []
            grad_sesgos_capa = []
            
            for j in range(len(self.pesos[capa])):
                # Gradiente del sesgo = delta
                grad_sesgos_capa.append(deltas[capa][j])
                
                # Gradiente de pesos = delta * activación_anterior
                grad_pesos_neurona = []
                for k in range(len(self.pesos[capa][j])):
                    grad = deltas[capa][j] * self.activaciones[capa][k]
                    grad_pesos_neurona.append(grad)
                
                grad_pesos_capa.append(grad_pesos_neurona)
            
            self.gradientes_pesos.append(grad_pesos_capa)
            self.gradientes_sesgos.append(grad_sesgos_capa)
        
        # Calcular MSE para monitoreo
        mse = sum((salida_deseada[i] - salida_red[i]) ** 2 
                  for i in range(len(salida_red))) / len(salida_red)
        
        return mse
    
    def actualizar_pesos(self):
        """
        Actualiza pesos usando el optimizador seleccionado.
        
        OPTIMIZADORES:
        - SGD: w = w - η * ∇w
        - Momentum: v = β*v + η*∇w; w = w - v
        - Adam: Combina momento y RMSprop con corrección de sesgo
        """
        if self.optimizador == 'momentum':
            self._actualizar_momentum()
        elif self.optimizador == 'adam':
            self._actualizar_adam()
        else:
            self._actualizar_sgd()
    
    def _actualizar_sgd(self):
        """
        Descenso de Gradiente Estocástico (SGD).
        Actualización simple: w = w - η * ∇w
        
        Es el método más básico pero puede ser lento y oscilar.
        """
        for capa in range(self.num_capas - 1):
            for j in range(len(self.pesos[capa])):
                # Actualizar sesgos
                # Movemos el sesgo en dirección opuesta al gradiente
                # Si el gradiente es positivo, disminuimos el sesgo
                self.sesgos[capa][j] -= self.tasa_aprendizaje * self.gradientes_sesgos[capa][j]
                
                # Actualizar pesos
                for k in range(len(self.pesos[capa][j])):
                    # Descenso de gradiente: paso en dirección opuesta al gradiente
                    # η controla qué tan grande es el paso
                    self.pesos[capa][j][k] -= self.tasa_aprendizaje * self.gradientes_pesos[capa][j][k]
    
    def _actualizar_momentum(self):
        """
        Descenso de Gradiente con Momento.
        v = β*v + η*∇w
        w = w - v
        
        Ventaja: Acelera convergencia y reduce oscilaciones.
        El momento "recuerda" la dirección previa y suaviza las actualizaciones.
        """
        for capa in range(self.num_capas - 1):
            for j in range(len(self.pesos[capa])):
                # Actualizar velocidad y sesgo
                # La velocidad combina el momento anterior (β*v) con el gradiente actual (η*∇w)
                # β típicamente = 0.9, significa que el 90% de la velocidad anterior se mantiene
                self.momento_sesgos[capa][j] = (
                    self.momento_factor * self.momento_sesgos[capa][j] +
                    self.tasa_aprendizaje * self.gradientes_sesgos[capa][j]
                )
                # Actualizar sesgo usando la velocidad acumulada
                self.sesgos[capa][j] -= self.momento_sesgos[capa][j]
                
                # Actualizar velocidad y pesos
                for k in range(len(self.pesos[capa][j])):
                    # Calcular nueva velocidad: mezcla de momento anterior y gradiente actual
                    # Esto crea una "inercia" que ayuda a:
                    # 1. Pasar por mínimos locales planos
                    # 2. Reducir oscilaciones en valles estrechos
                    self.momento_pesos[capa][j][k] = (
                        self.momento_factor * self.momento_pesos[capa][j][k] +
                        self.tasa_aprendizaje * self.gradientes_pesos[capa][j][k]
                    )
                    # Aplicar actualización usando la velocidad
                    self.pesos[capa][j][k] -= self.momento_pesos[capa][j][k]
    
    def _actualizar_adam(self):
        """
        Algoritmo Adam (Adaptive Moment Estimation).
        
        Combina:
        - Momento (primer momento): media de gradientes
        - RMSprop (segundo momento): media de gradientes al cuadrado
        - Corrección de sesgo para primeras iteraciones
        
        m = β₁*m + (1-β₁)*∇w      (primer momento - dirección)
        v = β₂*v + (1-β₂)*∇w²     (segundo momento - magnitud)
        m̂ = m/(1-β₁^t)            (corrección de sesgo)
        v̂ = v/(1-β₂^t)            (corrección de sesgo)
        w = w - α * m̂/(√v̂ + ε)   (actualización adaptativa)
        """
        # Incrementar contador de iteraciones (necesario para corrección de sesgo)
        self.t += 1
        
        for capa in range(self.num_capas - 1):
            for j in range(len(self.pesos[capa])):
                # Actualizar sesgos con Adam
                g_b = self.gradientes_sesgos[capa][j]  # Gradiente actual
                
                # Primer momento (media móvil exponencial del gradiente)
                # Captura la "dirección" general del gradiente
                # β₁ ≈ 0.9 significa que se mantiene 90% del momento anterior
                self.m_sesgos[capa][j] = (
                    self.beta1 * self.m_sesgos[capa][j] +
                    (1 - self.beta1) * g_b
                )
                
                # Segundo momento (media móvil exponencial del gradiente al cuadrado)
                # Captura la "magnitud" o "varianza" del gradiente
                # β₂ ≈ 0.999 significa que se mantiene 99.9% del momento anterior
                self.v_sesgos[capa][j] = (
                    self.beta2 * self.v_sesgos[capa][j] +
                    (1 - self.beta2) * g_b * g_b
                )
                
                # Corrección de sesgo
                # Al inicio, m y v están sesgados hacia 0
                # Esta corrección compensa ese sesgo dividing por (1 - β^t)
                m_hat = self.m_sesgos[capa][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v_sesgos[capa][j] / (1 - self.beta2 ** self.t)
                
                # Actualizar sesgo
                # Dividimos por √v̂ para tener tasas de aprendizaje adaptativas
                # ε evita división por cero
                self.sesgos[capa][j] -= self.tasa_aprendizaje * m_hat / (math.sqrt(v_hat) + self.epsilon)
                
                # Actualizar pesos con Adam (mismo proceso que sesgos)
                for k in range(len(self.pesos[capa][j])):
                    g_w = self.gradientes_pesos[capa][j][k]
                    
                    # Primer momento: dirección promedio del gradiente
                    self.m_pesos[capa][j][k] = (
                        self.beta1 * self.m_pesos[capa][j][k] +
                        (1 - self.beta1) * g_w
                    )
                    
                    # Segundo momento: magnitud promedio del gradiente
                    self.v_pesos[capa][j][k] = (
                        self.beta2 * self.v_pesos[capa][j][k] +
                        (1 - self.beta2) * g_w * g_w
                    )
                    
                    # Corrección de sesgo para obtener estimaciones no sesgadas
                    m_hat = self.m_pesos[capa][j][k] / (1 - self.beta1 ** self.t)
                    v_hat = self.v_pesos[capa][j][k] / (1 - self.beta2 ** self.t)
                    
                    # Actualizar peso con tasa de aprendizaje adaptativa
                    # Cada peso tiene su propia tasa efectiva basada en su historial de gradientes
                    self.pesos[capa][j][k] -= self.tasa_aprendizaje * m_hat / (math.sqrt(v_hat) + self.epsilon)
    
    def entrenar(self, 
                 datos: List[Tuple[List[float], List[float]]],
                 epocas: int = 1000,
                 tam_lote: int = 1,
                 verbose: bool = False) -> List[float]:
        """
        Entrena la red neuronal.
        
        Args:
            datos: Lista de (entrada, salida_deseada)
            epocas: Número de épocas de entrenamiento
            tam_lote: Tamaño del lote (batch size)
            verbose: Mostrar progreso
            
        Returns:
            Historial de errores
        """
        historial_error = []
        
        # Iterar por número de épocas
        # Una época = un pase completo por todos los datos
        for epoca in range(epocas):
            error_total = 0.0
            
            # Mini-batch gradient descent
            # Dividir datos en lotes y acumular gradientes antes de actualizar
            # Ventaja: más estable que online (tam=1), más rápido que batch completo
            for i in range(0, len(datos), tam_lote):
                # Obtener lote actual (puede ser más pequeño al final)
                lote = datos[i:i + tam_lote]
                
                # Inicializar acumuladores de gradientes
                # Vamos a sumar los gradientes de todas las muestras del lote
                grad_pesos_acum = [[[ 0.0 for _ in range(len(self.pesos[c][j]))]
                                    for j in range(len(self.pesos[c]))]
                                   for c in range(self.num_capas - 1)]
                
                grad_sesgos_acum = [[0.0 for _ in range(len(self.sesgos[c]))]
                                    for c in range(self.num_capas - 1)]
                
                # Procesar cada muestra del lote
                for entrada, salida_deseada in lote:
                    # Forward pass: calcular predicción
                    self.forward(entrada)
                    
                    # Backward pass: calcular gradientes para esta muestra
                    error = self.backward(salida_deseada)
                    error_total += error
                    
                    # Acumular gradientes de esta muestra
                    # No actualizamos pesos todavía, solo sumamos gradientes
                    for c in range(self.num_capas - 1):
                        for j in range(len(self.pesos[c])):
                            grad_sesgos_acum[c][j] += self.gradientes_sesgos[c][j]
                            for k in range(len(self.pesos[c][j])):
                                grad_pesos_acum[c][j][k] += self.gradientes_pesos[c][j][k]
                
                # Promediar gradientes del lote
                # Esto hace el aprendizaje más estable que usar gradientes individuales
                for c in range(self.num_capas - 1):
                    for j in range(len(self.pesos[c])):
                        self.gradientes_sesgos[c][j] = grad_sesgos_acum[c][j] / len(lote)
                        for k in range(len(self.pesos[c][j])):
                            self.gradientes_pesos[c][j][k] = grad_pesos_acum[c][j][k] / len(lote)
                
                # Ahora sí actualizar pesos con gradientes promediados
                # Esto se hace una vez por lote, no por cada muestra
                self.actualizar_pesos()
            
            # Calcular error promedio de la época
            error_promedio = error_total / len(datos)
            historial_error.append(error_promedio)
            
            # Mostrar progreso cada 100 épocas si verbose=True
            if verbose and (epoca + 1) % 100 == 0:
                print(f"Época {epoca + 1}/{epocas} - Error: {error_promedio:.6f}")
        
        return historial_error
    
    def predecir(self, entrada: List[float]) -> List[float]:
        """Realiza predicción para una entrada."""
        return self.forward(entrada)
    
    def obtener_magnitud_gradientes(self) -> Dict[str, float]:
        """
        Calcula la magnitud promedio de los gradientes por capa.
        Útil para detectar vanishing/exploding gradients.
        
        Vanishing gradient: magnitudes muy pequeñas (< 0.001)
        Exploding gradient: magnitudes muy grandes (> 10)
        
        Returns:
            Diccionario con magnitudes por capa
        """
        magnitudes = {}
        
        # Calcular magnitud promedio para cada capa
        for capa in range(self.num_capas - 1):
            suma_grad = 0.0
            contador = 0
            
            # Sumar valor absoluto de todos los gradientes de pesos
            # Usamos valor absoluto para medir magnitud sin importar dirección
            for j in range(len(self.gradientes_pesos[capa])):
                for k in range(len(self.gradientes_pesos[capa][j])):
                    suma_grad += abs(self.gradientes_pesos[capa][j][k])
                    contador += 1
            
            # Calcular promedio de magnitud
            # Si es muy pequeño → vanishing gradient problem
            # Si es muy grande → exploding gradient problem
            magnitud_promedio = suma_grad / contador if contador > 0 else 0.0
            magnitudes[f"Capa_{capa}"] = magnitud_promedio
        
        return magnitudes

# ============================================================================
# Funciones de Utilidad
# ============================================================================

def calcular_precision(red: RedNeuronalBackprop,
                       datos: List[Tuple[List[float], List[float]]],
                       umbral: float = 0.5) -> float:
    """Calcula precisión de clasificación."""
    correctos = 0
    
    for entrada, salida_esperada in datos:
        salida = red.predecir(entrada)
        prediccion = [1.0 if s >= umbral else 0.0 for s in salida]
        
        if prediccion == salida_esperada:
            correctos += 1
    
    return correctos / len(datos) if datos else 0.0

# ============================================================================
# Modo DEMO
# ============================================================================

def modo_demo():
    """
    Demuestra retropropagación con diferentes optimizadores y
    análisis de gradientes.
    """
    print("\n" + "="*70)
    print("MODO DEMO: Retropropagación del Error")
    print("="*70)
    
    # ========================================
    # Parte 1: Comparación de Optimizadores en XOR
    # ========================================
    print("\n--- Parte 1: Comparación de Optimizadores ---")
    print("\nProblema XOR con diferentes optimizadores:\n")
    
    # Datos XOR
    datos_xor = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]
    
    # Optimizadores a comparar
    optimizadores = ['sgd', 'momentum', 'adam']
    arquitectura = [2, 6, 1]  # 2 entradas, 6 ocultas, 1 salida
    epocas = 2000
    
    resultados = {}
    
    for opt in optimizadores:
        print(f"\nOptimizador: {opt.upper()}")
        print(f"Arquitectura: {arquitectura}")
        
        # Crear y entrenar red
        red = RedNeuronalBackprop(
            arquitectura=arquitectura,
            funcion_activacion='sigmoide',
            optimizador=opt,
            tasa_aprendizaje=0.1 if opt != 'adam' else 0.01,
            momento=0.9
        )
        
        historial = red.entrenar(datos_xor, epocas=epocas, tam_lote=4, verbose=False)
        
        # Evaluar
        precision = calcular_precision(red, datos_xor)
        
        print(f"  Error inicial: {historial[0]:.6f}")
        print(f"  Error final: {historial[-1]:.6f}")
        print(f"  Precisión: {precision * 100:.2f}%")
        
        # Mostrar predicciones
        print("  Predicciones:")
        for entrada, salida_esp in datos_xor:
            pred = red.predecir(entrada)
            print(f"    {entrada} → {pred[0]:.4f} (esperado: {salida_esp[0]})")
        
        resultados[opt] = {
            'historial': historial,
            'precision': precision,
            'red': red
        }
    
    # ========================================
    # Parte 2: Análisis de Gradientes
    # ========================================
    print("\n" + "="*70)
    print("--- Parte 2: Análisis de Magnitud de Gradientes ---")
    print("="*70)
    
    print("\nMagnitud de gradientes por capa (última época):")
    print("(Útil para detectar vanishing/exploding gradients)\n")
    
    for opt in optimizadores:
        red = resultados[opt]['red']
        
        # Calcular gradientes en la última muestra
        entrada, salida = datos_xor[0]
        red.forward(entrada)
        red.backward(salida)
        
        magnitudes = red.obtener_magnitud_gradientes()
        
        print(f"{opt.upper()}:")
        for capa, mag in magnitudes.items():
            print(f"  {capa}: {mag:.6f}")
        print()
    
    # ========================================
    # Parte 3: Efecto del Tamaño de Lote
    # ========================================
    print("="*70)
    print("--- Parte 3: Efecto del Tamaño de Lote ---")
    print("="*70)
    
    print("\nComparando diferentes tamaños de lote (batch sizes):\n")
    
    tamanos_lote = [1, 2, 4]  # 1=SGD estocástico, 4=batch completo
    
    for tam in tamanos_lote:
        red = RedNeuronalBackprop(
            arquitectura=[2, 6, 1],
            funcion_activacion='sigmoide',
            optimizador='momentum',
            tasa_aprendizaje=0.1
        )
        
        historial = red.entrenar(datos_xor, epocas=1000, tam_lote=tam, verbose=False)
        precision = calcular_precision(red, datos_xor)
        
        nombre_lote = "Estocástico (online)" if tam == 1 else f"Mini-batch ({tam})" if tam < 4 else "Batch completo"
        
        print(f"{nombre_lote}:")
        print(f"  Error final: {historial[-1]:.6f}")
        print(f"  Precisión: {precision * 100:.2f}%\n")
    
    # ========================================
    # Parte 4: Problema de Vanishing Gradient
    # ========================================
    print("="*70)
    print("--- Parte 4: Demostración de Vanishing Gradient ---")
    print("="*70)
    
    print("\nRed profunda con sigmoide (propensa a vanishing gradient):\n")
    
    # Red muy profunda
    arq_profunda = [2, 8, 8, 8, 1]
    
    red_profunda = RedNeuronalBackprop(
        arquitectura=arq_profunda,
        funcion_activacion='sigmoide',
        optimizador='adam',
        tasa_aprendizaje=0.01
    )
    
    print(f"Arquitectura: {arq_profunda}")
    historial = red_profunda.entrenar(datos_xor, epocas=1000, tam_lote=4, verbose=False)
    
    # Analizar gradientes capa por capa
    entrada, salida = datos_xor[0]
    red_profunda.forward(entrada)
    red_profunda.backward(salida)
    
    magnitudes = red_profunda.obtener_magnitud_gradientes()
    
    print("\nMagnitud de gradientes por capa:")
    for capa, mag in magnitudes.items():
        print(f"  {capa}: {mag:.8f}")
    
    print(f"\nError final: {historial[-1]:.6f}")
    print("\nNota: Las capas más cercanas a la entrada tienen gradientes")
    print("más pequeños (vanishing gradient problem).")
    
    # ========================================
    # Parte 5: Comparación con ReLU
    # ========================================
    print("\n" + "="*70)
    print("--- Parte 5: Comparación Sigmoide vs ReLU ---")
    print("="*70)
    
    print("\nMisma red profunda con ReLU:\n")
    
    red_relu = RedNeuronalBackprop(
        arquitectura=[2, 8, 8, 8, 1],
        funcion_activacion='relu',
        optimizador='adam',
        tasa_aprendizaje=0.01
    )
    
    historial_relu = red_relu.entrenar(datos_xor, epocas=1000, tam_lote=4, verbose=False)
    
    # Analizar gradientes
    red_relu.forward(entrada)
    red_relu.backward(salida)
    magnitudes_relu = red_relu.obtener_magnitud_gradientes()
    
    print("Magnitud de gradientes por capa (ReLU):")
    for capa, mag in magnitudes_relu.items():
        print(f"  {capa}: {mag:.8f}")
    
    print(f"\nError final: {historial_relu[-1]:.6f}")
    precision_relu = calcular_precision(red_relu, datos_xor)
    print(f"Precisión: {precision_relu * 100:.2f}%")
    
    print("\nNota: ReLU mitiga el vanishing gradient problem.")

# ============================================================================
# Modo INTERACTIVO
# ============================================================================

def modo_interactivo():
    """Permite configuración personalizada de retropropagación."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Configuración de Retropropagación")
    print("="*70)
    
    # Datos XOR por defecto
    datos_xor = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]
    
    print("\nProblema: XOR")
    print("Datos de entrenamiento: 4 patrones\n")
    
    # Configurar arquitectura
    print("--- Configuración de Arquitectura ---")
    n_ocultas = int(input("Neuronas en capa oculta (default=6): ").strip() or "6")
    
    # Seleccionar función de activación
    print("\nFunción de activación:")
    print("1) Sigmoide")
    print("2) Tanh")
    print("3) ReLU")
    act_op = input("Ingrese opción (1-3, default=1): ").strip()
    
    activacion = 'sigmoide'
    if act_op == '2':
        activacion = 'tanh'
    elif act_op == '3':
        activacion = 'relu'
    
    # Seleccionar optimizador
    print("\nOptimizador:")
    print("1) SGD (Descenso de Gradiente Estocástico)")
    print("2) Momentum (SGD con momento)")
    print("3) Adam (Adaptive Moment Estimation)")
    opt_op = input("Ingrese opción (1-3, default=3): ").strip()
    
    optimizador = 'adam'
    if opt_op == '1':
        optimizador = 'sgd'
    elif opt_op == '2':
        optimizador = 'momentum'
    
    # Configurar hiperparámetros
    print("\n--- Hiperparámetros ---")
    
    tasa_default = "0.01" if optimizador == 'adam' else "0.1"
    tasa = float(input(f"Tasa de aprendizaje (default={tasa_default}): ").strip() or tasa_default)
    
    epocas = int(input("Épocas de entrenamiento (default=2000): ").strip() or "2000")
    
    tam_lote = int(input("Tamaño de lote (1=estocástico, 4=batch, default=4): ").strip() or "4")
    
    # Crear y entrenar red
    print(f"\n--- Entrenando Red ---")
    print(f"Arquitectura: [2, {n_ocultas}, 1]")
    print(f"Activación: {activacion}")
    print(f"Optimizador: {optimizador}")
    print(f"Tasa de aprendizaje: {tasa}")
    print(f"Tamaño de lote: {tam_lote}\n")
    
    red = RedNeuronalBackprop(
        arquitectura=[2, n_ocultas, 1],
        funcion_activacion=activacion,
        optimizador=optimizador,
        tasa_aprendizaje=tasa
    )
    
    historial = red.entrenar(datos_xor, epocas=epocas, tam_lote=tam_lote, verbose=True)
    
    # Resultados
    print("\n--- Resultados ---")
    print(f"Error final: {historial[-1]:.6f}")
    
    precision = calcular_precision(red, datos_xor)
    print(f"Precisión: {precision * 100:.2f}%")
    
    print("\nPredicciones:")
    for entrada, salida_esp in datos_xor:
        pred = red.predecir(entrada)
        print(f"  {entrada} → {pred[0]:.4f} (esperado: {salida_esp[0]})")
    
    # Análisis de gradientes
    print("\n--- Análisis de Gradientes ---")
    entrada, salida = datos_xor[0]
    red.forward(entrada)
    red.backward(salida)
    
    magnitudes = red.obtener_magnitud_gradientes()
    print("\nMagnitud promedio de gradientes por capa:")
    for capa, mag in magnitudes.items():
        print(f"  {capa}: {mag:.6f}")

# ============================================================================
# Main
# ============================================================================

def main():
    """Punto de entrada del programa."""
    print("\n" + "="*70)
    print("043-E2: Retropropagación del Error (Backpropagation)")
    print("="*70)
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (comparación de optimizadores y análisis)")
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

