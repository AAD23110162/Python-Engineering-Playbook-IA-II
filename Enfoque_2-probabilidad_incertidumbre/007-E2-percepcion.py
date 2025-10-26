"""
007-E2-percepcion.py
--------------------------------
Este script implementa sistemas de Percepción bajo Incertidumbre:
- Modela sensores con ruido y errores de medición probabilísticos
- Implementa filtrado bayesiano para fusionar múltiples lecturas de sensores
- Aplica el Filtro de Kalman para estimar estados a partir de observaciones ruidosas
- Integra información de múltiples sensores mediante fusión probabilística
- Maneja incertidumbre en localización y mapeo (conceptos básicos de SLAM)
- Visualiza la evolución de las creencias sobre el estado del sistema
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente fusión de sensores en escenarios predefinidos
2. INTERACTIVO: permite configurar modelos de sensores y observar el proceso de filtrado

Autor: Alejandro Aguirre Díaz
"""

import numpy as np
import random

# ========== CLASE SENSOR ==========

class Sensor:
    """
    Representa un sensor con ruido gaussiano.
    """
    
    def __init__(self, nombre, media_ruido=0.0, desviacion_ruido=1.0):
        """
        Inicializa un sensor con parámetros de ruido.
        
        :param nombre: identificador del sensor
        :param media_ruido: media del ruido gaussiano
        :param desviacion_ruido: desviación estándar del ruido
        """
        self.nombre = nombre
        self.media_ruido = media_ruido
        self.desviacion_ruido = desviacion_ruido
    
    def medir(self, valor_real):
        """
        Simula una medición con ruido gaussiano.
        
        :param valor_real: valor verdadero a medir
        :return: medición ruidosa
        """
        # Añadir ruido gaussiano: medición = valor_real + N(media_ruido, desviacion²)
        ruido = np.random.normal(self.media_ruido, self.desviacion_ruido)
        medicion = valor_real + ruido
        return medicion
    
    def probabilidad_observacion(self, medicion, valor_hipotetico):
        """
        Calcula P(medición | valor_hipotetico) usando distribución gaussiana.
        
        :param medicion: valor observado
        :param valor_hipotetico: hipótesis sobre el valor real
        :return: probabilidad (verosimilitud)
        """
        # P(z | x) ~ N(x, σ²)
        # Función de densidad gaussiana
        diferencia = medicion - valor_hipotetico
        exponente = -0.5 * (diferencia / self.desviacion_ruido) ** 2
        coef = 1.0 / (self.desviacion_ruido * np.sqrt(2 * np.pi))
        return coef * np.exp(exponente)

# ========== FUSIÓN DE SENSORES ==========

class FusionSensores:
    """
    Implementa fusión de múltiples sensores usando inferencia bayesiana.
    """
    
    def __init__(self, sensores):
        """
        Inicializa el sistema de fusión con una lista de sensores.
        
        :param sensores: lista de objetos Sensor
        """
        self.sensores = sensores
    
    def estimar_bayes(self, mediciones, prior_media=0.0, prior_varianza=100.0, rango=(-50, 50), num_hipotesis=1000):
        """
        Estima el valor real usando actualización bayesiana.
        
        :param mediciones: dict {nombre_sensor: medición}
        :param prior_media: media de la distribución a priori
        :param prior_varianza: varianza de la distribución a priori
        :param rango: rango de valores posibles (min, max)
        :param num_hipotesis: número de hipótesis a evaluar
        :return: valor estimado (máximo a posteriori)
        """
        # Generar hipótesis sobre el valor real
        hipotesis = np.linspace(rango[0], rango[1], num_hipotesis)
        
        # Inicializar con probabilidad a priori (distribución gaussiana)
        # P(x) ~ N(prior_media, prior_varianza)
        prob_posterior = np.exp(-0.5 * ((hipotesis - prior_media) / np.sqrt(prior_varianza)) ** 2)
        prob_posterior /= np.sum(prob_posterior)  # Normalizar
        
        # Actualizar con cada observación usando regla de Bayes
        # P(x | z1, z2, ..., zn) ∝ P(z1 | x) × P(z2 | x) × ... × P(zn | x) × P(x)
        for nombre_sensor, medicion in mediciones.items():
            # Encontrar el sensor correspondiente
            sensor = next((s for s in self.sensores if s.nombre == nombre_sensor), None)
            
            if sensor is None:
                continue
            
            # Calcular verosimilitud P(medición | hipótesis) para cada hipótesis
            verosimilitud = np.array([sensor.probabilidad_observacion(medicion, h) for h in hipotesis])
            
            # Actualizar posterior: P(x | z) ∝ P(z | x) × P(x)
            prob_posterior *= verosimilitud
            
            # Normalizar para mantenerlo como distribución de probabilidad
            prob_posterior /= np.sum(prob_posterior)
        
        # Encontrar máximo a posteriori (MAP)
        idx_max = np.argmax(prob_posterior)
        estimacion_map = hipotesis[idx_max]
        
        # Calcular media (estimación de mínimos cuadrados)
        estimacion_media = np.sum(hipotesis * prob_posterior)
        
        # Calcular varianza posterior
        varianza_posterior = np.sum((hipotesis - estimacion_media) ** 2 * prob_posterior)
        
        return {
            'map': estimacion_map,
            'media': estimacion_media,
            'varianza': varianza_posterior,
            'distribucion': (hipotesis, prob_posterior)
        }
    
    def fusion_simple(self, mediciones):
        """
        Fusión simple: promedio ponderado por confianza de sensores.
        
        :param mediciones: dict {nombre_sensor: medición}
        :return: estimación fusionada
        """
        # Ponderación inversa por varianza (sensores más precisos tienen más peso)
        suma_ponderada = 0.0
        suma_pesos = 0.0
        
        for nombre_sensor, medicion in mediciones.items():
            sensor = next((s for s in self.sensores if s.nombre == nombre_sensor), None)
            if sensor:
                # Peso = 1 / varianza (mayor precisión → mayor peso)
                peso = 1.0 / (sensor.desviacion_ruido ** 2)
                suma_ponderada += peso * medicion
                suma_pesos += peso
        
        return suma_ponderada / suma_pesos if suma_pesos > 0 else 0.0

# ========== FILTRO DE KALMAN 1D ==========

class FiltroKalman1D:
    """
    Implementación simple del filtro de Kalman en 1D.
    Útil para seguimiento y estimación de estado con sensores ruidosos.
    """
    
    def __init__(self, media_inicial=0.0, varianza_inicial=1000.0):
        """
        Inicializa el filtro de Kalman.
        
        :param media_inicial: estimación inicial del estado
        :param varianza_inicial: incertidumbre inicial
        """
        # Creencia actual (distribución gaussiana)
        self.media = media_inicial
        self.varianza = varianza_inicial
        
        # Historial de estimaciones
        self.historial_media = [media_inicial]
        self.historial_varianza = [varianza_inicial]
    
    def predecir(self, movimiento, varianza_movimiento):
        """
        Paso de predicción: actualizar creencia según modelo de movimiento.
        
        :param movimiento: cambio esperado en el estado
        :param varianza_movimiento: incertidumbre del movimiento
        """
        # Modelo: x_nuevo = x_anterior + movimiento
        # La varianza aumenta (se vuelve más incierto)
        self.media += movimiento
        self.varianza += varianza_movimiento
    
    def actualizar(self, medicion, varianza_sensor):
        """
        Paso de actualización: incorporar nueva medición del sensor.
        
        :param medicion: valor medido
        :param varianza_sensor: varianza del sensor
        """
        # Ganancia de Kalman: K = σ²_predicción / (σ²_predicción + σ²_sensor)
        # Determina cuánto confiamos en la medición vs. la predicción
        ganancia_kalman = self.varianza / (self.varianza + varianza_sensor)
        
        # Actualizar media: μ_nuevo = μ_pred + K × (medición - μ_pred)
        self.media = self.media + ganancia_kalman * (medicion - self.media)
        
        # Actualizar varianza: σ²_nuevo = (1 - K) × σ²_pred
        # La varianza siempre disminuye (se vuelve más cierto)
        self.varianza = (1 - ganancia_kalman) * self.varianza
        
        # Guardar en historial
        self.historial_media.append(self.media)
        self.historial_varianza.append(self.varianza)

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con fusión de sensores."""
    print("\n" + "="*70)
    print("MODO DEMO: Percepción bajo Incertidumbre")
    print("="*70)
    
    # ========== EJEMPLO 1: Fusión de Sensores ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Fusión de Múltiples Sensores Ruidosos")
    print("="*70)
    
    # Valor real que queremos medir
    temperatura_real = 25.0
    print(f"\n--- Escenario: Medir temperatura real = {temperatura_real}°C ---")
    
    # Crear sensores con diferentes características de ruido
    sensor_a = Sensor("Termómetro A", media_ruido=0.0, desviacion_ruido=0.5)
    sensor_b = Sensor("Termómetro B", media_ruido=0.0, desviacion_ruido=1.5)
    sensor_c = Sensor("Termómetro C", media_ruido=0.5, desviacion_ruido=1.0)
    
    sensores = [sensor_a, sensor_b, sensor_c]
    
    print("\nSensores disponibles:")
    for s in sensores:
        print(f"  - {s.nombre}: Ruido ~ N({s.media_ruido}, {s.desviacion_ruido}²)")
    
    # Tomar mediciones
    print("\n--- Mediciones individuales ---")
    mediciones = {}
    for sensor in sensores:
        medicion = sensor.medir(temperatura_real)
        mediciones[sensor.nombre] = medicion
        error = abs(medicion - temperatura_real)
        print(f"{sensor.nombre}: {medicion:.2f}°C (error = {error:.2f}°C)")
    
    # Fusionar sensores
    fusion = FusionSensores(sensores)
    
    print("\n--- Fusión Bayesiana ---")
    resultado_bayes = fusion.estimar_bayes(mediciones, prior_media=20.0, prior_varianza=100.0)
    print(f"Estimación MAP: {resultado_bayes['map']:.2f}°C")
    print(f"Estimación Media: {resultado_bayes['media']:.2f}°C")
    print(f"Varianza Posterior: {resultado_bayes['varianza']:.2f}")
    print(f"Error MAP: {abs(resultado_bayes['map'] - temperatura_real):.2f}°C")
    
    print("\n--- Fusión Simple (Promedio Ponderado) ---")
    estimacion_simple = fusion.fusion_simple(mediciones)
    print(f"Estimación: {estimacion_simple:.2f}°C")
    print(f"Error: {abs(estimacion_simple - temperatura_real):.2f}°C")
    
    print("\n>>> OBSERVACIÓN:")
    print("    La fusión bayesiana combina información de todos los sensores,")
    print("    ponderando automáticamente según su precisión.")
    
    # ========== EJEMPLO 2: Filtro de Kalman ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Filtro de Kalman para Seguimiento")
    print("="*70)
    
    print("\n--- Escenario: Robot moviéndose en línea recta ---")
    print("Posición inicial real: 0.0 m")
    print("Movimiento por paso: 1.0 m")
    print("Sensor: desviación estándar = 0.5 m")
    
    # Inicializar filtro de Kalman
    filtro = FiltroKalman1D(media_inicial=0.0, varianza_inicial=1000.0)
    
    # Parámetros del sensor
    varianza_sensor = 0.5 ** 2
    varianza_movimiento = 0.1 ** 2
    
    # Simular 10 pasos
    posicion_real = 0.0
    movimiento_por_paso = 1.0
    
    print("\n--- Simulación de seguimiento ---")
    print("Paso | Pos. Real | Medición | Predicción | Actualización | Error")
    print("-" * 70)
    
    for paso in range(10):
        # Mover el robot
        posicion_real += movimiento_por_paso
        
        # Predicción (antes de medir)
        filtro.predecir(movimiento_por_paso, varianza_movimiento)
        prediccion = filtro.media
        
        # Tomar medición ruidosa
        medicion = posicion_real + np.random.normal(0, 0.5)
        
        # Actualizar con medición
        filtro.actualizar(medicion, varianza_sensor)
        estimacion = filtro.media
        
        # Calcular error
        error = abs(estimacion - posicion_real)
        
        print(f"{paso+1:4d} |  {posicion_real:7.2f} | {medicion:8.2f} | "
              f"{prediccion:10.2f} | {estimacion:13.2f} |  {error:.2f}")
    
    print("\n>>> OBSERVACIÓN:")
    print(f"    Varianza inicial: {filtro.historial_varianza[0]:.2f}")
    print(f"    Varianza final: {filtro.historial_varianza[-1]:.4f}")
    print("    El filtro de Kalman reduce la incertidumbre a medida que")
    print("    integra más mediciones, mejorando la estimación.")
    
    # ========== EJEMPLO 3: Comparación con y sin fusión ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Beneficios de la Fusión de Sensores")
    print("="*70)
    
    temperatura_prueba = 30.0
    num_experimentos = 50
    
    print(f"\nRealizando {num_experimentos} experimentos...")
    
    errores_sensor_a = []
    errores_sensor_b = []
    errores_fusion = []
    
    for _ in range(num_experimentos):
        med_a = sensor_a.medir(temperatura_prueba)
        med_b = sensor_b.medir(temperatura_prueba)
        
        mediciones_exp = {"Termómetro A": med_a, "Termómetro B": med_b}
        est_fusion = fusion.fusion_simple(mediciones_exp)
        
        errores_sensor_a.append(abs(med_a - temperatura_prueba))
        errores_sensor_b.append(abs(med_b - temperatura_prueba))
        errores_fusion.append(abs(est_fusion - temperatura_prueba))
    
    print(f"\n--- Error Promedio ---")
    print(f"Termómetro A solo: {np.mean(errores_sensor_a):.3f}°C")
    print(f"Termómetro B solo: {np.mean(errores_sensor_b):.3f}°C")
    print(f"Fusión A + B:       {np.mean(errores_fusion):.3f}°C")
    
    print("\n>>> CONCLUSIÓN:")
    print("    La fusión de sensores reduce el error promedio")
    print("    aprovechando las fortalezas de múltiples mediciones.")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo con configuración personalizada."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Diseña tu Sistema de Percepción")
    print("="*70)
    
    # ========== PASO 1: Configurar sensores ==========
    print("\n--- Configurar sensores ---")
    print("¿Cuántos sensores deseas usar? (recomendado: 2-4)")
    
    try:
        num_sensores = int(input("Número de sensores: ").strip() or "2")
    except:
        num_sensores = 2
    
    sensores = []
    for i in range(num_sensores):
        print(f"\nSensor {i+1}:")
        try:
            desv = float(input(f"  Desviación estándar del ruido (ej: 0.5-2.0): ").strip() or "1.0")
        except:
            desv = 1.0
        
        sensor = Sensor(f"Sensor_{i+1}", media_ruido=0.0, desviacion_ruido=desv)
        sensores.append(sensor)
    
    print(f"\n✓ {len(sensores)} sensores configurados")
    
    # ========== PASO 2: Definir valor real ==========
    print("\n--- Definir valor real a medir ---")
    try:
        valor_real = float(input("Valor real (ej: 25.0): ").strip() or "25.0")
    except:
        valor_real = 25.0
    
    print(f"Valor real establecido: {valor_real}")
    
    # ========== PASO 3: Tomar mediciones ==========
    print("\n--- Tomando mediciones ---")
    mediciones = {}
    
    for sensor in sensores:
        medicion = sensor.medir(valor_real)
        mediciones[sensor.nombre] = medicion
        error = abs(medicion - valor_real)
        print(f"{sensor.nombre}: {medicion:.2f} (error = {error:.2f})")
    
    # ========== PASO 4: Fusionar ==========
    print("\n--- Fusionando información de sensores ---")
    
    fusion = FusionSensores(sensores)
    estimacion = fusion.fusion_simple(mediciones)
    
    print(f"\nEstimación fusionada: {estimacion:.2f}")
    print(f"Valor real: {valor_real:.2f}")
    print(f"Error: {abs(estimacion - valor_real):.2f}")
    
    # Comparar con sensores individuales
    print("\n--- Comparación ---")
    for nombre, medicion in mediciones.items():
        print(f"{nombre}: error = {abs(medicion - valor_real):.2f}")
    print(f"Fusión:  error = {abs(estimacion - valor_real):.2f}")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("PERCEPCIÓN BAJO INCERTIDUMBRE")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (fusión de sensores y filtro de Kalman)")
    print("2. INTERACTIVO (configura tu propio sistema de sensores)")
    
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
