"""
002-E2-RP_red-bayesiana.py
--------------------------------
Este script implementa un sistema de Razonamiento Probabilístico usando Redes Bayesianas:
- Construye redes bayesianas con nodos y dependencias probabilísticas
- Permite definir tablas de probabilidad condicional (CPT) para cada nodo
- Realiza inferencia probabilística mediante eliminación de variables
- Calcula probabilidades posteriores dado un conjunto de evidencias
- Visualiza la estructura de la red y las dependencias entre variables
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente una red bayesiana predefinida (ej: diagnóstico médico)
2. INTERACTIVO: permite construir redes personalizadas y realizar consultas de inferencia

Autor: Alejandro Aguirre Díaz
"""

from itertools import product

class NodoRedBayesiana:
    """
    Representa un nodo en una Red Bayesiana con su tabla de probabilidad condicional (CPT).
    """
    def __init__(self, nombre, padres, cpt):
        """
        :parametro nombre: nombre de la variable (str)
        :parametro padres: lista de nombres de nodos padre
        :parametro cpt: diccionario de probabilidades condicionales
                       clave: tupla de valores de padres (en orden), valor: P(nodo=True | padres)
                       Si no tiene padres, clave es () y valor es P(nodo=True)
        """
        self.nombre = nombre
        self.padres = padres
        self.cpt = cpt
    
    def probabilidad(self, valor, valores_padres):
        """
        Obtiene P(nodo=valor | valores_padres).
        :parametro valor: True o False
        :parametro valores_padres: diccionario {nombre_padre: valor_booleano}
        :return: probabilidad (float)
        """
        # Construir clave de consulta en el orden de los padres
        # Si este nodo tiene padres, extraemos sus valores en el orden correcto
        if self.padres:
            # Ejemplo: si padres=['Robo', 'Terremoto'] y valores_padres={'Robo':True, 'Terremoto':False}
            # entonces clave = (True, False)
            clave = tuple(valores_padres[p] for p in self.padres)
        else:
            # Si no tiene padres, la clave es una tupla vacía
            clave = ()
        
        # Buscar en la tabla CPT la probabilidad de que este nodo sea True
        prob_true = self.cpt.get(clave, 0.5)  # Por defecto 0.5 si no está definido
        
        # Si consultamos P(nodo=True), devolvemos prob_true
        # Si consultamos P(nodo=False), devolvemos 1 - prob_true
        return prob_true if valor else (1.0 - prob_true)

class RedBayesiana:
    """
    Representa una Red Bayesiana completa.
    """
    def __init__(self):
        self.nodos = {}  # {nombre: NodoRedBayesiana}
        self.orden_topologico = []
    
    def agregar_nodo(self, nodo):
        """Agrega un nodo a la red."""
        self.nodos[nodo.nombre] = nodo
    
    def establecer_orden_topologico(self, orden):
        """Define el orden topológico de los nodos (padres antes que hijos)."""
        self.orden_topologico = orden
    
    def inferencia_por_enumeracion(self, consulta, evidencia, verbose=False):
        """
        Calcula P(consulta | evidencia) mediante enumeración completa.
        :parametro consulta: diccionario {variable: valor} de la variable de consulta
        :parametro evidencia: diccionario {variable: valor} de variables observadas
        :parametro verbose: si True, muestra el proceso paso a paso
        :return: probabilidad normalizada de la consulta
        """
        # Extraer la variable de consulta y su valor deseado
        var_consulta = list(consulta.keys())[0]
        valor_consulta = consulta[var_consulta]
        
        # Identificar variables ocultas: aquellas que no están en consulta ni en evidencia
        # Ejemplo: si consultamos P(Robo | Llamada=True), las ocultas son {Alarma, Terremoto}
        todas = set(self.nodos.keys())
        ocultas = todas - {var_consulta} - set(evidencia.keys())
        
        if verbose:
            print(f"\n--- Inferencia por Enumeración ---")
            print(f"Consulta: P({var_consulta}={valor_consulta} | {evidencia})")
            print(f"Variables ocultas: {sorted(ocultas)}")
        
        # Acumuladores para las probabilidades conjuntas
        # Calcularemos P(consulta=True, evidencia) y P(consulta=False, evidencia)
        prob_conjunta_true = 0.0
        prob_conjunta_false = 0.0
        
        # Generar todas las asignaciones posibles de variables ocultas
        # Si hay 2 variables ocultas, habrá 2^2 = 4 combinaciones: (T,T), (T,F), (F,T), (F,F)
        if ocultas:
            combinaciones = list(product([True, False], repeat=len(ocultas)))
            lista_ocultas = sorted(ocultas)
        else:
            # Si no hay variables ocultas, solo hay una "combinación" vacía
            combinaciones = [()]
            lista_ocultas = []
        
        # Para cada posible asignación de las variables ocultas...
        for asignacion_ocultas in combinaciones:
            # Construir una asignación completa de TODAS las variables
            valores = evidencia.copy()  # Empezamos con la evidencia conocida
            
            # Agregar los valores de las variables ocultas
            for i, var in enumerate(lista_ocultas):
                valores[var] = asignacion_ocultas[i]
            
            # Calcular P(consulta=True, evidencia, ocultas)
            valores[var_consulta] = True
            prob_true = self._probabilidad_conjunta(valores)
            prob_conjunta_true += prob_true  # Sumar a la marginal
            
            # Calcular P(consulta=False, evidencia, ocultas)
            valores[var_consulta] = False
            prob_false = self._probabilidad_conjunta(valores)
            prob_conjunta_false += prob_false  # Sumar a la marginal
        
        # Normalizar usando la regla de Bayes
        # P(consulta | evidencia) = P(consulta, evidencia) / P(evidencia)
        # donde P(evidencia) = P(consulta=True, evidencia) + P(consulta=False, evidencia)
        total = prob_conjunta_true + prob_conjunta_false
        if total == 0:
            return 0.0
        
        # Devolver la probabilidad correspondiente al valor consultado
        resultado = prob_conjunta_true / total if valor_consulta else prob_conjunta_false / total
        
        if verbose:
            print(f"P({var_consulta}=True | evidencia) = {prob_conjunta_true/total:.4f}")
            print(f"P({var_consulta}=False | evidencia) = {prob_conjunta_false/total:.4f}")
        
        return resultado
    
    def _probabilidad_conjunta(self, valores):
        """
        Calcula la probabilidad conjunta P(X1=x1, X2=x2, ..., Xn=xn).
        Usa la regla de la cadena: P(X1,...,Xn) = Π P(Xi | padres(Xi))
        """
        # Inicializar probabilidad en 1.0 (elemento neutro de la multiplicación)
        prob = 1.0
        
        # Recorrer todos los nodos en orden topológico (padres antes que hijos)
        # Multiplicar P(Xi | padres(Xi)) para cada nodo
        # Ejemplo: P(Robo, Terremoto, Alarma, Llamada) = 
        #          P(Robo) × P(Terremoto) × P(Alarma|Robo,Terremoto) × P(Llamada|Alarma)
        for var in self.orden_topologico:
            nodo = self.nodos[var]
            # Obtener P(var=valor | padres) y multiplicarlo al producto acumulado
            prob *= nodo.probabilidad(valores[var], valores)
        
        return prob

def crear_red_alarma_simple():
    """
    Crea una red bayesiana clásica simplificada de alarma antirrobo.
    
    Estructura:
        Robo    Terremoto
          \       /
           Alarma
             |
          Llamada
    """
    # ========== NODOS RAÍZ (SIN PADRES) ==========
    
    # Nodo Robo (sin padres): representa la probabilidad a priori de un robo
    nodo_robo = NodoRedBayesiana(
        nombre='Robo',
        padres=[],
        cpt={(): 0.001}  # P(Robo=True) = 0.1% (robo es poco probable)
    )
    
    # Nodo Terremoto (sin padres): probabilidad a priori de un terremoto
    nodo_terremoto = NodoRedBayesiana(
        nombre='Terremoto',
        padres=[],
        cpt={(): 0.002}  # P(Terremoto=True) = 0.2% (terremoto es poco probable)
    )
    
    # ========== NODO INTERMEDIO ==========
    
    # Nodo Alarma (padres: Robo, Terremoto)
    # La alarma puede sonar por un robo, un terremoto, o ambos
    nodo_alarma = NodoRedBayesiana(
        nombre='Alarma',
        padres=['Robo', 'Terremoto'],
        cpt={
            # Formato: (Robo, Terremoto): P(Alarma=True | Robo, Terremoto)
            (True, True): 0.95,    # Si hay robo Y terremoto, alarma suena con 95% prob.
            (True, False): 0.94,   # Si hay robo pero NO terremoto, 94% de prob.
            (False, True): 0.29,   # Si NO hay robo pero SÍ terremoto, 29% de prob.
            (False, False): 0.001  # Si NO hay ni robo ni terremoto, solo 0.1% (falsa alarma)
        }
    )
    
    # ========== NODO HOJA ==========
    
    # Nodo Llamada (padre: Alarma): alguien llama si escucha la alarma
    nodo_llamada = NodoRedBayesiana(
        nombre='Llamada',
        padres=['Alarma'],
        cpt={
            # Formato: (Alarma,): P(Llamada=True | Alarma)
            (True,): 0.90,   # Si la alarma suena, alguien llama con 90% de prob.
            (False,): 0.05   # Si la alarma NO suena, solo 5% de prob. de llamada
        }
    )
    
    # ========== CONSTRUCCIÓN DE LA RED ==========
    
    # Crear objeto red y agregar todos los nodos
    red = RedBayesiana()
    red.agregar_nodo(nodo_robo)
    red.agregar_nodo(nodo_terremoto)
    red.agregar_nodo(nodo_alarma)
    red.agregar_nodo(nodo_llamada)
    
    # Establecer orden topológico: padres antes que hijos
    # Este orden es crucial para calcular probabilidades conjuntas correctamente
    red.establecer_orden_topologico(['Robo', 'Terremoto', 'Alarma', 'Llamada'])
    
    return red

def modo_demo():
    """Ejecuta el modo demostrativo con red bayesiana predefinida."""
    print("\n" + "="*70)
    print("MODO DEMO: Red Bayesiana - Sistema de Alarma")
    print("="*70)
    
    # Crear la red bayesiana de alarma
    red = crear_red_alarma_simple()
    
    print("\n--- Estructura de la Red ---")
    print("Robo (P=0.001) ──┐")
    print("                 ├──> Alarma ──> Llamada (P=0.90|Alarma)")
    print("Terremoto (P=0.002) ──┘")
    
    print("\n--- Tablas de Probabilidad Condicional ---")
    print("\nP(Robo=True) = 0.001")
    print("P(Terremoto=True) = 0.002")
    print("\nP(Alarma=True | Robo, Terremoto):")
    print("  Robo=T, Terremoto=T: 0.95")
    print("  Robo=T, Terremoto=F: 0.94")
    print("  Robo=F, Terremoto=T: 0.29")
    print("  Robo=F, Terremoto=F: 0.001")
    print("\nP(Llamada=True | Alarma):")
    print("  Alarma=True:  0.90")
    print("  Alarma=False: 0.05")
    
    # ========== CONSULTA 1: P(Robo | Llamada=True) ==========
    # Pregunta: Si recibo una llamada, ¿qué tan probable es que haya un robo?
    print("\n" + "-"*70)
    print("CONSULTA 1: ¿Cuál es la probabilidad de robo si recibo una llamada?")
    print("P(Robo=True | Llamada=True)")
    print("-"*70)
    
    # Realizar inferencia: consultar P(Robo=True) dado que observamos Llamada=True
    prob = red.inferencia_por_enumeracion(
        consulta={'Robo': True},      # Variable de interés: Robo
        evidencia={'Llamada': True},  # Evidencia observada: alguien llamó
        verbose=True                   # Mostrar pasos intermedios
    )
    print(f"\n>>> Resultado: P(Robo=True | Llamada=True) = {prob:.6f}")
    print(f"    Interpretación: Hay un {prob*100:.4f}% de probabilidad de robo.")
    print(f"    (La probabilidad aumenta desde 0.1% a priori hasta ~1.6%)")
    
    # ========== CONSULTA 2: P(Robo | Llamada=True, Terremoto=False) ==========
    # Pregunta: Si recibo una llamada Y sé que NO hubo terremoto, ¿qué tan probable es el robo?
    # Esta consulta elimina la explicación alternativa del terremoto
    print("\n" + "-"*70)
    print("CONSULTA 2: ¿Probabilidad de robo si hay llamada pero NO hubo terremoto?")
    print("P(Robo=True | Llamada=True, Terremoto=False)")
    print("-"*70)
    
    # Realizar inferencia con más evidencia
    prob2 = red.inferencia_por_enumeracion(
        consulta={'Robo': True},                            # Variable de interés
        evidencia={'Llamada': True, 'Terremoto': False},   # Evidencia adicional
        verbose=True
    )
    print(f"\n>>> Resultado: P(Robo=True | Llamada=True, Terremoto=False) = {prob2:.6f}")
    print(f"    Interpretación: Aumenta a {prob2*100:.4f}% (descartamos terremoto).")
    print(f"    (Al descartar terremoto, la única explicación razonable es el robo)")
    
    # ========== CONSULTA 3: P(Alarma | Llamada=True) ==========
    # Pregunta: Si recibo una llamada, ¿qué tan probable es que la alarma haya sonado?
    # (Puede haber llamadas sin que la alarma suene, con pequeña probabilidad)
    print("\n" + "-"*70)
    print("CONSULTA 3: ¿Probabilidad de que la alarma haya sonado dado que hubo llamada?")
    print("P(Alarma=True | Llamada=True)")
    print("-"*70)
    
    # Consultar sobre un nodo intermedio
    prob3 = red.inferencia_por_enumeracion(
        consulta={'Alarma': True},    # Ahora consultamos sobre Alarma
        evidencia={'Llamada': True},  # Dada la evidencia de la llamada
        verbose=True
    )
    print(f"\n>>> Resultado: P(Alarma=True | Llamada=True) = {prob3:.6f}")
    print(f"    (Solo ~4.3% porque hay llamadas falsas con 5% de probabilidad)")

def modo_interactivo():
    """Ejecuta el modo interactivo con consultas del usuario."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Red Bayesiana - Sistema de Alarma")
    print("="*70)
    
    # Cargar la red bayesiana
    red = crear_red_alarma_simple()
    
    print("\nRed cargada: Sistema de Alarma")
    print("Variables disponibles: Robo, Terremoto, Alarma, Llamada")
    
    # ========== PASO 1: DEFINIR EVIDENCIA ==========
    # El usuario especifica qué variables ha observado y sus valores
    print("\n--- Definir Evidencia ---")
    print("Ingresa las variables observadas (deja en blanco para terminar)")
    
    evidencia = {}
    # Preguntar por cada variable observable
    for var in ['Llamada', 'Alarma', 'Terremoto']:
        respuesta = input(f"¿{var} observado? (s/n/enter para omitir): ").strip().lower()
        if respuesta == 's':
            evidencia[var] = True   # Variable observada como verdadera
        elif respuesta == 'n':
            evidencia[var] = False  # Variable observada como falsa
        # Si presiona enter, no se agrega a la evidencia (no observada)
    
    # Si no se definió ninguna evidencia, usar una por defecto
    if not evidencia:
        print("No se definió evidencia. Usando Llamada=True por defecto.")
        evidencia = {'Llamada': True}
    
    print(f"\nEvidencia definida: {evidencia}")
    
    # ========== PASO 2: DEFINIR CONSULTA ==========
    # El usuario especifica sobre qué variable desea conocer la probabilidad
    print("\n--- Definir Consulta ---")
    print("Variables disponibles para consulta: Robo, Terremoto, Alarma")
    var_consulta = input("¿Sobre qué variable deseas consultar? (default: Robo): ").strip() or 'Robo'
    
    # Validar que la variable sea válida
    if var_consulta not in ['Robo', 'Terremoto', 'Alarma']:
        print(f"Variable '{var_consulta}' no reconocida. Usando 'Robo'.")
        var_consulta = 'Robo'
    
    # ========== PASO 3: REALIZAR INFERENCIA ==========
    print(f"\nCalculando P({var_consulta}=True | {evidencia})...")
    
    # Ejecutar el algoritmo de inferencia por enumeración
    prob = red.inferencia_por_enumeracion(
        consulta={var_consulta: True},  # Consultar P(var=True)
        evidencia=evidencia,             # Dada la evidencia observada
        verbose=True                     # Mostrar detalles del cálculo
    )
    
    # ========== PASO 4: MOSTRAR RESULTADO ==========
    print(f"\n{'='*70}")
    print(f"RESULTADO: P({var_consulta}=True | evidencia) = {prob:.6f} ({prob*100:.2f}%)")
    print(f"{'='*70}")

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("RAZONAMIENTO PROBABILÍSTICO: REDES BAYESIANAS")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (consultas predefinidas sobre red de alarma)")
    print("2. INTERACTIVO (define tu propia evidencia y consulta)")
    
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
