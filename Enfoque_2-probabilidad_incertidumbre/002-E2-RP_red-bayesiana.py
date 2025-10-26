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
        if self.padres:
            clave = tuple(valores_padres[p] for p in self.padres)
        else:
            clave = ()
        
        prob_true = self.cpt.get(clave, 0.5)  # Por defecto 0.5 si no está definido
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
        var_consulta = list(consulta.keys())[0]
        valor_consulta = consulta[var_consulta]
        
        # Variables ocultas (no en consulta ni evidencia)
        todas = set(self.nodos.keys())
        ocultas = todas - {var_consulta} - set(evidencia.keys())
        
        if verbose:
            print(f"\n--- Inferencia por Enumeración ---")
            print(f"Consulta: P({var_consulta}={valor_consulta} | {evidencia})")
            print(f"Variables ocultas: {sorted(ocultas)}")
        
        # Enumerar sobre todas las combinaciones de variables ocultas
        prob_conjunta_true = 0.0
        prob_conjunta_false = 0.0
        
        # Generar todas las asignaciones posibles de variables ocultas
        if ocultas:
            combinaciones = list(product([True, False], repeat=len(ocultas)))
            lista_ocultas = sorted(ocultas)
        else:
            combinaciones = [()]
            lista_ocultas = []
        
        for asignacion_ocultas in combinaciones:
            # Construir asignación completa
            valores = evidencia.copy()
            for i, var in enumerate(lista_ocultas):
                valores[var] = asignacion_ocultas[i]
            
            # Calcular para consulta=True
            valores[var_consulta] = True
            prob_true = self._probabilidad_conjunta(valores)
            prob_conjunta_true += prob_true
            
            # Calcular para consulta=False
            valores[var_consulta] = False
            prob_false = self._probabilidad_conjunta(valores)
            prob_conjunta_false += prob_false
        
        # Normalizar
        total = prob_conjunta_true + prob_conjunta_false
        if total == 0:
            return 0.0
        
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
        prob = 1.0
        for var in self.orden_topologico:
            nodo = self.nodos[var]
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
    # Nodo Robo (sin padres)
    nodo_robo = NodoRedBayesiana(
        nombre='Robo',
        padres=[],
        cpt={(): 0.001}  # P(Robo=True) = 0.001
    )
    
    # Nodo Terremoto (sin padres)
    nodo_terremoto = NodoRedBayesiana(
        nombre='Terremoto',
        padres=[],
        cpt={(): 0.002}  # P(Terremoto=True) = 0.002
    )
    
    # Nodo Alarma (padres: Robo, Terremoto)
    nodo_alarma = NodoRedBayesiana(
        nombre='Alarma',
        padres=['Robo', 'Terremoto'],
        cpt={
            (True, True): 0.95,    # P(Alarma | Robo, Terremoto)
            (True, False): 0.94,   # P(Alarma | Robo, ¬Terremoto)
            (False, True): 0.29,   # P(Alarma | ¬Robo, Terremoto)
            (False, False): 0.001  # P(Alarma | ¬Robo, ¬Terremoto)
        }
    )
    
    # Nodo Llamada (padre: Alarma)
    nodo_llamada = NodoRedBayesiana(
        nombre='Llamada',
        padres=['Alarma'],
        cpt={
            (True,): 0.90,   # P(Llamada | Alarma)
            (False,): 0.05   # P(Llamada | ¬Alarma)
        }
    )
    
    # Construir la red
    red = RedBayesiana()
    red.agregar_nodo(nodo_robo)
    red.agregar_nodo(nodo_terremoto)
    red.agregar_nodo(nodo_alarma)
    red.agregar_nodo(nodo_llamada)
    red.establecer_orden_topologico(['Robo', 'Terremoto', 'Alarma', 'Llamada'])
    
    return red

def modo_demo():
    """Ejecuta el modo demostrativo con red bayesiana predefinida."""
    print("\n" + "="*70)
    print("MODO DEMO: Red Bayesiana - Sistema de Alarma")
    print("="*70)
    
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
    
    # Consulta 1: P(Robo | Llamada=True)
    print("\n" + "-"*70)
    print("CONSULTA 1: ¿Cuál es la probabilidad de robo si recibo una llamada?")
    print("P(Robo=True | Llamada=True)")
    print("-"*70)
    
    prob = red.inferencia_por_enumeracion(
        consulta={'Robo': True},
        evidencia={'Llamada': True},
        verbose=True
    )
    print(f"\n>>> Resultado: P(Robo=True | Llamada=True) = {prob:.6f}")
    print(f"    Interpretación: Hay un {prob*100:.4f}% de probabilidad de robo.")
    
    # Consulta 2: P(Robo | Llamada=True, Terremoto=False)
    print("\n" + "-"*70)
    print("CONSULTA 2: ¿Probabilidad de robo si hay llamada pero NO hubo terremoto?")
    print("P(Robo=True | Llamada=True, Terremoto=False)")
    print("-"*70)
    
    prob2 = red.inferencia_por_enumeracion(
        consulta={'Robo': True},
        evidencia={'Llamada': True, 'Terremoto': False},
        verbose=True
    )
    print(f"\n>>> Resultado: P(Robo=True | Llamada=True, Terremoto=False) = {prob2:.6f}")
    print(f"    Interpretación: Aumenta a {prob2*100:.4f}% (descartamos terremoto).")
    
    # Consulta 3: P(Alarma | Llamada=True)
    print("\n" + "-"*70)
    print("CONSULTA 3: ¿Probabilidad de que la alarma haya sonado dado que hubo llamada?")
    print("P(Alarma=True | Llamada=True)")
    print("-"*70)
    
    prob3 = red.inferencia_por_enumeracion(
        consulta={'Alarma': True},
        evidencia={'Llamada': True},
        verbose=True
    )
    print(f"\n>>> Resultado: P(Alarma=True | Llamada=True) = {prob3:.6f}")

def modo_interactivo():
    """Ejecuta el modo interactivo con consultas del usuario."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Red Bayesiana - Sistema de Alarma")
    print("="*70)
    
    red = crear_red_alarma_simple()
    
    print("\nRed cargada: Sistema de Alarma")
    print("Variables disponibles: Robo, Terremoto, Alarma, Llamada")
    
    print("\n--- Definir Evidencia ---")
    print("Ingresa las variables observadas (deja en blanco para terminar)")
    
    evidencia = {}
    for var in ['Llamada', 'Alarma', 'Terremoto']:
        respuesta = input(f"¿{var} observado? (s/n/enter para omitir): ").strip().lower()
        if respuesta == 's':
            evidencia[var] = True
        elif respuesta == 'n':
            evidencia[var] = False
    
    if not evidencia:
        print("No se definió evidencia. Usando Llamada=True por defecto.")
        evidencia = {'Llamada': True}
    
    print(f"\nEvidencia definida: {evidencia}")
    
    print("\n--- Definir Consulta ---")
    print("Variables disponibles para consulta: Robo, Terremoto, Alarma")
    var_consulta = input("¿Sobre qué variable deseas consultar? (default: Robo): ").strip() or 'Robo'
    
    if var_consulta not in ['Robo', 'Terremoto', 'Alarma']:
        print(f"Variable '{var_consulta}' no reconocida. Usando 'Robo'.")
        var_consulta = 'Robo'
    
    print(f"\nCalculando P({var_consulta}=True | {evidencia})...")
    
    prob = red.inferencia_por_enumeracion(
        consulta={var_consulta: True},
        evidencia=evidencia,
        verbose=True
    )
    
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
