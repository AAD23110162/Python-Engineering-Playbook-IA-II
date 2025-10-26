"""
010-E2-Probabilidad_condicionada_normalizacion.py
--------------------------------
Este script implementa Probabilidad Condicionada y Normalización:
- Calcula probabilidades condicionadas P(A|B) a partir de probabilidades conjuntas
- Implementa el proceso de normalización para garantizar distribuciones válidas
- Aplica la regla de la cadena para descomponer probabilidades conjuntas
- Calcula probabilidades marginales mediante suma sobre variables
- Maneja tablas de probabilidad condicional (CPT) y su normalización
- Muestra la relación entre probabilidad conjunta, marginal y condicionada
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente ejemplos de cálculo de probabilidades condicionadas
2. INTERACTIVO: permite al usuario ingresar eventos y calcular probabilidades condicionadas

Autor: Alejandro Aguirre Díaz
"""

import numpy as np
from itertools import product

# ========== TABLA DE PROBABILIDAD CONJUNTA ==========

class TablaProbabilidadConjunta:
    """
    Representa una tabla de probabilidad conjunta P(X, Y).
    Permite calcular probabilidades marginales y condicionadas.
    """
    
    def __init__(self, variables, valores, probabilidades):
        """
        Inicializa la tabla de probabilidad conjunta.
        
        :param variables: lista de nombres de variables ['X', 'Y']
        :param valores: dict {variable: lista_de_valores}
        :param probabilidades: dict {(val_x, val_y): probabilidad}
        """
        self.variables = variables
        self.valores = valores
        self.probabilidades = probabilidades
        
        # Normalizar si es necesario
        suma = sum(self.probabilidades.values())
        if abs(suma - 1.0) > 0.001:
            print(f"Advertencia: La suma de probabilidades es {suma:.4f}. Normalizando...")
            self.normalizar()
    
    def normalizar(self):
        """Normaliza la tabla para que las probabilidades sumen 1."""
        suma_total = sum(self.probabilidades.values())
        if suma_total > 0:
            for key in self.probabilidades:
                self.probabilidades[key] /= suma_total
    
    def prob_conjunta(self, asignacion):
        """
        Obtiene P(asignacion) de la tabla.
        
        :param asignacion: tupla de valores (val_x, val_y, ...)
        :return: probabilidad conjunta
        """
        return self.probabilidades.get(tuple(asignacion), 0.0)
    
    def prob_marginal(self, variable, valor):
        """
        Calcula probabilidad marginal P(variable=valor) sumando sobre otras variables.
        
        :param variable: nombre de la variable
        :param valor: valor de la variable
        :return: probabilidad marginal
        """
        # Encontrar índice de la variable
        idx_var = self.variables.index(variable)
        
        # Sumar sobre todas las asignaciones donde variable=valor
        prob_total = 0.0
        for asignacion, prob in self.probabilidades.items():
            if asignacion[idx_var] == valor:
                prob_total += prob
        
        return prob_total
    
    def prob_condicionada(self, variable_consulta, valor_consulta, evidencia):
        """
        Calcula P(variable_consulta=valor_consulta | evidencia).
        
        :param variable_consulta: variable a consultar
        :param valor_consulta: valor deseado
        :param evidencia: dict {variable: valor} con las variables observadas
        :return: probabilidad condicionada
        """
        # Encontrar índice de la variable de consulta
        idx_consulta = self.variables.index(variable_consulta)
        
        # Calcular P(consulta, evidencia) - probabilidad conjunta
        prob_conjunta = 0.0
        prob_evidencia = 0.0
        
        for asignacion, prob in self.probabilidades.items():
            # Verificar si la asignación es consistente con la evidencia
            consistente = all(
                asignacion[self.variables.index(var)] == val
                for var, val in evidencia.items()
            )
            
            if consistente:
                # Sumar a P(evidencia)
                prob_evidencia += prob
                
                # Si además la variable de consulta tiene el valor deseado, sumar a P(consulta, evid.)
                if asignacion[idx_consulta] == valor_consulta:
                    prob_conjunta += prob
        
        # P(consulta | evidencia) = P(consulta, evidencia) / P(evidencia)
        if prob_evidencia > 0:
            return prob_conjunta / prob_evidencia
        else:
            return 0.0
    
    def mostrar(self):
        """Muestra la tabla de probabilidad conjunta."""
        print("\n--- Tabla de Probabilidad Conjunta ---")
        # Encabezado
        encabezado = " | ".join(self.variables) + " | Probabilidad"
        print(encabezado)
        print("-" * len(encabezado))
        
        # Filas
        for asignacion, prob in sorted(self.probabilidades.items()):
            valores_str = " | ".join(str(v) for v in asignacion)
            print(f"{valores_str} | {prob:.4f}")

# ========== TABLA DE PROBABILIDAD CONDICIONAL (CPT) ==========

class TablaProbabilidadCondicional:
    """
    Representa una CPT (Conditional Probability Table) P(X | Padres).
    """
    
    def __init__(self, variable, padres, valores_variable, valores_padres):
        """
        Inicializa una CPT.
        
        :param variable: nombre de la variable
        :param padres: lista de nombres de variables padre
        :param valores_variable: posibles valores de la variable
        :param valores_padres: dict {padre: lista_de_valores}
        """
        self.variable = variable
        self.padres = padres
        self.valores_variable = valores_variable
        self.valores_padres = valores_padres
        
        # Tabla de probabilidades: {(padres_valores): {variable_valor: prob}}
        self.tabla = {}
    
    def establecer_probabilidades(self, asignacion_padres, probabilidades):
        """
        Establece probabilidades P(variable | asignacion_padres).
        
        :param asignacion_padres: tupla de valores de padres
        :param probabilidades: dict {valor_variable: probabilidad}
        """
        # Normalizar
        suma = sum(probabilidades.values())
        if abs(suma - 1.0) > 0.001:
            probabilidades = {k: v/suma for k, v in probabilidades.items()}
        
        self.tabla[tuple(asignacion_padres)] = probabilidades
    
    def obtener_probabilidad(self, valor_variable, asignacion_padres):
        """
        Obtiene P(variable=valor_variable | padres=asignacion_padres).
        
        :param valor_variable: valor de la variable
        :param asignacion_padres: tupla de valores de padres
        :return: probabilidad
        """
        if tuple(asignacion_padres) in self.tabla:
            return self.tabla[tuple(asignacion_padres)].get(valor_variable, 0.0)
        return 0.0
    
    def normalizar_fila(self, asignacion_padres):
        """Normaliza una fila específica de la CPT."""
        if tuple(asignacion_padres) in self.tabla:
            suma = sum(self.tabla[tuple(asignacion_padres)].values())
            if suma > 0:
                for valor in self.tabla[tuple(asignacion_padres)]:
                    self.tabla[tuple(asignacion_padres)][valor] /= suma
    
    def mostrar(self):
        """Muestra la CPT."""
        print(f"\n--- Tabla de Probabilidad Condicional: P({self.variable} | {', '.join(self.padres)}) ---")
        
        # Encabezado
        encabezado = " | ".join(self.padres)
        for val in self.valores_variable:
            encabezado += f" | P({val})"
        print(encabezado)
        print("-" * len(encabezado))
        
        # Filas
        for asignacion, probs in sorted(self.tabla.items()):
            fila = " | ".join(str(v) for v in asignacion)
            for val in self.valores_variable:
                fila += f" | {probs.get(val, 0.0):.3f}"
            print(fila)

# ========== FUNCIONES AUXILIARES ==========

def calcular_prob_marginal_desde_conjunta(tabla_conjunta, variable, valor):
    """
    Calcula P(variable=valor) marginalizando sobre otras variables.
    Ejemplo de marginalización: P(X=x) = Σ_y P(X=x, Y=y)
    """
    return tabla_conjunta.prob_marginal(variable, valor)

def aplicar_regla_cadena(probabilidades_cond):
    """
    Aplica la regla de la cadena: P(X1, X2, ..., Xn) = P(X1) × P(X2|X1) × P(X3|X1,X2) × ...
    
    :param probabilidades_cond: lista de probabilidades en orden
    :return: probabilidad conjunta
    """
    prob_conjunta = 1.0
    for prob in probabilidades_cond:
        prob_conjunta *= prob
    return prob_conjunta

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con ejemplos de probabilidades condicionadas."""
    print("\n" + "="*70)
    print("MODO DEMO: Probabilidad Condicionada y Normalización")
    print("="*70)
    
    # ========== EJEMPLO 1: Probabilidad Conjunta y Marginal ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: De Probabilidad Conjunta a Marginal")
    print("="*70)
    
    print("\n--- Escenario: Clima (X) y Tráfico (Y) ---")
    print("X: Clima = {Soleado, Lluvioso}")
    print("Y: Tráfico = {Ligero, Pesado}")
    
    # Definir tabla de probabilidad conjunta
    variables = ['Clima', 'Tráfico']
    valores = {
        'Clima': ['Soleado', 'Lluvioso'],
        'Tráfico': ['Ligero', 'Pesado']
    }
    
    prob_conjuntas = {
        ('Soleado', 'Ligero'): 0.4,
        ('Soleado', 'Pesado'): 0.2,
        ('Lluvioso', 'Ligero'): 0.1,
        ('Lluvioso', 'Pesado'): 0.3
    }
    
    tabla = TablaProbabilidadConjunta(variables, valores, prob_conjuntas)
    tabla.mostrar()
    
    print("\n--- Probabilidades Marginales (Marginalizando sobre otra variable) ---")
    
    # P(Clima = Soleado) = P(Soleado, Ligero) + P(Soleado, Pesado)
    p_soleado = tabla.prob_marginal('Clima', 'Soleado')
    p_lluvioso = tabla.prob_marginal('Clima', 'Lluvioso')
    print(f"P(Clima=Soleado) = {p_soleado:.2f}")
    print(f"P(Clima=Lluvioso) = {p_lluvioso:.2f}")
    
    # P(Tráfico = Pesado) = P(Soleado, Pesado) + P(Lluvioso, Pesado)
    p_ligero = tabla.prob_marginal('Tráfico', 'Ligero')
    p_pesado = tabla.prob_marginal('Tráfico', 'Pesado')
    print(f"\nP(Tráfico=Ligero) = {p_ligero:.2f}")
    print(f"P(Tráfico=Pesado) = {p_pesado:.2f}")
    
    print("\n>>> OBSERVACIÓN:")
    print("    La marginalización suma sobre los valores de las variables no consultadas.")
    
    # ========== EJEMPLO 2: Probabilidad Condicionada ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Cálculo de Probabilidad Condicionada")
    print("="*70)
    
    print("\n--- P(Tráfico | Clima) ---")
    
    # P(Tráfico=Pesado | Clima=Lluvioso)
    p_pesado_dado_lluvia = tabla.prob_condicionada('Tráfico', 'Pesado', {'Clima': 'Lluvioso'})
    print(f"P(Tráfico=Pesado | Clima=Lluvioso) = {p_pesado_dado_lluvia:.3f}")
    
    # Verificar manualmente: P(T=P|C=Ll) = P(C=Ll, T=P) / P(C=Ll)
    p_conj = prob_conjuntas[('Lluvioso', 'Pesado')]
    p_evidencia = p_lluvioso
    verificacion = p_conj / p_evidencia
    print(f"Verificación: P(Ll, Pesado)/P(Ll) = {p_conj:.2f}/{p_evidencia:.2f} = {verificacion:.3f}")
    
    # P(Tráfico=Ligero | Clima=Soleado)
    p_ligero_dado_sol = tabla.prob_condicionada('Tráfico', 'Ligero', {'Clima': 'Soleado'})
    print(f"\nP(Tráfico=Ligero | Clima=Soleado) = {p_ligero_dado_sol:.3f}")
    
    print("\n>>> FÓRMULA:")
    print("    P(A|B) = P(A, B) / P(B)")
    print("    La probabilidad condicionada normaliza la prob. conjunta por la prob. de la evidencia.")
    
    # ========== EJEMPLO 3: Normalización de Distribuciones ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Normalización de Distribuciones No Normalizadas")
    print("="*70)
    
    print("\n--- Escenario: Distribución sin normalizar ---")
    
    # Supon que medimos frecuencias pero no suman 1
    freq_no_norm = {
        ('A', 1): 5,
        ('A', 2): 3,
        ('B', 1): 2,
        ('B', 2): 6
    }
    
    print("Frecuencias observadas (no normalizadas):")
    for key, freq in freq_no_norm.items():
        print(f"  {key}: {freq}")
    
    print(f"\nSuma total: {sum(freq_no_norm.values())}")
    
    # Normalizar
    tabla_norm = TablaProbabilidadConjunta(['X', 'Y'], {'X': ['A', 'B'], 'Y': [1, 2]}, freq_no_norm)
    
    print("\nDespués de normalizar:")
    tabla_norm.mostrar()
    
    print(f"Suma de probabilidades: {sum(tabla_norm.probabilidades.values()):.4f} ✓")
    
    print("\n>>> OBSERVACIÓN:")
    print("    La normalización garantiza que las probabilidades sumen 1,")
    print("    convirtiendo frecuencias o valores no normalizados en distribución válida.")
    
    # ========== EJEMPLO 4: Regla de la Cadena ==========
    print("\n" + "="*70)
    print("EJEMPLO 4: Regla de la Cadena")
    print("="*70)
    
    print("\n--- Descomponiendo P(A, B, C) ---")
    print("Regla de la cadena: P(A, B, C) = P(A) × P(B|A) × P(C|A,B)")
    
    # Ejemplo con valores específicos
    P_A = 0.6
    P_B_dado_A = 0.7
    P_C_dado_AB = 0.8
    
    print(f"\nDado:")
    print(f"  P(A) = {P_A}")
    print(f"  P(B|A) = {P_B_dado_A}")
    print(f"  P(C|A,B) = {P_C_dado_AB}")
    
    # Aplicar regla de la cadena
    P_ABC = aplicar_regla_cadena([P_A, P_B_dado_A, P_C_dado_AB])
    print(f"\nP(A, B, C) = {P_A} × {P_B_dado_A} × {P_C_dado_AB} = {P_ABC:.4f}")
    
    print("\n>>> OBSERVACIÓN:")
    print("    La regla de la cadena permite descomponer probabilidades conjuntas complejas")
    print("    en productos de probabilidades condicionadas más simples.")
    
    # ========== EJEMPLO 5: Tabla CPT ==========
    print("\n" + "="*70)
    print("EJEMPLO 5: Tabla de Probabilidad Condicional (CPT)")
    print("="*70)
    
    print("\n--- Red Bayesiana: P(Alarma | Robo, Terremoto) ---")
    
    # Crear CPT
    cpt = TablaProbabilidadCondicional(
        variable='Alarma',
        padres=['Robo', 'Terremoto'],
        valores_variable=[True, False],
        valores_padres={'Robo': [True, False], 'Terremoto': [True, False]}
    )
    
    # Llenar la CPT
    cpt.establecer_probabilidades((True, True), {True: 0.95, False: 0.05})
    cpt.establecer_probabilidades((True, False), {True: 0.90, False: 0.10})
    cpt.establecer_probabilidades((False, True), {True: 0.85, False: 0.15})
    cpt.establecer_probabilidades((False, False), {True: 0.01, False: 0.99})
    
    cpt.mostrar()
    
    print("\n>>> INTERPRETACIÓN:")
    print("    Si hay Robo=True y Terremoto=True, P(Alarma=True) = 0.950")
    print("    Si no hay ni Robo ni Terremoto, P(Alarma=True) = 0.010 (falsa alarma)")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Calculadora de Probabilidad Condicionada")
    print("="*70)
    
    print("\n--- Definir tabla de probabilidad conjunta 2D ---")
    print("Variables: X e Y")
    
    # Obtener valores de X
    x_vals_input = input("Valores de X separados por comas (ej: A,B,C): ").strip()
    if x_vals_input:
        x_vals = [v.strip() for v in x_vals_input.split(',')]
    else:
        x_vals = ['A', 'B']
    
    # Obtener valores de Y
    y_vals_input = input("Valores de Y separados por comas (ej: 1,2): ").strip()
    if y_vals_input:
        y_vals = [v.strip() for v in y_vals_input.split(',')]
    else:
        y_vals = ['1', '2']
    
    print(f"\nX = {x_vals}")
    print(f"Y = {y_vals}")
    
    # Obtener probabilidades conjuntas
    print("\n--- Ingresar probabilidades conjuntas ---")
    prob_conj = {}
    
    for x in x_vals:
        for y in y_vals:
            try:
                p = float(input(f"P(X={x}, Y={y}): ").strip() or "0.25")
            except:
                p = 0.25
            prob_conj[(x, y)] = p
    
    # Crear tabla
    tabla = TablaProbabilidadConjunta(['X', 'Y'], {'X': x_vals, 'Y': y_vals}, prob_conj)
    tabla.mostrar()
    
    # Calcular probabilidad condicionada
    print("\n--- Calcular P(X | Y=y) ---")
    y_evidencia = input(f"Ingresa el valor de Y (opciones: {y_vals}): ").strip()
    
    if y_evidencia in y_vals:
        print(f"\nP(X | Y={y_evidencia}):")
        for x in x_vals:
            prob = tabla.prob_condicionada('X', x, {'Y': y_evidencia})
            print(f"  P(X={x} | Y={y_evidencia}) = {prob:.4f}")
    else:
        print("Valor de Y no válido.")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("PROBABILIDAD CONDICIONADA Y NORMALIZACIÓN")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (ejemplos de prob. conjunta, marginal, condicionada)")
    print("2. INTERACTIVO (calcula tus propias probabilidades condicionadas)")
    
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
