"""
026-E1-valor_información.py
--------------------------------------
Este script implementa el cálculo del Valor de la Información en decisiones bajo incertidumbre:
- Una alternativa con coste/beneficio incierto; se evalúa cuánto vale **obtener información adicional** antes de decidir.
- Modo DEMO: escenario predefinido con cálculo visible paso a paso.
- Modo INTERACTIVO: usuario define sus probabilidades, ganancias/perdidas y opciones de información.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def utilidad(valor):
    """
    Función de utilidad simple (lineal) para este ejemplo.
    :parametro valor: resultado cuantitativo
    :return: utilidad (float)
    """
    # Utilidad lineal: la utilidad es igual al valor
    return valor

def utilidad_esperada(probabilidades, valores):
    """
    Calcula utilidad esperada de un conjunto de probabilidades y valores.
    :parametro probabilidades: lista de probabilidades que suman 1
    :parametro valores: lista de resultados cuantitativos correspondientes
    :return: utilidad esperada (float)
    """
    # Inicializar acumulador de utilidad esperada
    ue = 0.0
    # Sumar la contribución de cada resultado: probabilidad × utilidad del valor
    for p, v in zip(probabilidades, valores):
        ue += p * utilidad(v)
    return ue

def valor_de_la_informacion(prob_original, valores_original, costo_info, prob_condicionales=None, valores_condicionales=None):
    """
    Calcula el Valor Esperado de Información (VEI) de una información adicional:
    - prob_original / valores_original definen el escenario sin información.
    - costo_info es el coste de obtener esa información.
    - Si se tiene prob_condicionales y valores_condicionales para cada "estado de info", se calcula utilidad esperada con info menos coste.
    :parametro prob_original: lista de probabilidades antes de información
    :parametro valores_original: lista de valores correspondientes sin info
    :parametro costo_info: coste de la información
    :parametro prob_condicionales: lista de listas de probabilidades condicionadas para cada posible "estado de información"
    :parametro valores_condicionales: lista de listas de valores correspondientes para cada estado de información
    :return: VEI (float)
    """
    # Paso 1: Calcular utilidad esperada SIN información adicional
    ue_sin = utilidad_esperada(prob_original, valores_original)
    print(f"Utilidad esperada sin información: {ue_sin:.3f}")

    # Si no hay información condicional definida, retornar utilidad base
    if prob_condicionales is None or valores_condicionales is None:
        return ue_sin  # sin información adicional definida

    # Paso 2: Calcular utilidad esperada CON información adicional
    ue_con_info = 0.0
    estados_info = len(prob_condicionales)
    
    # Para cada posible estado que la información puede revelar
    for i in range(estados_info):
        # Probabilidad de que se revele este estado de información
        p_info = prob_condicionales[i][0]  # asumimos que primer elemento es la probabilidad del estado de info
        # Utilidad esperada condicional dado este estado de información
        u_info = utilidad_esperada(prob_condicionales[i][1], valores_condicionales[i])
        print(f"  Estado de información {i+1}: P = {p_info:.3f}, utilidad condicional = {u_info:.3f}")
        # Acumular: probabilidad del estado × utilidad condicional
        ue_con_info += p_info * u_info
    
    # Restar el coste de obtener la información
    ue_con_info -= costo_info
    print(f"Utilidad esperada con información menos coste: {ue_con_info:.3f}")

    # Paso 3: Calcular el Valor Esperado de la Información (VEI)
    # VEI = (utilidad con info - coste) - (utilidad sin info)
    vei = ue_con_info - ue_sin
    print(f"→ Valor esperado de la información (VEI) = {vei:.3f}")
    return vei

def modo_demo():
    print("\n--- MODO DEMO ---")
    # Ejemplo: decisión de inversión con información adicional
    # Sin información adicional, hay 60% de probabilidad de buen resultado (+100)
    # y 40% de probabilidad de mal resultado (-50)
    prob_original = [0.6, 0.4]           # sin información: 60% buen resultado, 40% malo
    valores_original = [100, -50]        # valor: 100 si buen resultado, -50 si malo
    costo_info = 10                      # coste de obtener información adicional
    
    # Supongamos que la información revela dos estados: "info1" con prob 0.7, "info2" con prob 0.3
    # En cada estado, las probabilidades de los resultados cambian (información condicional)
    prob_cond = [
        [0.7, ( [0.8, 0.2] )],            # en info1 (70% de ocurrir): 80% buen resultado, 20% malo
        [0.3, ( [0.3, 0.7] )]             # en info2 (30% de ocurrir): 30% buen resultado, 70% malo
    ]
    valores_cond = [
        [100, -50],                      # mismos valores condicionales en info1
        [100, -50]                       # mismos valores condicionales en info2
    ]
    print("Sin información adicional:")
    print(f"  Probabilidades: {prob_original}, valores: {valores_original}")
    print(f"Con opción de información (coste = {costo_info}): dos posibles estados de info con probabilidades 0.7 y 0.3.")
    
    # Calcular el valor esperado de la información
    valor_de_la_informacion(prob_original, valores_original, costo_info, prob_cond, valores_cond)

def modo_interactivo():
    print("\n--- MODO INTERACTIVO ---")
    print("Vamos a evaluar si vale la pena obtener información adicional antes de tomar una decisión.\n")
    
    # Paso 1: Escenario base (sin información adicional)
    print("PASO 1: Define tu decisión SIN información adicional")
    print("-" * 50)
    n = int(input("¿Cuántos resultados posibles tiene tu decisión? (ej: 2 = éxito/fracaso): ").strip())
    
    prob_original = []
    valores_original = []
    print(f"\nAhora ingresa los {n} resultados posibles:")
    for i in range(n):
        print(f"\n  Resultado {i+1}:")
        p = float(input(f"    ¿Probabilidad? (entre 0 y 1): ").strip())
        v = float(input(f"    ¿Valor/ganancia? (puede ser negativo si es pérdida): ").strip())
        prob_original.append(p)
        valores_original.append(v)
    
    # Mostrar resumen del escenario base
    ue_sin = utilidad_esperada(prob_original, valores_original)
    print(f"\n  → Utilidad esperada sin información adicional: {ue_sin:.2f}")
    
    # Paso 2: Coste de obtener información
    print("\n" + "="*50)
    print("PASO 2: Coste de obtener información adicional")
    print("-" * 50)
    costo_info = float(input("¿Cuánto cuesta obtener la información adicional?: ").strip())
    
    # Paso 3: Información condicional (simplificado)
    print("\n" + "="*50)
    print("PASO 3: ¿Qué información adicional puedes obtener?")
    print("-" * 50)
    print("La información adicional puede revelar diferentes 'señales' o 'estados'.")
    print("Por ejemplo: 'señal positiva' vs 'señal negativa', o 'mercado favorable' vs 'mercado desfavorable'\n")
    
    m = int(input("¿Cuántas señales/estados diferentes puede revelar la información? (típicamente 2): ").strip())
    
    prob_cond = []
    valores_cond = []
    
    print(f"\nAhora vamos a definir cada una de las {m} señales:\n")
    for j in range(m):
        print(f"SEÑAL {j+1}:")
        p_senal = float(input(f"  ¿Probabilidad de que ocurra esta señal? (0-1): ").strip())
        
        print(f"  Si recibes esta señal, ¿cómo cambian las probabilidades de tus {n} resultados?")
        probs_cond_senal = []
        for i in range(n):
            p_cond = float(input(f"    Prob. de resultado {i+1} dado esta señal (0-1): ").strip())
            probs_cond_senal.append(p_cond)
        
        # Usar los mismos valores originales (solo cambian probabilidades)
        prob_cond.append([p_senal, probs_cond_senal])
        valores_cond.append(valores_original.copy())
        print()
    
    # Paso 4: Calcular el valor de la información
    print("="*50)
    print("CALCULANDO VALOR DE LA INFORMACIÓN...")
    print("="*50)
    valor_de_la_informacion(prob_original, valores_original, costo_info, prob_cond, valores_cond)
    
    # Interpretación final
    print("\n" + "="*50)
    print("INTERPRETACIÓN:")
    print("-" * 50)
    vei = valor_de_la_informacion(prob_original, valores_original, costo_info, prob_cond, valores_cond)
    if vei > 0:
        print(f"✓ VALE LA PENA obtener la información (ganancia esperada: {vei:.2f})")
    elif vei < 0:
        print(f"✗ NO VALE LA PENA obtener la información (pérdida esperada: {vei:.2f})")
    else:
        print("○ Es indiferente obtener o no la información (ganancia/pérdida: 0)")

def main():
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    
    # Leer la opción del usuario
    opcion = input("Ingrese el número de opción: ").strip()
    
    # Ejecutar el modo correspondiente según la opción elegida
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        # Opción no válida: ejecutar DEMO por defecto
        print("Opción no válida. Se usará MODO DEMO.")
        modo_demo()

# Punto de entrada del programa
if __name__ == "__main__":
    main()
