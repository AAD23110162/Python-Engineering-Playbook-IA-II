"""
024-E1-teoria_de_la_utilidad.py
--------------------------------
Este script implementa una versión simplificada de la Teoría de la Utilidad / Utilidad Esperada:
- Permite al usuario definir varias alternativas con probabilidades y resultados.
- Calcula la utilidad esperada si se define una función de utilidad (lineal o no).
- Muestra en modo DEMO un conjunto predefinido, y en modo INTERACTIVO permite ingresar datos.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

def funcion_utilidad(valor, tipo='lineal', parametro=1.0):
    """
    Función de utilidad u(x) que transforma un resultado cuantitativo en utilidad subjetiva.
    :parametro valor: resultado cuantitativo (por ejemplo ganancia monetaria)
    :parametro tipo: tipo de función de utilidad ('lineal', 'logaritmica', 'exponencial')
    :parametro parametro: parámetro adicional para la función
    :return: utilidad correspondiente (float)
    """
    # Función de utilidad lineal: la utilidad es igual al valor
    if tipo == 'lineal':
        return valor
    # Función logarítmica: modelo de aversión al riesgo decreciente
    elif tipo == 'logaritmica':
        # Evita valor <=0 para logaritmo (no está definido)
        return parametro * (valor if valor>0 else 0)
    # Función exponencial: modelo de aversión al riesgo constante
    elif tipo == 'exponencial':
        import math
        # u(x) = 1 - e^(-parametro * x)
        return 1 - math.exp(- parametro * valor)
    else:
        # Por defecto, utilizar función lineal
        return valor

def utilidad_esperada(alternativas, tipo='lineal', parametro=1.0, verbose=False):
    """
    Calcula la utilidad esperada de un conjunto de alternativas.
    :parametro alternativas: lista de tuplas (probabilidad, resultado cuantitativo)
    :parametro tipo: tipo de función de utilidad para todos los resultados
    :parametro parametro: parámetro de la función de utilidad
    :parametro verbose: si True, muestra paso a paso el cálculo
    :return: utilidad esperada (float)
    """
    # Inicializar el acumulador de utilidad esperada
    ue = 0.0
    # Contador de suma de probabilidades para verificación (debe ser ~1.0)
    suma_p = 0.0

    # Si verbose está activo, mostrar encabezado del proceso
    if verbose:
        print("\n[PROCESO] Cálculo de Utilidad Esperada")
        print(f"  Función de utilidad: {tipo} (parámetro={parametro})")
        print("  Alternativas (p, x):")
        # Listar todas las alternativas antes de calcular
        for idx, (p0, x0) in enumerate(alternativas, start=1):
            print(f"    {idx:>2}) p={p0:.6f}, x={x0}")
        print("  Paso a paso:")

    # Procesar cada alternativa: calcular u(x) y su contribución p*u(x)
    for i, (p, valor) in enumerate(alternativas, start=1):
        # Calcular la utilidad del resultado usando la función de utilidad elegida
        u = funcion_utilidad(valor, tipo, parametro)
        # Contribución de esta alternativa a la utilidad esperada: p * u(x)
        contrib = p * u
        # Acumular en la utilidad esperada total
        ue += contrib
        # Acumular probabilidades para verificación
        suma_p += p
        # Si verbose, mostrar el detalle de este paso
        if verbose:
            print(f"    [{i:>2}] u(x={valor})={u:.6f} -> p*u={p:.6f}*{u:.6f}={contrib:.6f} | acumulado={ue:.6f}")

    # Si verbose, mostrar resumen final
    if verbose:
        print(f"  Suma de probabilidades: {suma_p:.6f}")
        print(f"  Utilidad esperada total: {ue:.6f}")

    return ue

def modo_demo():
    print("\n--- MODO DEMO ---")
    # Conjunto predefinido de alternativas: 50% de ganar 100, 50% de ganar 0
    alternativas = [
        (0.5, 100),  # 50% de ganar 100
        (0.5, 0)     # 50% de ganar 0
    ]
    print("Alternativas:", alternativas)
    
    # Calcular utilidad esperada con función lineal
    print("\n>> Cálculo con función de utilidad LINEAL")
    ue_lineal = utilidad_esperada(alternativas, tipo='lineal', verbose=True)
    print(f"Utilidad esperada (lineal): {ue_lineal:.2f}")
    
    # Calcular utilidad esperada con función exponencial (aversión al riesgo)
    print("\n>> Cálculo con función de utilidad EXPONENCIAL (parámetro 0.02)")
    ue_exp = utilidad_esperada(alternativas, tipo='exponencial', parametro=0.02, verbose=True)
    print(f"Utilidad esperada (exponencial, parámetro 0.02): {ue_exp:.4f}")
    
    # Interpretación de resultados
    print("\nInterpretación: bajo función lineal prefieres el valor promedio; bajo función exponencial se refleja aversión al riesgo.")

def modo_interactivo():
    print("\n--- MODO INTERACTIVO ---")
    # Solicitar al usuario el número de alternativas a definir
    n = int(input("¿Cuántas alternativas desea definir? ").strip())
    alternativas = []
    
    # Recopilar cada alternativa (probabilidad y resultado)
    for i in range(n):
        p = float(input(f"  Probabilidad de alternativa {i+1} (entre 0 y 1): ").strip())
        valor = float(input(f"  Resultado cuantitativo de alternativa {i+1}: ").strip())
        alternativas.append((p, valor))
    
    # Solicitar el tipo de función de utilidad
    tipo = input("Tipo de función de utilidad (lineal / logaritmica / exponencial): ").strip().lower()
    if tipo not in ['lineal','logaritmica','exponencial']:
        print("Tipo no reconocido, se usará 'lineal'.")
        tipo = 'lineal'
    
    # Si no es lineal, solicitar el parámetro de la función
    parametro = 1.0
    if tipo != 'lineal':
        parametro = float(input("Introduce el parámetro para la función de utilidad: ").strip())
    
    # Calcular la utilidad esperada con verbose activado
    print("\nIniciando cálculo detallado de utilidad esperada...")
    ue = utilidad_esperada(alternativas, tipo, parametro, verbose=True)
    
    # Mostrar resumen final
    print(f"\nAlternativas: {alternativas}")
    print(f"Utilidad esperada con función '{tipo}' (parámetro={parametro}): {ue:.4f}")

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
