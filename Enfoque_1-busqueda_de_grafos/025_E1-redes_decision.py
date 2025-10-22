"""
025-E1-redes_de_decisión.py
--------------------------------
Este script implementa una versión simplificada de una Red de Decisión (Decision Network / Influence Diagram) con visualización paso-a-paso:
- Define nodos de azar (chance), nodos de decisión (acciones) y nodo(s) de utilidad.
- Modo DEMO: escenario predefinido con cálculo visible.
- Modo INTERACTIVO: usuario elige entre acciones predefinidas (por ejemplo comprar/vender/esperar) y el script muestra todo el proceso de cálculo de utilidades esperadas.
- Variables y funciones en español.

Autor: Alejandro Aguirre Díaz
"""

import itertools

def funcion_utilidad(resultado, tipo='lineal', parametro=1.0):
    """
    Función de utilidad u(x) que transforma un resultado cuantitativo en utilidad subjetiva.
    :parametro resultado: valor cuantitativo del resultado
    :parametro tipo: tipo de función de utilidad ('lineal', 'logarítmica', 'exponencial')
    :parametro parametro: parámetro de la función
    :return: utilidad (float)
    """
    if tipo == 'lineal':
        return resultado
    elif tipo == 'logarítmica':
        import math
        return parametro * (math.log(resultado) if resultado > 0 else 0)
    elif tipo == 'exponencial':
        import math
        return 1 - math.exp(- parametro * resultado)
    else:
        return resultado

def utilidad_esperada(accion, probabilidades, resultados, tipo_utilidad='lineal', parametro=1.0):
    """
    Calcula la utilidad esperada de una acción específica mostrando los pasos.
    :parametro accion: nombre de la acción
    :parametro probabilidades: lista de probabilidades para los posibles resultados de la acción
    :parametro resultados: lista de valores cuantitativos correspondientes
    :parametro tipo_utilidad: tipo de función de utilidad
    :parametro parametro: parámetro para la función de utilidad
    :return: utilidad esperada (float)
    """
    suma = 0.0
    print(f"\nCálculo para la acción: {accion}")
    for p, valor in zip(probabilidades, resultados):
        u = funcion_utilidad(valor, tipo_utilidad, parametro)
        contrib = p * u
        print(f"  Resultado cuantitativo = {valor}, P = {p:.3f}, u(valor) = {u:.3f}, contribución = {contrib:.3f}")
        suma += contrib
    print(f"  → Utilidad esperada U(EU) = {suma:.3f}")
    return suma

def modo_demo():
    print("\n--- MODO DEMO ---")
    # Escenario demo: 3 acciones predefinidas de inversión/compras
    acciones = ['Comprar A', 'Comprar B', 'No comprar']
    probabilidades = {
        'Comprar A': [0.4, 0.6],
        'Comprar B': [0.2, 0.8],
        'No comprar':  [1.0]  # sin incertidumbre
    }
    resultados = {
        'Comprar A': [150, 20],     # 40% gana 150, 60% gana 20
        'Comprar B': [200, 5],      # 20% gana 200, 80% gana 5
        'No comprar':  [50]         # 100% obtiene 50 (por mantener)
    }
    print("Acciones disponibles:", acciones)
    print("Probabilidades y resultados definidos.")
    tipo = 'lineal'
    parametro = 1.0
    utilidades = {}
    actions = acciones
    for act in actions:
        utilidades[act] = utilidad_esperada(act, probabilidades[act], resultados[act], tipo, parametro)
    mejor = max(utilidades, key=lambda a: utilidades[a])
    print("\nResumen de utilidades esperadas:")
    for act, u in utilidades.items():
        print(f"  {act}: {u:.3f}")
    print(f"\n→ Mejor decisión: {mejor} con utilidad esperada {utilidades[mejor]:.3f}")

def modo_interactivo():
    print("\n--- MODO INTERACTIVO ---")
    # Presentar acciones predefinidas al usuario
    acciones_predefinidas = ['Comprar A', 'Comprar B', 'Comprar C', 'Esperar', 'Vender']
    print("Acciones predefinidas disponibles:")
    for i, act in enumerate(acciones_predefinidas, start=1):
        print(f"  {i}) {act}")
    seleccion = input("Introduce los números de las acciones que quieres evaluar (separados por comas): ").strip()
    seleccion_indices = [int(s.strip())-1 for s in seleccion.split(",") if s.strip().isdigit() and 1 <= int(s.strip()) <= len(acciones_predefinidas)]
    acciones = [acciones_predefinidas[i] for i in seleccion_indices]
    if not acciones:
        print("No se seleccionó ninguna acción válida. Se usará todas por defecto.")
        acciones = acciones_predefinidas.copy()

    probabilidades = {}
    resultados = {}
    for act in acciones:
        print(f"\nConfigurando la acción '{act}':")
        m = int(input("  ¿Cuántos posibles resultados tiene esta acción? ").strip())
        probs = []
        vals = []
        for j in range(m):
            p = float(input(f"    Probabilidad del resultado {j+1} (0-1): ").strip())
            v = float(input(f"    Resultado cuantitativo del resultado {j+1}: ").strip())
            probs.append(p)
            vals.append(v)
        suma_p = sum(probs)
        if abs(suma_p - 1.0) > 1e-6:
            print(f"    Advertencia: las probabilidades suman {suma_p:.3f} ≠ 1. Se normalizarán.")
            probs = [p/suma_p for p in probs]
        probabilidades[act] = probs
        resultados[act]     = vals

    tipo = input("Tipo de función de utilidad (lineal / logarítmica / exponencial): ").strip().lower()
    if tipo not in ['lineal','logarítmica','exponencial']:
        print("Tipo no reconocido. Se usará 'lineal'.")
        tipo = 'lineal'
    parametro = 1.0
    if tipo != 'lineal':
        parametro = float(input("Introduce el parámetro para la función de utilidad: ").strip())

    utilidades = {}
    print("\n=== CÁLCULO DE UTILIDADES ESPERADAS ===")
    for act in acciones:
        utilidades[act] = utilidad_esperada(act, probabilidades[act], resultados[act], tipo, parametro)

    print("\nResumen de utilidades esperadas:")
    for act, u in utilidades.items():
        print(f"  {act}: {u:.3f}")
    mejor = max(utilidades, key=lambda a: utilidades[a])
    print(f"\n→ Mejor decisión: {mejor} con utilidad esperada {utilidades[mejor]:.3f}")

def main():
    print("Seleccione modo de ejecución:")
    print("1) Modo DEMO")
    print("2) Modo INTERACTIVO\n")
    opcion = input("Ingrese el número de opción: ").strip()
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("Opción no válida. Se usará MODO DEMO.")
        modo_demo()

if __name__ == "__main__":
    main()
