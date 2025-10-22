"""
023-E1-acondicionamiento_corte.py
---------------------------------
Este script implementa Búsqueda Adversaria con Poda Alfa-Beta (conditioning & cutoff):
- Minimax con poda alfa-beta para recortar ramas que no afectan al resultado óptimo
- Permite especificar una profundidad de corte (cutoff) y usar una heurística cuando no se llega a un estado terminal
- Incluye dos modos:
    1. MODO DEMO: muestra, paso a paso, cómo la poda alfa-beta evita explorar ramas en Tres en Raya
    2. MODO INTERACTIVO: jugar contra la IA (minimax + alfa-beta) en Tres en Raya
- Traza el proceso: valores de alfa/beta, decisiones max/min y momentos de PODA

Autor: Alejandro Aguirre Díaz
"""

from typing import List, Optional, Tuple

# Representación del tablero: lista de 9 posiciones
# Índices: 0 1 2
#          3 4 5
#          6 7 8
# Valores: 'X', 'O' o ' '


def imprimir_tablero(tablero: List[str]) -> None:
    """
    Imprime el tablero de Tres en Raya en formato 3x3.
    """
    # Partimos el tablero (lista de 9) en 3 filas de 3 elementos
    filas = [tablero[i:i+3] for i in range(0, 9, 3)]
    print("\nTablero:")
    # Mostramos '.' en lugar de espacios para visualizar casillas vacías
    for fila in filas:
        print(" " + " | ".join(c if c != ' ' else '.' for c in fila))
    print()


def lineas_ganadoras() -> List[Tuple[int, int, int]]:
    """
    Retorna todas las combinaciones de índices que forman una línea ganadora.
    """
    # Tres filas, tres columnas y dos diagonales
    return [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # filas
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columnas
        (0, 4, 8), (2, 4, 6)              # diagonales
    ]


def hay_ganador(tablero: List[str], jugador: str) -> bool:
    """
    Verifica si 'jugador' ha ganado en el tablero actual.
    """
    # Recorremos todas las líneas ganadoras y comprobamos si están completas para 'jugador'
    for a, b, c in lineas_ganadoras():
        if tablero[a] == tablero[b] == tablero[c] == jugador:
            return True
    return False


def movimientos_validos(tablero: List[str]) -> List[int]:
    """
    Retorna la lista de índices vacíos donde se puede jugar.
    """
    # Un movimiento válido es cualquier casilla cuyo valor sea un espacio ' '
    return [i for i, v in enumerate(tablero) if v == ' ']


def estado_terminal(tablero: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Determina si el tablero está en un estado terminal y retorna el ganador ('X'/'O') o None si hay empate.
    """
    # Si 'X' ha completado una línea, el estado es terminal y el ganador es 'X'
    if hay_ganador(tablero, 'X'):
        return True, 'X'
    # Si 'O' ha completado una línea, el estado es terminal y el ganador es 'O'
    if hay_ganador(tablero, 'O'):
        return True, 'O'
    # Si no hay movimientos válidos restantes y nadie ganó, es un empate
    if not movimientos_validos(tablero):
        return True, None
    return False, None


def evaluar_terminal(tablero: List[str]) -> int:
    """
    Función de utilidad para estados terminales:
    - +1 si gana 'X'
    - -1 si gana 'O'
    -  0 si es empate
    """
    # Primero determinamos si el tablero es terminal y quién (si alguien) ganó
    terminal, ganador = estado_terminal(tablero)
    if not terminal:
        return 0
    # Asignamos utilidad según el ganador
    if ganador == 'X':
        return 1
    if ganador == 'O':
        return -1
    return 0


def heuristica(tablero: List[str]) -> int:
    """
    Heurística simple para posiciones no terminales (cuando aplicamos corte de profundidad):
    - Cuenta líneas "aún ganables" por X y por O y retorna su diferencia.
    - Una línea es "ganable" por X si no contiene 'O'; y por O si no contiene 'X'.
    """
    # Inicializamos los contadores de líneas potenciales para X y para O
    punt_x = 0
    punt_o = 0
    for a, b, c in lineas_ganadoras():
        # Obtenemos los tres símbolos de la línea
        linea = [tablero[a], tablero[b], tablero[c]]
        # Si la línea no tiene 'O', X aún podría completarla
        if 'O' not in linea:
            punt_x += 1
        # Si la línea no tiene 'X', O aún podría completarla
        if 'X' not in linea:
            punt_o += 1
    # Diferencia: más positivo favorece a X, más negativo favorece a O
    return punt_x - punt_o


def minimax_alfa_beta(
    tablero: List[str],
    jugador: str,
    profundidad: int,
    alfa: int,
    beta: int,
    profundidad_max: Optional[int] = None,
    usar_heuristica: bool = True,
    verbose: bool = True,
    indent: str = ""
) -> Tuple[int, Optional[int]]:
    """
    Minimax con poda alfa-beta.

    Parámetros:
    - tablero: estado actual (lista de 9 con 'X','O',' ')
    - jugador: 'X' (max) o 'O' (min)
    - profundidad: nivel actual en el árbol
    - alfa, beta: cotas para la poda
    - profundidad_max: si se especifica, detiene la búsqueda a esta profundidad (cutoff)
    - usar_heuristica: si True, evalúa con heurística al llegar al cutoff
    - verbose: si True, imprime el proceso paso a paso
    - indent: sangría para una traza más legible

    Retorna: (mejor_valor, mejor_movimiento)
    """
    # Comprobamos si el estado actual ya es terminal
    es_terminal, _ = estado_terminal(tablero)

    # Caso 1: estado terminal → evaluar y no hay movimientos
    if es_terminal:
        # Utilidad exacta del resultado final (victoria/derrota/empate)
        valor = evaluar_terminal(tablero)
        if verbose:
            print(f"{indent}[T] Terminal -> utilidad = {valor}")
        return valor, None

    # Caso 2: profundidad de corte alcanzada → usar heurística si se permite
    if profundidad_max is not None and profundidad >= profundidad_max:
        # Al alcanzar el cutoff aplicamos una evaluación aproximada (heurística)
        valor = heuristica(tablero) if usar_heuristica else 0
        if verbose:
            print(f"{indent}[CUTOFF] Prof={profundidad} -> heurística = {valor}")
        return valor, None

    # Determinar si maximizamos (X) o minimizamos (O)
    es_max = (jugador == 'X')
    # Inicializamos el mejor valor fuera del rango de utilidades posibles [-1, 1]
    mejor_valor = -10 if es_max else 10  # valores fuera de rango de utilidad [-1,1]
    mejor_mov: Optional[int] = None

    if verbose:
        tipo = 'MAX' if es_max else 'MIN'
        print(f"{indent}{tipo} Jugador='{jugador}' prof={profundidad} alfa={alfa} beta={beta}")

    # Exploramos todos los movimientos legales desde el estado actual
    for mov in movimientos_validos(tablero):
        # Aplicar movimiento: colocamos la ficha del jugador actual
        tablero[mov] = jugador

        # Cambiar turno: alternamos entre 'X' y 'O'
        siguiente = 'O' if jugador == 'X' else 'X'

        # Explorar hijo: evaluamos recursivamente el estado resultante
        valor_hijo, _ = minimax_alfa_beta(
            tablero, siguiente, profundidad + 1, alfa, beta,
            profundidad_max=profundidad_max, usar_heuristica=usar_heuristica,
            verbose=verbose, indent=indent + "  "
        )

        # Deshacer movimiento para restaurar el estado antes de probar el siguiente
        tablero[mov] = ' '

        if es_max:
            # Maximiza X
            # Actualizamos el mejor valor y movimiento si el hijo es mejor
            if valor_hijo > mejor_valor:
                mejor_valor, mejor_mov = valor_hijo, mov
            # Actualizamos alfa con el mejor valor encontrado para MAX
            alfa = max(alfa, mejor_valor)
            if verbose:
                print(f"{indent}  -> mov {mov} val={valor_hijo} | mejor={mejor_valor} alfa={alfa} beta={beta}")
            # Poda beta
            # Si el mejor que puede MIN (beta) es <= que lo que ya asegura MAX (alfa), podar
            if beta <= alfa:
                if verbose:
                    print(f"{indent}  [PODA] beta <= alfa ({beta} <= {alfa})")
                break
        else:
            # Minimiza O
            # Actualizamos el mejor valor y movimiento si el hijo es peor (más bajo)
            if valor_hijo < mejor_valor:
                mejor_valor, mejor_mov = valor_hijo, mov
            # Actualizamos beta con el mejor valor encontrado para MIN
            beta = min(beta, mejor_valor)
            if verbose:
                print(f"{indent}  -> mov {mov} val={valor_hijo} | mejor={mejor_valor} alfa={alfa} beta={beta}")
            # Poda alfa
            # Si el mejor que puede MIN (beta) es <= que lo que asegura MAX (alfa), podar
            if beta <= alfa:
                if verbose:
                    print(f"{indent}  [PODA] beta <= alfa ({beta} <= {alfa})")
                break

    return mejor_valor, mejor_mov


def mejor_movimiento(tablero: List[str], jugador: str, profundidad_max: Optional[int] = None, verbose: bool = True) -> int:
    """
    Calcula el mejor movimiento para 'jugador' usando minimax + poda alfa-beta.
    """
    # Llamamos a minimax_alfa_beta partiendo de la raíz con alfa/beta iniciales
    valor, mov = minimax_alfa_beta(
        tablero=tablero,
        jugador=jugador,
        profundidad=0,
        alfa=-10,
        beta=10,
        profundidad_max=profundidad_max,
        usar_heuristica=True,
        verbose=verbose,
        indent=""
    )
    # Por seguridad, si no se encontró (no debería ocurrir), elegir el primero libre
    return mov if mov is not None else movimientos_validos(tablero)[0]


def modo_demo() -> None:
    """
    Modo DEMO: Usa una posición intermedia de Tres en Raya y muestra el proceso
    de decisión con poda alfa-beta, incluyendo podas y valores.
    """
    print("\n" + "=" * 70)
    print("--- MODO DEMO: Minimax con Poda Alfa-Beta en Tres en Raya ---")
    print("=" * 70)

    # Posición de ejemplo (X a mover):
    # X O X
    # . O .
    # . . .
    # Usamos espacios ' ' para casillas vacías
    tablero = ['X', 'O', 'X',
               ' ', 'O', ' ',
               ' ', ' ', ' ']

    imprimir_tablero(tablero)
    print("Buscando mejor movimiento para 'X' con poda alfa-beta (sin cutoff)...\n")
    # Obtenemos el mejor movimiento para X sin profundidad de corte
    mejor = mejor_movimiento(tablero, 'X', profundidad_max=None, verbose=True)
    print(f"\n[RESULTADO DEMO] Mejor movimiento para 'X': {mejor} (índice 0-8)")

    # Aplicar y mostrar el tablero resultante
    # Colocamos la 'X' en la casilla sugerida y mostramos el tablero
    tablero[mejor] = 'X'
    imprimir_tablero(tablero)


def pedir_movimiento_usuario(tablero: List[str]) -> int:
    """
    Solicita al usuario un movimiento válido (1-9) y lo convierte a índice (0-8).
    """
    # Calculamos los índices disponibles en el tablero actual
    # y validamos la entrada del usuario hasta que sea correcta
    validos = movimientos_validos(tablero)
    while True:
        s = input("Tu movimiento (1-9, filas por filas): ").strip()
        if not s.isdigit():
            print("Introduce un número del 1 al 9.")
            continue
        pos = int(s) - 1
        if pos in validos:
            return pos
        print("Movimiento inválido. Elige una casilla libre.")


def modo_interactivo() -> None:
    """
    Modo INTERACTIVO: Juega contra la IA (minimax + alfa-beta) en Tres en Raya.
    Puedes escoger si juegas con 'X' (sales primero) o con 'O'.
    """
    print("\n" + "=" * 70)
    print("--- MODO INTERACTIVO: Juega contra la IA (Alfa-Beta) ---")
    print("=" * 70)

    tablero = [' '] * 9
    # Elegimos el bando del usuario (X comienza primero)
    jugador_usuario = input("¿Quieres ser 'X' (primero) o 'O' (segundo)? [X/O]: ").strip().upper() or 'X'
    if jugador_usuario not in ('X', 'O'):
        jugador_usuario = 'X'
        print("Entrada inválida. Serás 'X'.")

    jugador_ia = 'O' if jugador_usuario == 'X' else 'X'
    turno = 'X'  # siempre empieza X

    # Opcional: permitir un cutoff de profundidad
    try:
        corte = input("¿Profundidad de corte (vacío=sin cutoff)? ").strip()
        profundidad_max = int(corte) if corte else None
    except ValueError:
        profundidad_max = None
        print("Entrada inválida, sin cutoff.")

    # Bucle principal de juego: alterna entre usuario e IA hasta estado terminal
    while True:
        imprimir_tablero(tablero)
        es_term, ganador = estado_terminal(tablero)
        if es_term:
            # Si no hay ganador, es empate
            if ganador is None:
                print("[FIN] Empate.")
            else:
                print(f"[FIN] Gana '{ganador}'.")
            break

        if turno == jugador_usuario:
            # Turno del usuario: pedimos movimiento válido
            mov = pedir_movimiento_usuario(tablero)
            tablero[mov] = jugador_usuario
        else:
            # Turno de la IA: usamos minimax con poda alfa-beta
            print("La IA está pensando con poda alfa-beta...")
            mov = mejor_movimiento(tablero, jugador_ia, profundidad_max=profundidad_max, verbose=False)
            tablero[mov] = jugador_ia
            print(f"IA juega en posición {mov} (índice 0-8).")

        # Alternar turno
        turno = 'O' if turno == 'X' else 'X'


def main() -> None:
    """
    Función principal: muestra el menú y ejecuta el modo elegido.
    """
    print("\n" + "=" * 70)
    print("BÚSQUEDA ADVERSARIA: Minimax con Poda Alfa-Beta (conditioning & cutoff)")
    print("=" * 70)
    print("\nSeleccione modo de ejecución:")
    print("1) Modo DEMO (traza con podas en una posición fija)")
    print("2) Modo INTERACTIVO (jugar contra la IA)\n")

    # Leemos la opción del usuario y ejecutamos el modo correspondiente
    opcion = input("Ingrese el número de opción: ").strip()
    if opcion == '1':
        modo_demo()
    elif opcion == '2':
        modo_interactivo()
    else:
        print("\nOpción no válida. Ejecutando modo DEMO por defecto.\n")
        modo_demo()


if __name__ == '__main__':
    main()

