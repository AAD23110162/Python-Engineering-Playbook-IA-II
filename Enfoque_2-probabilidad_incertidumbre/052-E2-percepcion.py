"""
052-E2-percepcion.py
---------------------------------
Este script recorre un flujo didáctico de visión por computador:

- Gráficos por computadora: Creación y visualización de imágenes y formas.
- Preprocesado: filtros (gaussiano, mediana, bilateral), ecualización.
- Detección de aristas y segmentación (Sobel/Canny, Otsu, K-means).
- Texturas y sombras (LBP/GLCM; normalización de iluminación).
- Reconocimiento de objetos (template matching o HOG + KNN/SVM ligero).
- Reconocimiento de escritura (digits de sklearn con KNN/SVM).
- Etiquetado de líneas (Transformada de Hough: IDs, ángulo, longitud).
- Movimiento (diferencia de frames, flujo óptico de Farnebäck).

Modos:
1) DEMO: ejecuta un pipeline predefinido con datos sintéticos/embebidos.
2) INTERACTIVO: permite escoger módulo, cargar imagen y ajustar parámetros.

Autor: Alejandro Aguirre Díaz
"""

import os  # Utilidades del sistema (ruta de archivos si se requiere)
import math  # Operaciones matemáticas (ángulos, trigonometría, etc.)
import numpy as np  # Cálculo numérico y matrices
import matplotlib.pyplot as plt  # Visualización

# Librerías opcionales; si no están instaladas, el script avisará en tiempo de ejecución
try:
    import cv2  # type: ignore  # OpenCV para aristas, Hough, flujo óptico, etc.
except Exception:
    # Si OpenCV no está disponible, algunas funciones levantarán un error controlado
    cv2 = None

try:
    # Módulos de scikit-image para filtros, características y transformaciones
    from skimage import filters, feature, exposure, color, morphology, measure  # type: ignore
    from skimage.feature import local_binary_pattern  # type: ignore
    from skimage.filters import threshold_otsu  # type: ignore
except Exception:
    # Fallback cuando scikit-image no está disponible
    filters = feature = exposure = color = morphology = measure = None
    local_binary_pattern = threshold_otsu = None

try:
    # scikit-learn para el ejemplo de reconocimiento de escritura
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
except Exception:
    # Si no está disponible, se avisará al ejecutar el módulo correspondiente
    load_digits = train_test_split = accuracy_score = KNeighborsClassifier = None


# -------------------------
# Utilidades de visualización
# -------------------------
def mostrar_lado_a_lado(imgs, titulos=None, cmap='gray', filas=1):
    """
    Muestra una lista de imágenes lado a lado para comparación rápida.
    """
    n = len(imgs)  # Número total de imágenes
    cols = int(math.ceil(n / filas))  # Columnas según cantidad y filas solicitadas
    plt.figure(figsize=(4*cols, 3*filas))  # Tamaño de figura proporcional
    for i, im in enumerate(imgs):
        ax = plt.subplot(filas, cols, i+1)
        ax.imshow(im, cmap=cmap)  # Mostrar imagen con el colormap indicado
        ax.axis('off')  # Quitar ejes para claridad
        if titulos and i < len(titulos):
            ax.set_title(titulos[i])  # Título por subgráfico si se proporciona
    plt.tight_layout()  # Ajustar espaciamiento automáticamente
    plt.show()  # Mostrar en pantalla


# -------------------------
# 1) Preprocesado: Filtros
# -------------------------
def preprocesado_filtros(img, sigma=1.2, k_mediana=3, d_bilateral=9, sigma_color=75, sigma_espacio=75):
    """
    Aplica filtros típicos de preprocesado: Gaussiano, Mediana y Bilateral.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no disponible: instala opencv-python-headless")

    # Si la imagen es color, trabajamos en escala de grises para filtros base
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a niveles de gris
    else:
        img_gray = img.copy()  # Ya es 1 canal

    # Suavizado Gaussiano: reduce ruido, preserva menos bordes que mediana
    gauss = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma)
    # Filtro de Mediana: elimina ruido sal y pimienta
    med = cv2.medianBlur(img_gray, k_mediana)
    # Filtro Bilateral: suaviza preservando bordes (depende de intensidad y distancia)
    bil = cv2.bilateralFilter(img_gray, d_bilateral, sigma_color, sigma_espacio)
    return img_gray, gauss, med, bil


# ---------------------------------------
# 2) Aristas (Sobel/Canny) y Segmentación
# ---------------------------------------
def aristas_y_segmentacion(img_gray, canny1=100, canny2=200, kmeans_k=2):
    """
    Detecta aristas con Canny y segmenta con K-means (en intensidad).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no disponible")

    # Canny: bordes binarios
    # canny1/canny2: umbrales inferior y superior para histéresis
    edges = cv2.Canny(img_gray, canny1, canny2)

    # Segmentación K-means en intensidades (1 canal)
    datos = img_gray.reshape(-1, 1).astype(np.float32)  # Vectorizar imagen
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)  # Criterio de parada
    ret, labels, centers = cv2.kmeans(datos, kmeans_k, None, criterios, 5, cv2.KMEANS_PP_CENTERS)
    seg = labels.reshape(img_gray.shape)
    # Normalizar a [0,255] para visualización
    seg_norm = (255 * (seg - seg.min()) / max(1, (seg.max() - seg.min()))).astype(np.uint8)
    return edges, seg_norm


# ----------------------------
# 3) Texturas y 4) Sombras
# ----------------------------
def texturas_y_sombras(img_gray, metodo_lbp='uniform', radius=1, n_points=8):
    """
    Extrae LBP para texturas y aplica una corrección simple de iluminación (sombras).
    """
    if local_binary_pattern is None:
        raise RuntimeError("scikit-image no disponible para LBP")

    # LBP: descriptor local de textura
    # n_points: número de vecinos; radius: radio en píxeles
    lbp = local_binary_pattern(img_gray, n_points, radius, method=metodo_lbp)

    # Corrección de iluminación (homomorphic-like): log -> blur -> resta
    img_f = img_gray.astype(np.float32) + 1.0  # Evitar log(0)
    log_img = np.log(img_f)
    if cv2 is None:
        # Fallback: blur con convolución gaussiana de skimage si cv2 no está (opcional)
        if filters is None:
            corr = img_gray  # No se corrige si no hay filtros disponibles
        else:
            blur = filters.gaussian(log_img, sigma=2.0)  # Suavizado en dominio log
            corr = np.clip(np.exp(log_img - blur), 0, 255).astype(np.uint8)  # Volver a dominio intensidad
    else:
        blur = cv2.GaussianBlur(log_img, (0, 0), 2.0)
        corr = np.clip(np.exp(log_img - blur), 0, 255).astype(np.uint8)

    return lbp, corr


# ---------------------------
# 5) Reconocimiento de objetos
# ---------------------------
def reconocimiento_objetos_template(img_gray, plantilla):
    """
    Reconocimiento de objetos vía template matching (demo sin entrenamiento).
    Retorna mapa de similitud y la mejor coincidencia.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV no disponible")

    res = cv2.matchTemplate(img_gray, plantilla, cv2.TM_CCOEFF_NORMED)  # Mapa de similitud normalizado
    minv, maxv, minl, maxl = cv2.minMaxLoc(res)  # Obtener mejor coincidencia
    top_left = maxl  # Coordenadas esquina superior izquierda
    h, w = plantilla.shape[:2]  # Tamaño de la plantilla
    # Dibujar rectángulo sobre coincidencia
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_rgb, top_left, (top_left[0]+w, top_left[1]+h), (0, 0, 255), 2)
    return res, img_rgb, top_left, (w, h), maxv


# --------------------------------
# 6) Reconocimiento de escritura
# --------------------------------
def reconocimiento_escritura_knn(test_size=0.3, n_neighbors=3):
    """
    Entrena un KNN sobre el dataset 'digits' de sklearn (8x8 píxeles).
    Retorna accuracy y ejemplos de predicción.
    """
    if load_digits is None:
        raise RuntimeError("scikit-learn no disponible")

    digits = load_digits()  # Cargar dataset de dígitos (8x8)
    X = digits.data  # Vectorización de imágenes
    y = digits.target  # Etiquetas 0-9
    # Separar en entrenamiento/prueba estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)  # Clasificador KNN simple
    clf.fit(X_train, y_train)  # Entrenamiento
    y_pred = clf.predict(X_test)  # Predicción
    acc = accuracy_score(y_test, y_pred)  # Precisión
    return acc, (X_test[:8], y_test[:8], y_pred[:8])


# --------------------------
# 7) Etiquetado de líneas
# --------------------------
def etiquetar_lineas_hough(img_gray, th_canny1=50, th_canny2=150, th_hough=80, min_long=30, max_gap=10):
    """
    Detecta y etiqueta líneas con Canny + Hough probabilística.
    Devuelve imagen anotada y listado de líneas con ID, longitud y ángulo.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV no disponible")

    edges = cv2.Canny(img_gray, th_canny1, th_canny2)  # Binarización de bordes
    # Hough probabilística: detecta segmentos de líneas con parámetros configurables
    lineas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=th_hough, minLineLength=min_long, maxLineGap=max_gap)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    resultados = []
    if lineas is not None:
        for idx, l in enumerate(lineas[:, 0, :], start=1):
            x1, y1, x2, y2 = l
            # Calcular longitud y ángulo (en grados)
            longitud = np.hypot(x2 - x1, y2 - y1)
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            resultados.append({"id": idx, "p1": (x1, y1), "p2": (x2, y2), "longitud": longitud, "angulo": ang})
            cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"#{idx}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 180, 20), 1, cv2.LINE_AA)
    return img_rgb, resultados, edges


# --------------------------
# 8) Movimiento (óptico/DFG)
# --------------------------
def movimiento_sintetico_y_optico(n_frames=10, tam=128, velocidad=(2, 1)):
    """
    Genera una secuencia sintética (un cuadrado moviéndose) y calcula:
    - Diferencia entre frames consecutivos (detección de movimiento)
    - Flujo óptico de Farnebäck (magnitud media)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV no disponible")

    # Generar frames sintéticos en escala de grises
    frames = []  # Lista de frames
    pos = np.array([20, 20], dtype=np.int32)  # Posición inicial del cuadrado
    for t in range(n_frames):
        im = np.zeros((tam, tam), dtype=np.uint8)  # Fondo negro
        x, y = pos
        cv2.rectangle(im, (x, y), (x+30, y+30), 200, -1)  # Dibujar cuadrado
        frames.append(im)  # Agregar frame a la secuencia
        pos += np.array(velocidad, dtype=np.int32)  # Actualizar posición

    # Diferencia simple entre frames
    diffs = [cv2.absdiff(frames[i+1], frames[i]) for i in range(n_frames-1)]  # Magnitud de cambio

    # Flujo óptico denso (Farnebäck): entre frame t y t+1
    flujos = []  # Campos de flujo por par de frames
    mags = []  # Magnitudes medias del flujo
    for i in range(n_frames-1):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flujos.append(flow)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(mag.mean())  # Magnitud media por par de frames
    mag_media = float(np.mean(mags)) if mags else 0.0
    return frames, diffs, flujos, mag_media


# --------------------------
# Utilidad: Generar imagen sintética
# --------------------------
def generar_imagen_sintetica():
    """
    Genera una imagen sintética de 256x256 con formas básicas.
    Usa OpenCV si está disponible, o numpy si no.
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    
    if cv2 is not None:
        # Usar OpenCV para dibujar formas
        cv2.circle(img, (80, 80), 40, 180, -1)
        cv2.rectangle(img, (150, 50), (220, 120), 120, -1)
        cv2.line(img, (30, 200), (230, 200), 255, 3)
    else:
        # Fallback con numpy puro: círculo aproximado, rectángulo y línea
        # Círculo (aproximado con disco)
        y, x = np.ogrid[:256, :256]
        mask_circle = (x - 80)**2 + (y - 80)**2 <= 40**2
        img[mask_circle] = 180
        
        # Rectángulo
        img[50:120, 150:220] = 120
        
        # Línea horizontal (grosor 3 píxeles)
        img[199:202, 30:230] = 255
    
    return img


# --------------------------
# DEMO
# --------------------------
def modo_demo():
    print("\n--- MODO DEMO: Flujo didáctico de visión por computador ---")

    # Crear imagen sintética con formas y ruido
    img = generar_imagen_sintetica()

    # 1) Preprocesado
    if cv2 is None:
        print("\n[1. Preprocesado] SALTADO: OpenCV no disponible.")
        gray = img.copy()
    else:
        # Convertimos a BGR solo para reutilizar la función que espera 3 canales opcionalmente
        gray, g, m, b = preprocesado_filtros(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        mostrar_lado_a_lado([gray, g, m, b], ["Gray", "Gauss", "Mediana", "Bilateral"])

    # 2) Aristas y segmentación
    if cv2 is None:
        print("[2. Aristas/Segmentación] SALTADO: OpenCV no disponible.")
    else:
        edges, seg = aristas_y_segmentacion(gray)
        mostrar_lado_a_lado([gray, edges, seg], ["Entrada", "Canny", "Segmentado"])

    # 3–4) Texturas y sombras
    if local_binary_pattern is None:
        print("[3. Texturas/Sombras] SALTADO: scikit-image no disponible.")
    else:
        lbp, corr = texturas_y_sombras(gray)
        mostrar_lado_a_lado([gray, lbp, corr], ["Entrada", "LBP", "Corrección iluminación"], cmap=None)

    # 5) Reconocimiento de objetos (template del círculo recortado)
    if cv2 is None:
        print("[5. Objetos] SALTADO: OpenCV no disponible.")
    else:
        plantilla = img[40:120, 40:120]
        scoremap, match_vis, tl, wh, score = reconocimiento_objetos_template(gray, plantilla)
        mostrar_lado_a_lado([gray, match_vis], ["Entrada", f"Template match (score={score:.2f})"])

    # 6) Reconocimiento de escritura (digits)
    if load_digits is None:
        print("[6. Escritura] SALTADO: scikit-learn no disponible.")
    else:
        acc, (X8, y8, pred8) = reconocimiento_escritura_knn()
        print(f"[Escritura] Accuracy KNN (digits): {acc:.3f}")
        # Visualizar ejemplos (reconstruyendo 8x8)
        muestras = [X8[i].reshape(8, 8) for i in range(len(y8))]
        tit = [f"y={y8[i]} pred={pred8[i]}" for i in range(len(y8))]
        mostrar_lado_a_lado(muestras, titulos=tit, filas=2)

    # 7) Etiquetado de líneas (usar la misma imagen sintética)
    if cv2 is None:
        print("[7. Líneas] SALTADO: OpenCV no disponible.")
    else:
        img_lines, info_lineas, edges2 = etiquetar_lineas_hough(gray)
        mostrar_lado_a_lado([gray, edges2, cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)],
                            ["Entrada", "Bordes", "Líneas etiquetadas"])
        print(f"[Líneas] Detectadas: {len(info_lineas)}")
        if info_lineas:
            print("  Ejemplo:", info_lineas[0])

    # 8) Movimiento
    if cv2 is None:
        print("[8. Movimiento] SALTADO: OpenCV no disponible.")
    else:
        frames, diffs, flujos, mag_media = movimiento_sintetico_y_optico()
        mostrar_lado_a_lado([frames[0], frames[1], diffs[0]], ["t=0", "t=1", "diff"])
        print(f"[Movimiento] Magnitud media de flujo óptico: {mag_media:.4f}")


# --------------------------
# INTERACTIVO
# --------------------------
def modo_interactivo():
    print("\n--- MODO INTERACTIVO ---")
    print("Seleccione módulo:")
    print("1) Preprocesado: Filtros")
    print("2) Aristas y Segmentación")
    print("3) Texturas y Sombras")
    print("4) Reconocimiento de Objetos (template)")
    print("5) Reconocimiento de Escritura (digits)")
    print("6) Etiquetado de Líneas (Hough)")
    print("7) Movimiento (flujo óptico)")
    op = input("Opción [1-7]: ").strip()

    # Para simplicidad, usamos imagen sintética en módulos 1-2-3-6
    img = generar_imagen_sintetica()
    gray = img.copy()

    if op == '1':
        # Demostración de filtros básicos de preprocesado
        _, g, m, b = preprocesado_filtros(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        mostrar_lado_a_lado([gray, g, m, b], ["Gray", "Gauss", "Mediana", "Bilateral"])
    elif op == '2':
        # Detección de bordes y segmentación con K-means
        edges, seg = aristas_y_segmentacion(gray)
        mostrar_lado_a_lado([gray, edges, seg], ["Entrada", "Canny", "Segmentado"])
    elif op == '3':
        # LBP para textura y corrección de iluminación como ejemplo de sombras
        lbp, corr = texturas_y_sombras(gray)
        mostrar_lado_a_lado([gray, lbp, corr], ["Entrada", "LBP", "Corrección iluminación"], cmap=None)
    elif op == '4':
        # Matching de plantilla para localizar un patrón conocido
        plantilla = img[40:120, 40:120]
        _, vis, tl, wh, score = reconocimiento_objetos_template(gray, plantilla)
        mostrar_lado_a_lado([gray, vis], ["Entrada", f"Template match (score={score:.2f})"])
    elif op == '5':
        # Clasificación de dígitos (escritura) con KNN
        acc, (X8, y8, pred8) = reconocimiento_escritura_knn()
        print(f"[Escritura] Accuracy KNN (digits): {acc:.3f}")
        muestras = [X8[i].reshape(8, 8) for i in range(len(y8))]
        tit = [f"y={y8[i]} pred={pred8[i]}" for i in range(len(y8))]
        mostrar_lado_a_lado(muestras, titulos=tit, filas=2)
    elif op == '6':
        # Detección de líneas y etiquetado (ángulo/longitud)
        img_lines, info_lineas, edges2 = etiquetar_lineas_hough(gray)
        mostrar_lado_a_lado([gray, edges2, cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)],
                            ["Entrada", "Bordes", "Líneas etiquetadas"])
        print(f"[Líneas] Detectadas: {len(info_lineas)}")
    elif op == '7':
        # Detección de movimiento y flujo óptico Farnebäck
        frames, diffs, flujos, mag_media = movimiento_sintetico_y_optico()
        mostrar_lado_a_lado([frames[0], frames[1], diffs[0]], ["t=0", "t=1", "diff"])
        print(f"[Movimiento] Magnitud media de flujo óptico: {mag_media:.4f}")
    else:
        print("Opción no válida. Ejecutando DEMO.")
        modo_demo()


# --------------------------
# MAIN
# --------------------------
def main():
    print("\n052-E2-gráficos_por_computador.py")
    print("Seleccione modo:")
    print("1) DEMO")
    print("2) INTERACTIVO")
    op = input("Opción [1/2]: ").strip()
    if op == '1':
        modo_demo()  # Ejecuta la secuencia completa con datos sintéticos
    elif op == '2':
        modo_interactivo()  # Permite elegir módulo y parámetros
    else:
        print("Opción no válida. DEMO por defecto.")
        modo_demo()


if __name__ == "__main__":
    if cv2 is None:
        # Mensaje informativo si falta OpenCV. Algunas funciones no podrán ejecutarse.
        print("[AVISO] OpenCV no está disponible. Instala 'opencv-python-headless' para funciones completas.")
    main()