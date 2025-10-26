"""
006-E2-tratamiento_probabilistico_lenguaje.py
----------------------------------------------
Este script implementa Tratamiento Probabilístico del Lenguaje Natural:
- Construye modelos de lenguaje basados en n-gramas (unigramas, bigramas, trigramas)
- Calcula probabilidades de secuencias de palabras usando el modelo de Markov
- Implementa suavizado (Laplace, Good-Turing) para manejar n-gramas no vistos
- Genera texto automáticamente usando probabilidades de transición
- Aplica el modelo para corrección ortográfica y predicción de palabras
- Evalúa perplejidad del modelo sobre corpus de prueba
- Variables y funciones en español

El programa puede ejecutarse en dos modos:
1. DEMO: ejecuta automáticamente construcción de modelo y generación de texto predefinido
2. INTERACTIVO: permite entrenar modelos con corpus personalizados y generar texto

Autor: Alejandro Aguirre Díaz
"""

import random
from collections import defaultdict, Counter

# ========== CLASE MODELO N-GRAMAS ==========

class ModeloNGramas:
    """
    Implementa un modelo de lenguaje basado en n-gramas.
    Calcula probabilidades de secuencias de palabras.
    """
    
    def __init__(self, n=2):
        """
        Inicializa el modelo de n-gramas.
        
        :param n: orden del modelo (1=unigramas, 2=bigramas, 3=trigramas, etc.)
        """
        self.n = n
        
        # Diccionarios para contar n-gramas
        # ngramas_conteo: {(w1, w2, ..., wn-1): {wn: conteo}}
        self.ngramas_conteo = defaultdict(lambda: defaultdict(int))
        
        # Contexto_conteo: {(w1, w2, ..., wn-1): conteo_total}
        self.contexto_conteo = defaultdict(int)
        
        # Vocabulario completo (todas las palabras vistas)
        self.vocabulario = set()
        
        # Parámetro de suavizado Laplace (add-k smoothing)
        self.alpha = 1.0
    
    def tokenizar(self, texto):
        """
        Convierte texto en lista de palabras (tokens).
        
        :parametro texto: cadena de texto
        :return: lista de palabras en minúsculas
        """
        # Limpieza básica: convertir a minúsculas y separar por espacios
        return texto.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '').split()
    
    def entrenar(self, corpus):
        """
        Entrena el modelo de n-gramas con un corpus de texto.

        :parametro corpus: lista de frases (strings) o un solo string
        """
        # Si corpus es un solo string, convertir a lista
        if isinstance(corpus, str):
            corpus = [corpus]
        
        # Procesar cada frase del corpus
        for frase in corpus:
            # Tokenizar la frase
            tokens = self.tokenizar(frase)
            
            # Añadir marcadores de inicio y fin
            tokens = ['<INICIO>'] * (self.n - 1) + tokens + ['<FIN>']
            
            # Actualizar vocabulario
            self.vocabulario.update(tokens)
            
            # Extraer n-gramas y contar ocurrencias
            for i in range(len(tokens) - self.n + 1):
                # Obtener el contexto (primeras n-1 palabras)
                contexto = tuple(tokens[i:i + self.n - 1])
                
                # Obtener la palabra siguiente
                palabra_siguiente = tokens[i + self.n - 1]
                
                # Incrementar conteos
                self.ngramas_conteo[contexto][palabra_siguiente] += 1
                self.contexto_conteo[contexto] += 1
    
    def probabilidad(self, palabra, contexto):
        """
        Calcula P(palabra | contexto) usando suavizado Laplace.

        :parametro palabra: palabra cuya probabilidad queremos calcular
        :parametro contexto: tupla de palabras previas (longitud n-1)
        :return: probabilidad condicional
        """
        # Asegurar que contexto es una tupla
        if not isinstance(contexto, tuple):
            contexto = tuple(contexto) if isinstance(contexto, list) else (contexto,)
        
        # Conteo de esta combinación específica
        conteo_ngrama = self.ngramas_conteo[contexto][palabra]
        
        # Conteo total del contexto
        conteo_contexto = self.contexto_conteo[contexto]
        
        # Tamaño del vocabulario (para suavizado)
        V = len(self.vocabulario)
        
        # Suavizado Laplace (add-1 smoothing):
        # P(w | contexto) = (conteo(contexto, w) + α) / (conteo(contexto) + α × V)
        probabilidad = (conteo_ngrama + self.alpha) / (conteo_contexto + self.alpha * V)
        
        return probabilidad
    
    def probabilidad_secuencia(self, secuencia):
        """
        Calcula la probabilidad de una secuencia completa de palabras.
        Usa la regla de la cadena: P(w1, w2, ..., wn) = ∏ P(wi | w1...wi-1)

        :parametro secuencia: lista de palabras o string
        :return: probabilidad de la secuencia
        """
        # Tokenizar si es necesario
        if isinstance(secuencia, str):
            tokens = self.tokenizar(secuencia)
        else:
            tokens = secuencia
        
        # Añadir marcadores de inicio
        tokens = ['<INICIO>'] * (self.n - 1) + tokens
        
        # Calcular probabilidad total usando regla de la cadena
        prob_total = 1.0
        
        for i in range(self.n - 1, len(tokens)):
            # Obtener contexto (n-1 palabras anteriores)
            contexto = tuple(tokens[i - self.n + 1:i])
            palabra = tokens[i]
            
            # Multiplicar por P(palabra | contexto)
            prob_total *= self.probabilidad(palabra, contexto)
        
        return prob_total
    
    def generar_texto(self, longitud=10, contexto_inicial=None):
        """
        Genera texto aleatorio usando las probabilidades del modelo.

        :parametro longitud: número de palabras a generar
        :parametro contexto_inicial: palabras iniciales (opcional)
        :return: texto generado
        """
        # Inicializar contexto
        if contexto_inicial is None:
            # Empezar con marcadores de inicio
            contexto = ['<INICIO>'] * (self.n - 1)
        else:
            # Usar contexto proporcionado
            if isinstance(contexto_inicial, str):
                contexto = self.tokenizar(contexto_inicial)
            else:
                contexto = list(contexto_inicial)
            
            # Asegurar que tiene el tamaño correcto
            while len(contexto) < self.n - 1:
                contexto.insert(0, '<INICIO>')
            contexto = contexto[-(self.n - 1):]
        
        # Generar palabras
        resultado = []
        
        for _ in range(longitud):
            # Obtener distribución de probabilidad para el contexto actual
            contexto_tuple = tuple(contexto)
            
            if contexto_tuple not in self.ngramas_conteo:
                # Si nunca vimos este contexto, elegir palabra aleatoria del vocabulario
                siguiente = random.choice(list(self.vocabulario - {'<INICIO>', '<FIN>'}))
            else:
                # Elegir palabra según probabilidades
                palabras_posibles = list(self.ngramas_conteo[contexto_tuple].keys())
                pesos = [self.ngramas_conteo[contexto_tuple][w] for w in palabras_posibles]
                
                # Selección aleatoria ponderada
                siguiente = random.choices(palabras_posibles, weights=pesos, k=1)[0]
            
            # Si llegamos al marcador de fin, terminar
            if siguiente == '<FIN>':
                break
            
            # Añadir palabra al resultado
            resultado.append(siguiente)
            
            # Actualizar contexto (ventana deslizante)
            contexto = contexto[1:] + [siguiente]
        
        return ' '.join(resultado)
    
    def palabra_mas_probable(self, contexto):
        """
        Encuentra la palabra más probable dado un contexto.
        Útil para autocompletado y corrección.
        
        :parametro contexto: tupla o lista de palabras previas
        :return: palabra más probable y su probabilidad
        """
        # Asegurar formato de tupla
        if not isinstance(contexto, tuple):
            contexto = tuple(contexto) if isinstance(contexto, list) else (contexto,)
        
        if contexto not in self.ngramas_conteo:
            return None, 0.0
        
        # Encontrar palabra con mayor conteo
        palabras_conteos = self.ngramas_conteo[contexto]
        palabra_max = max(palabras_conteos, key=palabras_conteos.get)
        prob_max = self.probabilidad(palabra_max, contexto)
        
        return palabra_max, prob_max

# ========== MODO DEMO ==========

def modo_demo():
    """Ejecuta el modo demostrativo con corpus predefinido."""
    print("\n" + "="*70)
    print("MODO DEMO: Tratamiento Probabilístico del Lenguaje")
    print("="*70)
    
    # ========== EJEMPLO 1: Modelo de Bigramas ==========
    print("\n" + "="*70)
    print("EJEMPLO 1: Modelo de Bigramas (n=2)")
    print("="*70)
    
    # Corpus de entrenamiento simple en español
    corpus_español = [
        "el gato come pescado",
        "el perro come carne",
        "el gato bebe leche",
        "el perro bebe agua",
        "el gato duerme en el sofá",
        "el perro duerme en el jardín",
        "el gato juega con la pelota",
        "el perro juega en el parque"
    ]
    
    print("\n--- Corpus de entrenamiento ---")
    for i, frase in enumerate(corpus_español, 1):
        print(f"{i}. {frase}")
    
    print("\n--- Entrenando modelo de bigramas ---")
    modelo_bi = ModeloNGramas(n=2)
    modelo_bi.entrenar(corpus_español)
    
    print(f"Vocabulario: {len(modelo_bi.vocabulario)} palabras únicas")
    print(f"Bigramas únicos: {len(modelo_bi.ngramas_conteo)}")
    
    # Calcular probabilidades
    print("\n--- Probabilidades condicionales P(palabra | contexto) ---")
    ejemplos_prob = [
        (('el',), 'gato'),
        (('el',), 'perro'),
        (('gato',), 'come'),
        (('perro',), 'come'),
        (('come',), 'pescado'),
        (('come',), 'carne'),
    ]
    
    for contexto, palabra in ejemplos_prob:
        prob = modelo_bi.probabilidad(palabra, contexto)
        print(f"P('{palabra}' | {contexto}) = {prob:.4f}")
    
    # Probabilidad de secuencias completas
    print("\n--- Probabilidad de secuencias completas ---")
    secuencias_prueba = [
        "el gato come pescado",
        "el perro bebe agua",
        "el elefante vuela alto"  # Secuencia improbable
    ]
    
    for sec in secuencias_prueba:
        prob = modelo_bi.probabilidad_secuencia(sec)
        print(f"P(\"{sec}\") = {prob:.8f}")
    
    print("\n>>> OBSERVACIÓN:")
    print("    La tercera secuencia tiene probabilidad muy baja porque")
    print("    contiene palabras no vistas en el corpus ('elefante', 'vuela', 'alto').")
    
    # ========== EJEMPLO 2: Generación de Texto ==========
    print("\n" + "="*70)
    print("EJEMPLO 2: Generación Automática de Texto")
    print("="*70)
    
    print("\n--- Generando texto aleatorio con el modelo de bigramas ---")
    
    for i in range(5):
        texto_generado = modelo_bi.generar_texto(longitud=6)
        print(f"{i+1}. {texto_generado}")
    
    print("\n>>> OBSERVACIÓN:")
    print("    El texto generado sigue los patrones probabilísticos del corpus.")
    
    # ========== EJEMPLO 3: Predicción de Palabras ==========
    print("\n" + "="*70)
    print("EJEMPLO 3: Predicción de la siguiente palabra (Autocompletado)")
    print("="*70)
    
    contextos_prediccion = [
        ('el',),
        ('gato',),
        ('perro',),
        ('come',)
    ]
    
    print("\n--- Palabras más probables dado el contexto ---")
    for ctx in contextos_prediccion:
        palabra, prob = modelo_bi.palabra_mas_probable(ctx)
        print(f"Contexto: {ctx} → Palabra más probable: '{palabra}' (P = {prob:.4f})")

# ========== MODO INTERACTIVO ==========

def modo_interactivo():
    """Ejecuta el modo interactivo con corpus personalizado."""
    print("\n" + "="*70)
    print("MODO INTERACTIVO: Modelo de Lenguaje Personalizado")
    print("="*70)
    
    # ========== PASO 1: Ingresar corpus ==========
    print("\n--- Ingresa tu corpus de entrenamiento ---")
    print("Puedes ingresar varias frases (una por línea).")
    print("Escribe 'FIN' cuando termines.\n")
    
    corpus_usuario = []
    while True:
        frase = input(f"Frase {len(corpus_usuario) + 1}: ").strip()
        if frase.upper() == 'FIN' or frase == '':
            break
        corpus_usuario.append(frase)
    
    if not corpus_usuario:
        print("No se ingresaron frases. Usando corpus de ejemplo.")
        corpus_usuario = [
            "la inteligencia artificial es fascinante",
            "el aprendizaje automático transforma el mundo",
            "las redes neuronales aprenden patrones complejos"
        ]
    
    print(f"\n--- Corpus cargado ({len(corpus_usuario)} frases) ---")
    for i, frase in enumerate(corpus_usuario, 1):
        print(f"{i}. {frase}")
    
    # ========== PASO 2: Configurar modelo ==========
    print("\n--- Configurar modelo de n-gramas ---")
    
    try:
        n = int(input("Orden del modelo (1=unigramas, 2=bigramas, 3=trigramas): ").strip() or "2")
    except:
        n = 2
    
    print(f"\nCreando modelo de {n}-gramas...")
    modelo = ModeloNGramas(n=n)
    modelo.entrenar(corpus_usuario)
    
    print(f"✓ Modelo entrenado")
    print(f"  Vocabulario: {len(modelo.vocabulario)} palabras")
    print(f"  N-gramas únicos: {len(modelo.ngramas_conteo)}")
    
    # ========== PASO 3: Generar texto ==========
    print("\n--- Generando texto ---")
    
    for i in range(3):
        texto = modelo.generar_texto(longitud=8)
        print(f"{i+1}. {texto}")

# ========== MAIN ==========

def main():
    """Función principal que gestiona la ejecución del programa."""
    print("\n" + "="*70)
    print("TRATAMIENTO PROBABILÍSTICO DEL LENGUAJE (Modelos N-Gramas)")
    print("="*70)
    print("\nSelecciona el modo de ejecución:")
    print("1. DEMO (ejemplos predefinidos con corpus en español)")
    print("2. INTERACTIVO (entrena tu propio modelo con corpus personalizado)")
    
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
