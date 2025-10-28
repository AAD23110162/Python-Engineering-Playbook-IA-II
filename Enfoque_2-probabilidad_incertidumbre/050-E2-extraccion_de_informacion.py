"""
050-E2-extraccion_de_informacion.py
--------------------------------
Este script introduce Extracción de Información (IE):
- Reconocimiento de entidades, relaciones y eventos a nivel conceptual.
- Enfoques basados en reglas y en aprendizaje supervisado/probabilístico.
- Evaluación de IE: precisión, exhaustividad, F1.
- Variables y funciones en español.

El programa puede ejecutarse en dos modos:
1. DEMO: extracción sobre textos de ejemplo con patrones simples.
2. INTERACTIVO: definición de plantillas y evaluación sobre un corpus cargado.

Autor: Alejandro Aguirre Díaz
"""

# Importar librerías necesarias
import re  # Expresiones regulares para patrones
from collections import defaultdict  # Diccionario con listas por defecto

# -----------------------------
# Extracción de Entidades con regex simples
# -----------------------------
PATRONES_ENTIDADES = {
	# Patrón para correos electrónicos
	'EMAIL': re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I),
	# Patrón para fechas (varios formatos)
	'FECHA': re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
	# Patrón para teléfonos
	'TELEFONO': re.compile(r"\b\+?\d{1,3}?[\s-]?(\d{2,4}[\s-]?){2,4}\d{2,4}\b"),
}

def extraer_personas(texto):
	"""
	Heurística: secuencias Capitalizadas (Nombre Apellido), evitando inicio de oración genérico.
	"""
	# Busca secuencias de palabras capitalizadas (ejemplo: Nombre Apellido)
	candidatos = re.findall(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)\b", texto)
	return list(set(candidatos))

def extraer_organizaciones(texto):
	"""
	Heurística: palabras en MAYÚSCULAS de 2+ letras o sufijos típicos (S.A., S.L.).
	"""
	# Busca palabras en MAYÚSCULAS y organizaciones con sufijos típicos
	mayus = re.findall(r"\b([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*)\b", texto)
	sufijos = re.findall(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+(?:S\.A\.|S\.L\.|Inc\.|Ltd\.))\b", texto)
	return list(set(mayus + sufijos))

def extraer_entidades(texto):
	entidades = defaultdict(list)
	# Extraer entidades usando patrones regex directos
	for tipo, patron in PATRONES_ENTIDADES.items():
		entidades[tipo] = patron.findall(texto)
	# Extraer personas y organizaciones con heurísticas
	entidades['PERSONA'] = extraer_personas(texto)
	entidades['ORGANIZACION'] = extraer_organizaciones(texto)
	return entidades

# -----------------------------
# Extracción de Relaciones (patrones simples)
# -----------------------------
PATRONES_RELACIONES = [
	# Patrón para relaciones de trabajo
	('TRABAJA_EN', re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\s+trabaja\s+en\s+([A-ZÁÉÍÓÚÑ][\w\.]+(?:\s+[A-ZÁÉÍÓÚÑ][\w\.]+)*)", re.I)),
	# Patrón para relaciones de residencia
	('VIVE_EN', re.compile(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)\s+vive\s+en\s+([A-ZÁÉÍÓÚÑ][\w\.]+(?:\s+[A-ZÁÉÍÓÚÑ][\w\.]+)*)", re.I)),
]

def extraer_relaciones(texto):
	relaciones = []
	# Buscar relaciones en el texto usando patrones
	for tipo, patron in PATRONES_RELACIONES:
		for m in patron.findall(texto):
			sujeto, objeto = m
			relaciones.append({'tipo': tipo, 'sujeto': sujeto.strip(), 'objeto': objeto.strip()})
	return relaciones

# -----------------------------
# DEMO
# -----------------------------
def modo_demo():
	print("\n--- MODO DEMO: Extracción de Información ---")
	# Texto de ejemplo para extracción
	texto = (
		"María Pérez trabaja en ACME S.A. desde 2021-05-10. "
		"Su correo es maria.perez@example.com y su teléfono es +34 600-123-456. "
		"Juan López vive en Madrid y trabaja en TECNOLOGIAS GLOBALES."
	)
	print("Texto:")
	print(texto)
	# Extraer entidades
	ents = extraer_entidades(texto)
	print("\nEntidades encontradas:")
	for tipo, valores in ents.items():
		print(f"- {tipo}: {valores}")
	# Extraer relaciones
	rels = extraer_relaciones(texto)
	print("\nRelaciones:")
	for r in rels:
		print(f"- {r['tipo']}: {r['sujeto']} -> {r['objeto']}")

# -----------------------------
# INTERACTIVO
# -----------------------------
def modo_interactivo():
	print("\n--- MODO INTERACTIVO: Ingresa tu texto ---")
	# Solicitar texto al usuario
	texto = input("Texto: \n")
	# Extraer entidades
	ents = extraer_entidades(texto)
	print("\nEntidades encontradas:")
	for tipo, valores in ents.items():
		print(f"- {tipo}: {valores}")
	# Extraer relaciones
	rels = extraer_relaciones(texto)
	print("\nRelaciones:")
	for r in rels:
		print(f"- {r['tipo']}: {r['sujeto']} -> {r['objeto']}")

# -----------------------------
# Menú principal
# -----------------------------
if __name__ == "__main__":
	# Menú principal para seleccionar el modo de ejecución
	print("\nScript 050-E2-extraccion_de_informacion.py")
	print("Selecciona modo de ejecución:")
	print("1. DEMO (patrones simples)")
	print("2. INTERACTIVO (extrae de tu texto)")
	modo = input("Modo [1/2]: ").strip()
	if modo == "1":
		modo_demo()
	elif modo == "2":
		modo_interactivo()
	else:
		print("Opción no válida. Ejecutando DEMO por defecto.")
		modo_demo()
