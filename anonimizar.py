"""
Reemplaza nombres de tags por su ID de la tabla `tag` (ej. "#37"), para poder compartir
reportes sin exponer los nombres reales.
"""


def obtener_mapa_anonimo(conexion):
    """Devuelve {nombre_en_minúsculas: "#id"} con los tags de video de la DB."""
    cursor = conexion.cursor()
    cursor.execute("SELECT id, name FROM tag WHERE type = 'VIDEO' ORDER BY id")
    mapa = {}
    for tag_id, nombre in cursor.fetchall():
        mapa.setdefault((nombre or "").strip().lower(), f"#{tag_id}")  # si hay duplicados, gana el id menor
    cursor.close()
    return mapa


def anonimizar_tag(nombre, mapa):
    """"#id" del tag; "#?" si el nombre no existe en la tabla `tag` (ej. sugerencia de un tag borrado)."""
    return mapa.get((nombre or "").strip().lower(), "#?")
