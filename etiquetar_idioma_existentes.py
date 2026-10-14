"""
Detecta música e idioma hablado (ver detectar_idioma.py) de los videos ya existentes y lo deja
como sugerencia en `video_tag_temporal` (confirm = 0) para revisarlo en el frontend.

Videos que procesa:
  - Etiquetados (en video_tag): la sugerencia = sus tags actuales + los tags detectados.
    Debe incluir los tags actuales porque al confirmar el backend REEMPLAZA todos los tags
    del video por los que vienen en la sugerencia.
  - Pendientes de revisión (video_tag_temporal con confirm = 0 y sin video_tag): se agrega el
    tags detectados a la sugerencia existente (descartando idioma/música que hubiera puesto el KNN).

--corregir-music: vuelve a procesar solo los videos a los que una versión anterior les sugirió
"Music" (entonces Music = "sonido sin voz", que marcaba el ruido de fondo como música). Les quita
ese Music y deja la sugerencia nueva pendiente de revisión, aunque ya la hubieras confirmado.

Si un video ya tenía fila en video_tag_temporal, se guarda un respaldo de la fila original
en salidas/ antes de sobrescribirla. Se puede cortar y reanudar: los videos ya procesados
quedan registrados en salidas/resultados_idioma.csv y se saltan.
"""
import os
import csv
import argparse
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

from detectar_idioma import detectar_idioma, asegurar_tags_video, TAGS_IDIOMA_Y_MUSICA, TAG_MUSICA

load_dotenv()

BASE_PATH        = os.path.normpath(os.getenv("BASE_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "media"))
CARPETA_SALIDAS  = "salidas"
ARCHIVO_RESULTADOS = os.path.join(CARPETA_SALIDAS, "resultados_idioma.csv")
ARCHIVO_RESPALDO   = os.path.join(CARPETA_SALIDAS, "respaldo_video_tag_temporal_idioma.csv")

# "tag" guarda todos los tags detectados separados por "; " (ej. "Music; English")
CAMPOS_RESULTADOS = ["video_id", "origen", "tag", "codigo", "probabilidad", "segundos_voz",
                     "accion", "fecha", "prob_musica", "prob_habla", "prob_canto"]


def conectar_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE")
    )


def separar_tags(texto):
    return [t.strip() for t in (texto or "").split(",") if t.strip()]

# ============================================================
# CANDIDATOS
# ============================================================

def obtener_candidatos(cursor):
    """
    Devuelve lista de dicts {video_id, video_path, origen, tags_base, temporal_id}.
    origen = "etiquetado" | "pendiente".
    """
    cursor.execute("""
        SELECT v.id AS video_id, v.video_path,
               GROUP_CONCAT(t.name SEPARATOR ',') AS tags_reales,
               vtt.id AS temporal_id, vtt.confirm
        FROM video v
        JOIN video_tag vt ON vt.video_id = v.id
        JOIN tag t ON t.id = vt.tag_id
        LEFT JOIN video_tag_temporal vtt ON vtt.video_id = v.id
        GROUP BY v.id, v.video_path, vtt.id, vtt.confirm
        ORDER BY v.id
    """)
    candidatos = [{
        "video_id":    f["video_id"],
        "video_path":  f["video_path"],
        "origen":      "etiquetado",
        "tags_base":   separar_tags(f["tags_reales"]),  # tags confirmados: se respetan tal cual
        "temporal_id": f["temporal_id"],
        # sin fila temporal (etiquetado a mano) cuenta como confirmado
        "confirmado":  f["temporal_id"] is None or bool(f["confirm"]),
    } for f in cursor.fetchall()]

    cursor.execute("""
        SELECT vtt.id AS temporal_id, v.id AS video_id, v.video_path, vtt.tags_suggest
        FROM video_tag_temporal vtt
        JOIN video v ON v.id = vtt.video_id
        LEFT JOIN video_tag vt ON vt.video_id = v.id
        WHERE vtt.confirm = FALSE AND vt.video_id IS NULL
        ORDER BY v.id
    """)
    for f in cursor.fetchall():
        candidatos.append({
            "video_id":    f["video_id"],
            "video_path":  f["video_path"],
            "origen":      "pendiente",
            # idioma/música puestos por el KNN no son fiables: se reemplazan por la detección
            "tags_base":   [t for t in separar_tags(f["tags_suggest"])
                            if t.lower() not in TAGS_IDIOMA_Y_MUSICA],
            "temporal_id": f["temporal_id"],
            "confirmado":  False,
        })
    return candidatos

# ============================================================
# ARCHIVOS DE SALIDA
# ============================================================

def leer_resultados():
    """Último registro de cada video en resultados_idioma.csv → {video_id: fila}."""
    if not os.path.exists(ARCHIVO_RESULTADOS):
        return {}
    with open(ARCHIVO_RESULTADOS, newline="", encoding="utf-8") as f:
        return {int(fila["video_id"]): fila for fila in csv.DictReader(f)}


def _anexar_csv(ruta, campos, fila):
    nuevo = not os.path.exists(ruta)
    if not nuevo:
        with open(ruta, newline="", encoding="utf-8") as f:
            encabezado = next(csv.reader(f), [])
        if encabezado != campos:
            # Archivo de una versión anterior con otras columnas: se reescribe con las nuevas
            with open(ruta, newline="", encoding="utf-8") as f:
                filas = list(csv.DictReader(f))
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=campos)
                writer.writeheader()
                writer.writerows({c: fila_vieja.get(c, "") for c in campos} for fila_vieja in filas)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if nuevo:
            writer.writeheader()
        writer.writerow(fila)


def respaldar_temporal(cursor, temporal_id):
    cursor.execute(
        "SELECT id, video_id, tags_suggest, confirm, date_creation FROM video_tag_temporal WHERE id = %s",
        (temporal_id,)
    )
    for fila in cursor.fetchall():
        _anexar_csv(ARCHIVO_RESPALDO, ["id", "video_id", "tags_suggest", "confirm", "date_creation"], fila)

# ============================================================
# FLUJO PRINCIPAL
# ============================================================

def guardar_sugerencia(cursor, candidato, tags_sugeridos):
    texto = ", ".join(tags_sugeridos)
    if candidato["temporal_id"]:
        respaldar_temporal(cursor, candidato["temporal_id"])
        cursor.execute("""
            UPDATE video_tag_temporal
            SET tags_suggest = %s, confirm = FALSE, date_creation = %s
            WHERE id = %s
        """, (texto, datetime.now(), candidato["temporal_id"]))
    else:
        cursor.execute("""
            INSERT INTO video_tag_temporal (video_id, tags_suggest, confirm, date_creation)
            VALUES (%s, %s, %s, %s)
        """, (candidato["video_id"], texto, False, datetime.now()))


def _sin_music(tags):
    return [t for t in tags if t.lower() != TAG_MUSICA.lower()]


def etiquetar_idioma_existentes(limite=None, dry_run=False, rehacer=False, corregir_music=False):
    os.makedirs(CARPETA_SALIDAS, exist_ok=True)

    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)

    candidatos = obtener_candidatos(cursor)
    resultados = leer_resultados()

    if corregir_music:
        # Videos a los que la versión anterior (sin AST: registros sin prob_musica) les sugirió Music
        ids = {vid for vid, fila in resultados.items()
               if fila["accion"] == "sugerido" and not fila.get("prob_musica")
               and TAG_MUSICA.lower() in {t.strip().lower() for t in fila["tag"].split(";")}}
        por_procesar = [c for c in candidatos if c["video_id"] in ids]
        for c in por_procesar:
            # Ese Music lo puso el script (no estaba antes), así que se puede quitar sin perder nada
            c["tags_originales"] = list(c["tags_base"])
            c["tags_base"] = _sin_music(c["tags_base"])
        print(f"\n🎵 Corrigiendo Music: {len(por_procesar)} videos con Music sugerido anteriormente")
    else:
        procesados   = set() if rehacer else set(resultados)
        por_procesar = [c for c in candidatos if c["video_id"] not in procesados]
        ya_procesados = sum(1 for c in candidatos if c["video_id"] in procesados)
        print(f"\n🗣  Videos candidatos: {len(candidatos)} | Ya procesados: {ya_procesados} "
              f"| A procesar ahora: {len(por_procesar)}")

    if limite:
        por_procesar = por_procesar[:limite]
    if dry_run:
        print("   (dry-run: no se escribe en la DB ni en salidas/)")
    print()

    resumen = {}
    try:
        for n, cand in enumerate(por_procesar, start=1):
            ruta = os.path.join(BASE_PATH, cand["video_path"])
            print(f"  [{n}/{len(por_procesar)}] Video {cand['video_id']} ({cand['origen']})...", end=" ")

            deteccion = {}
            if not os.path.exists(ruta):
                accion = "archivo_no_encontrado"
            else:
                deteccion = detectar_idioma(ruta)
                base_lower = {t.lower() for t in cand["tags_base"]}
                nuevos = [t for t in deteccion["tags"] if t.lower() not in base_lower]

                if corregir_music:
                    final = {t.lower() for t in cand["tags_base"]} | {t.lower() for t in nuevos}
                    if cand["confirmado"] and final == {t.lower() for t in cand["tags_originales"]}:
                        accion = "sin_cambios"  # lo ya confirmado coincide con la detección nueva
                    else:
                        # Se reescribe aunque no haya tags nuevos: hay que quitar el Music mal puesto
                        accion = "corregido"
                elif nuevos:
                    accion = "sugerido"
                elif deteccion["tags"]:
                    accion = "ya_tenia_tag"
                else:
                    accion = "sin_tag"  # silencio, ruido o idioma dudoso: no hay nada que sugerir

                if accion in ("sugerido", "corregido") and not dry_run:
                    asegurar_tags_video(cursor, nuevos)
                    guardar_sugerencia(cursor, cand, sorted(set(cand["tags_base"]) | set(nuevos)))
                    conexion.commit()

            detalle = ", ".join(deteccion.get("tags") or []) or "-"
            if deteccion.get("probabilidad") is not None:
                detalle += f" (idioma {deteccion['codigo']} {deteccion['probabilidad']:.0%})"
            print(f"{accion}: {detalle}")
            resumen[accion] = resumen.get(accion, 0) + 1

            # Archivo no encontrado no se registra: así se reintenta en la próxima corrida.
            if not dry_run and accion != "archivo_no_encontrado":
                _anexar_csv(ARCHIVO_RESULTADOS, CAMPOS_RESULTADOS, {
                    "video_id":     cand["video_id"],
                    "origen":       cand["origen"],
                    "tag":          "; ".join(deteccion.get("tags") or []),
                    "codigo":       deteccion.get("codigo") or "",
                    "probabilidad": _valor(deteccion.get("probabilidad")),
                    "segundos_voz": deteccion.get("segundos_voz", ""),
                    "accion":       accion,
                    "fecha":        datetime.now().isoformat(timespec="seconds"),
                    "prob_musica":  _valor(deteccion.get("prob_musica")),
                    "prob_habla":   _valor(deteccion.get("prob_habla")),
                    "prob_canto":   _valor(deteccion.get("prob_canto")),
                })
    except KeyboardInterrupt:
        print("\n⚠  Detenido por el usuario. Vuelve a correr el script para continuar donde quedó.")
    finally:
        cursor.close()
        conexion.close()

    print("\n📊 Resumen:")
    for accion, cantidad in sorted(resumen.items()):
        print(f"   · {accion:<22} {cantidad}")
    if not dry_run:
        print(f"\n   Detalle por video → {ARCHIVO_RESULTADOS}")


def _valor(v):
    return "" if v is None else v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sugiere los tags de música e idioma hablado para los videos existentes."
    )
    parser.add_argument("--limite", type=int, help="Procesar como máximo N videos en esta corrida")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detecta y muestra el resultado sin escribir en la DB ni en salidas/")
    parser.add_argument("--rehacer", action="store_true",
                        help="Ignora salidas/resultados_idioma.csv y vuelve a procesar todo")
    parser.add_argument("--corregir-music", action="store_true",
                        help="Reprocesa solo los videos a los que antes se les sugirió Music, quitando ese Music")
    args = parser.parse_args()

    etiquetar_idioma_existentes(limite=args.limite, dry_run=args.dry_run, rehacer=args.rehacer,
                                corregir_music=args.corregir_music)
