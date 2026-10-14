"""
Quita los tags de idioma puestos a mano para que etiquetar_idioma_existentes.py los genere de cero.

  - Borra las relaciones video ↔ tag en `video_tag` (NO borra los tags de la tabla `tag`:
    se comparten con imágenes/mangas y se reutilizan al sugerir el idioma detectado).
  - Quita esos tags de las sugerencias pendientes (`video_tag_temporal` con confirm = 0).
  - Antes de tocar nada guarda un CSV de respaldo en salidas/ para poder restaurar.

--borrar-pendientes: borra TODAS las sugerencias pendientes de video_tag_temporal (confirm = 0),
tanto las del script de idioma como las del clasificador, para empezar la prueba desde cero:
  - Las sugerencias ya confirmadas no se tocan (ni video_tag).
  - Guarda un CSV de respaldo con las filas borradas en salidas/.
  - Quita de salidas/resultados_idioma.csv todos los videos sin sugerencia confirmada, para que
    etiquetar_idioma_existentes.py los vuelva a procesar.
  - Los videos sin etiquetar quedan sin sugerencia: procesar_lote() los vuelve a clasificar.
"""
import os
import csv
import argparse
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TAGS_POR_DEFECTO = ["Spanish", "English"]
CARPETA_SALIDAS  = "salidas"


def conectar_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE")
    )


def quitar_tags_de_sugerencia(tags_suggest, nombres_lower):
    tags = [t.strip() for t in (tags_suggest or "").split(",") if t.strip()]
    return ", ".join(t for t in tags if t.lower() not in nombres_lower)


def borrar_tags_idioma(nombres, dry_run=False):
    nombres_lower = {n.lower() for n in nombres}
    marcadores    = ", ".join(["%s"] * len(nombres_lower))

    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)

    # 1. Relaciones a borrar en video_tag
    cursor.execute(f"""
        SELECT vt.video_id, vt.tag_id, t.name
        FROM video_tag vt
        JOIN tag t ON t.id = vt.tag_id
        WHERE t.type = 'VIDEO' AND LOWER(t.name) IN ({marcadores})
    """, tuple(nombres_lower))
    relaciones = cursor.fetchall()

    # 2. Sugerencias pendientes que mencionan esos tags
    cursor.execute("SELECT id, video_id, tags_suggest FROM video_tag_temporal WHERE confirm = FALSE")
    pendientes = []
    for fila in cursor.fetchall():
        nuevo = quitar_tags_de_sugerencia(fila["tags_suggest"], nombres_lower)
        if nuevo != ", ".join(t.strip() for t in (fila["tags_suggest"] or "").split(",") if t.strip()):
            pendientes.append({**fila, "nuevo": nuevo})

    print(f"\n🏷  Tags a quitar: {', '.join(nombres)}")
    conteo = {}
    for r in relaciones:
        conteo[r["name"]] = conteo.get(r["name"], 0) + 1
    for nombre, n in sorted(conteo.items()):
        print(f"   · {nombre:<12} en {n} videos (video_tag)")
    print(f"   · Sugerencias pendientes a limpiar: {len(pendientes)}")

    if not relaciones and not pendientes:
        print("✅ No hay nada que borrar.")
        cursor.close()
        conexion.close()
        return

    if dry_run:
        print("\n🔎 --dry-run: no se modificó nada.")
        cursor.close()
        conexion.close()
        return

    # 3. Respaldo antes de modificar
    os.makedirs(CARPETA_SALIDAS, exist_ok=True)
    ruta_respaldo = os.path.join(
        CARPETA_SALIDAS, f"respaldo_borrado_tags_idioma_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    with open(ruta_respaldo, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tabla", "id", "video_id", "tag_id", "tag_name", "tags_suggest_anterior"])
        for r in relaciones:
            writer.writerow(["video_tag", "", r["video_id"], r["tag_id"], r["name"], ""])
        for p in pendientes:
            writer.writerow(["video_tag_temporal", p["id"], p["video_id"], "", "", p["tags_suggest"]])
    print(f"\n💾 Respaldo guardado → {ruta_respaldo}")

    # 4. Borrado en una sola transacción
    try:
        cursor.executemany(
            "DELETE FROM video_tag WHERE video_id = %s AND tag_id = %s",
            [(r["video_id"], r["tag_id"]) for r in relaciones]
        )
        cursor.executemany(
            "UPDATE video_tag_temporal SET tags_suggest = %s WHERE id = %s",
            [(p["nuevo"], p["id"]) for p in pendientes]
        )
        conexion.commit()
    except Exception:
        conexion.rollback()
        print("❌ Error al borrar; no se aplicó ningún cambio.")
        raise
    finally:
        cursor.close()
        conexion.close()

    print(f"✅ Borradas {len(relaciones)} relaciones y limpiadas {len(pendientes)} sugerencias pendientes.")
    print("   La memoria se actualiza sola en el próximo lote (sincronización incremental).")


def _leer_csv(ruta):
    if not os.path.exists(ruta):
        return []
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def borrar_sugerencias_pendientes(dry_run=False):
    # Rutas de los archivos que escribe el script de etiquetado (fuente única)
    from etiquetar_idioma_existentes import ARCHIVO_RESULTADOS, ARCHIVO_RESPALDO

    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)
    cursor.execute("""
        SELECT vtt.id, vtt.video_id, vtt.tags_suggest, vtt.confirm, vtt.date_creation,
               EXISTS (SELECT 1 FROM video_tag vt WHERE vt.video_id = vtt.video_id) AS etiquetado
        FROM video_tag_temporal vtt
        WHERE vtt.confirm = FALSE
    """)
    pendientes = cursor.fetchall()
    cursor.execute("SELECT video_id FROM video_tag_temporal WHERE confirm = TRUE")
    confirmados = {f["video_id"] for f in cursor.fetchall()}

    resultados = _leer_csv(ARCHIVO_RESULTADOS)
    ids_script = {int(r["video_id"]) for r in resultados}
    resultados_restantes = [r for r in resultados if int(r["video_id"]) in confirmados]

    del_script     = sum(1 for f in pendientes if f["video_id"] in ids_script)
    sin_etiquetar  = sum(1 for f in pendientes if not f["etiquetado"])
    print(f"\n🧹 Sugerencias pendientes a borrar: {len(pendientes)}")
    print(f"   · Del script de idioma                      : {del_script}")
    print(f"   · Del clasificador u otras                  : {len(pendientes) - del_script}")
    print(f"   · De videos sin etiquetar (se reclasifican) : {sin_etiquetar}")
    print(f"   · Sugerencias confirmadas, no se tocan      : {len(confirmados)}")
    print(f"   · Videos que el script de idioma reprocesará: "
          f"{len(ids_script) - len({int(r['video_id']) for r in resultados_restantes})}")

    if not pendientes and len(resultados_restantes) == len(resultados):
        print("✅ No hay nada que borrar.")
        cursor.close()
        conexion.close()
        return

    if dry_run:
        print("\n🔎 --dry-run: no se modificó nada.")
        cursor.close()
        conexion.close()
        return

    os.makedirs(CARPETA_SALIDAS, exist_ok=True)
    ruta_respaldo = os.path.join(
        CARPETA_SALIDAS, f"respaldo_borrado_pendientes_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    with open(ruta_respaldo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "video_id", "tags_suggest", "confirm", "date_creation"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pendientes)
    print(f"\n💾 Respaldo de las sugerencias borradas → {ruta_respaldo}")

    try:
        cursor.executemany("DELETE FROM video_tag_temporal WHERE id = %s AND confirm = FALSE",
                           [(f["id"],) for f in pendientes])
        conexion.commit()
    except Exception:
        conexion.rollback()
        print("❌ Error al borrar; no se aplicó ningún cambio.")
        raise
    finally:
        cursor.close()
        conexion.close()

    # Solo se reescriben los registros una vez confirmada la transacción
    if resultados:
        with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
            writer.writeheader()
            writer.writerows(resultados_restantes)

    # Los respaldos de filas que ya no existen no sirven para restaurar nada
    respaldos = _leer_csv(ARCHIVO_RESPALDO)
    ids_borrados = {f["id"] for f in pendientes}
    if respaldos:
        with open(ARCHIVO_RESPALDO, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(respaldos[0].keys()))
            writer.writeheader()
            writer.writerows(r for r in respaldos if int(r["id"]) not in ids_borrados)

    print(f"✅ Borradas {len(pendientes)} sugerencias pendientes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quita tags de idioma puestos a mano de video_tag, o borra todas las sugerencias pendientes."
    )
    parser.add_argument("--tags", nargs="+", default=TAGS_POR_DEFECTO,
                        help=f"Nombres de los tags a quitar (default: {' '.join(TAGS_POR_DEFECTO)})")
    parser.add_argument("--borrar-pendientes", action="store_true",
                        help="Borra TODAS las sugerencias pendientes (confirm = 0) de video_tag_temporal")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra lo que se haría")
    args = parser.parse_args()

    if args.borrar_pendientes:
        borrar_sugerencias_pendientes(dry_run=args.dry_run)
    else:
        borrar_tags_idioma(args.tags, dry_run=args.dry_run)
