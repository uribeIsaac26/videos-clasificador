"""
Quita los tags de idioma puestos a mano para que etiquetar_idioma_existentes.py los genere de cero.

  - Borra las relaciones video ↔ tag en `video_tag` (NO borra los tags de la tabla `tag`:
    se comparten con imágenes/mangas y se reutilizan al sugerir el idioma detectado).
  - Quita esos tags de las sugerencias pendientes (`video_tag_temporal` con confirm = 0).
  - Antes de tocar nada guarda un CSV de respaldo en salidas/ para poder restaurar.

--revertir-sugerencias: deshace las sugerencias de etiquetar_idioma_existentes.py que siguen
pendientes (confirm = 0), para volver a probar la detección desde cero:
  - Si el video ya tenía fila en video_tag_temporal antes del script, se restaura la original
    (sacada de salidas/respaldo_video_tag_temporal_idioma.csv).
  - Si la fila la creó el script, se borra.
  - Las sugerencias que ya confirmaste no se tocan.
  - Se quitan de salidas/resultados_idioma.csv todos los videos salvo los ya confirmados, para
    que el script los vuelva a procesar.
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


def revertir_sugerencias_idioma(dry_run=False):
    # Rutas de los archivos que escribe el script de etiquetado (fuente única)
    from etiquetar_idioma_existentes import ARCHIVO_RESULTADOS, ARCHIVO_RESPALDO

    resultados = _leer_csv(ARCHIVO_RESULTADOS)
    if not resultados:
        print(f"✅ No hay {ARCHIVO_RESULTADOS}: no hay sugerencias de idioma que revertir.")
        return

    # Cuántas veces escribió el script la sugerencia de cada video
    escrituras = {}
    for r in resultados:
        if r["accion"] in ("sugerido", "corregido"):
            vid = int(r["video_id"])
            escrituras[vid] = escrituras.get(vid, 0) + 1
    ids_sugeridos = sorted(escrituras)

    # Cada escritura sobre una fila existente deja un respaldo previo. Si una fila tiene tantos
    # respaldos como escrituras, el primero es la fila original (anterior al script); si tiene
    # uno menos, la primera escritura fue un INSERT (la fila la creó el script).
    respaldos = _leer_csv(ARCHIVO_RESPALDO)
    respaldos_por_id = {}
    for fila in respaldos:
        respaldos_por_id.setdefault(int(fila["id"]), []).append(fila)

    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)

    filas = []
    if ids_sugeridos:
        marcadores = ", ".join(["%s"] * len(ids_sugeridos))
        cursor.execute(f"""
            SELECT id, video_id, tags_suggest, confirm, date_creation
            FROM video_tag_temporal WHERE video_id IN ({marcadores})
        """, tuple(ids_sugeridos))
        filas = cursor.fetchall()

    pendientes  = [f for f in filas if not f["confirm"]]
    confirmados = {f["video_id"] for f in filas if f["confirm"]}
    originales  = {
        f["id"]: respaldos_por_id[f["id"]][0] for f in pendientes
        if len(respaldos_por_id.get(f["id"], [])) >= escrituras[f["video_id"]]
    }
    a_restaurar = [f for f in pendientes if f["id"] in originales]
    a_borrar    = [f for f in pendientes if f["id"] not in originales]
    resultados_restantes = [r for r in resultados if int(r["video_id"]) in confirmados]

    print(f"\n↩️  Sugerencias de idioma pendientes: {len(pendientes)}")
    print(f"   · Se restauran a la sugerencia original : {len(a_restaurar)}")
    print(f"   · Se borran (las creó el script)        : {len(a_borrar)}")
    print(f"   · Ya confirmadas, no se tocan           : {len(confirmados)}")
    print(f"   · Videos que se volverán a procesar     : "
          f"{len({int(r['video_id']) for r in resultados}) - len(confirmados)}")

    if dry_run:
        print("\n🔎 --dry-run: no se modificó nada.")
        cursor.close()
        conexion.close()
        return

    os.makedirs(CARPETA_SALIDAS, exist_ok=True)
    ruta_respaldo = os.path.join(
        CARPETA_SALIDAS, f"respaldo_revertir_sugerencias_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    with open(ruta_respaldo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "video_id", "tags_suggest", "confirm", "date_creation"])
        writer.writeheader()
        writer.writerows(pendientes)
    print(f"\n💾 Respaldo de las sugerencias actuales → {ruta_respaldo}")

    try:
        cursor.executemany(
            "UPDATE video_tag_temporal SET tags_suggest = %s, confirm = %s, date_creation = %s WHERE id = %s",
            [(originales[f["id"]]["tags_suggest"],
              bool(int(originales[f["id"]]["confirm"] or 0)),
              originales[f["id"]]["date_creation"] or None,
              f["id"]) for f in a_restaurar]
        )
        cursor.executemany(
            "DELETE FROM video_tag_temporal WHERE id = %s",
            [(f["id"],) for f in a_borrar]
        )
        conexion.commit()
    except Exception:
        conexion.rollback()
        print("❌ Error al revertir; no se aplicó ningún cambio.")
        raise
    finally:
        cursor.close()
        conexion.close()

    # Solo se reescriben los registros una vez confirmada la transacción. Los respaldos de las
    # filas revertidas se quitan: esas filas vuelven a su estado previo al script (o ya no existen),
    # así el conteo respaldos/escrituras sigue cuadrando en la próxima corrida.
    with open(ARCHIVO_RESULTADOS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados_restantes)

    ids_revertidos = {f["id"] for f in pendientes}
    if respaldos:
        with open(ARCHIVO_RESPALDO, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(respaldos[0].keys()))
            writer.writeheader()
            writer.writerows(r for r in respaldos if int(r["id"]) not in ids_revertidos)

    print(f"✅ Restauradas {len(a_restaurar)} y borradas {len(a_borrar)} sugerencias. "
          f"Ya puedes volver a correr etiquetar_idioma_existentes.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quita tags de idioma puestos a mano de video_tag, o revierte las sugerencias de idioma pendientes."
    )
    parser.add_argument("--tags", nargs="+", default=TAGS_POR_DEFECTO,
                        help=f"Nombres de los tags a quitar (default: {' '.join(TAGS_POR_DEFECTO)})")
    parser.add_argument("--revertir-sugerencias", action="store_true",
                        help="Deshace las sugerencias pendientes de etiquetar_idioma_existentes.py para repetir la prueba")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra lo que se haría")
    args = parser.parse_args()

    if args.revertir_sugerencias:
        revertir_sugerencias_idioma(dry_run=args.dry_run)
    else:
        borrar_tags_idioma(args.tags, dry_run=args.dry_run)
