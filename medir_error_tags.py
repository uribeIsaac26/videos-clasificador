import os
import csv
import argparse
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

# ============================================================
# CONFIGURACIÓN
# ============================================================
load_dotenv()

# Tags con menos videos reales que esto se marcan como "pocos datos" en el reporte
MIN_SOPORTE_CONFIABLE = 10

# Cuántas confusiones (tag sugerido de más ↔ tag que faltó) mostrar
TOP_CONFUSIONES = 15

# ============================================================
# BASE DE DATOS
# ============================================================

def conectar_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE")
    )


def obtener_pares_sugerido_real(desde=None):
    """
    Devuelve [(video_id, tags_sugeridos, tags_reales)] para cada sugerencia de la IA
    que ya fue confirmada/corregida. Los tags reales salen de video_tag (lo que quedó
    guardado tras la revisión manual).
    """
    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)

    filtro_fecha = "AND vtt.date_creation >= %s" if desde else ""
    cursor.execute(f"""
        SELECT vtt.video_id, vtt.tags_suggest,
               GROUP_CONCAT(t.name SEPARATOR ',') AS tags_reales
        FROM video_tag_temporal vtt
        JOIN video_tag vt ON vt.video_id = vtt.video_id
        JOIN tag t ON t.id = vt.tag_id
        WHERE vtt.confirm = TRUE {filtro_fecha}
        GROUP BY vtt.video_id, vtt.tags_suggest
    """, (desde,) if desde else ())
    filas = cursor.fetchall()
    cursor.close()
    conexion.close()

    return [
        (f["video_id"], parsear_tags(f["tags_suggest"], ","), parsear_tags(f["tags_reales"], ","))
        for f in filas
    ]


def parsear_tags(texto, separador):
    return {t.strip() for t in (texto or "").split(separador) if t.strip()}

# ============================================================
# MÉTRICAS
# ============================================================

def _ratio(a, b):
    return a / b if b else None


def calcular_metricas(pares):
    """
    Calcula métricas por tag y globales a partir de [(video_id, sugeridos, reales)].
    La comparación de nombres ignora mayúsculas; se reporta el nombre tal como está en video_tag.

    Por tag:
      - acertados (TP):  sugerido y se mantuvo
      - sobrantes (FP):  sugerido y se quitó al revisar
      - faltantes (FN):  no sugerido y se agregó al revisar
      - precisión = TP / (TP + FP) → de las veces que lo sugiere, cuántas acierta
      - recall    = TP / (TP + FN) → de los videos que lo tienen, a cuántos se lo sugiere
    """
    por_tag     = {}
    nombres     = {}
    confusiones = {}
    exactos     = 0

    def stats(clave):
        return por_tag.setdefault(clave, {"tp": 0, "fp": 0, "fn": 0})

    for _, sugeridos, reales in pares:
        sug  = {t.lower(): t for t in sugeridos}
        real = {t.lower(): t for t in reales}
        for clave, nombre in sug.items():
            nombres.setdefault(clave, nombre)
        nombres.update(real)  # el nombre de video_tag siempre gana sobre el de la sugerencia

        sobrantes = sug.keys() - real.keys()
        faltantes = real.keys() - sug.keys()
        for clave in sug.keys() & real.keys():
            stats(clave)["tp"] += 1
        for clave in sobrantes:
            stats(clave)["fp"] += 1
        for clave in faltantes:
            stats(clave)["fn"] += 1

        if not sobrantes and not faltantes:
            exactos += 1

        # En un mismo video, un tag sugerido de más junto a uno que faltó suele indicar
        # que el modelo los está confundiendo.
        for s in sobrantes:
            for f in faltantes:
                confusiones[(s, f)] = confusiones.get((s, f), 0) + 1

    filas = []
    for clave, c in por_tag.items():
        precision = _ratio(c["tp"], c["tp"] + c["fp"])
        recall    = _ratio(c["tp"], c["tp"] + c["fn"])
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else 0.0)
        filas.append({
            "tag":       nombres[clave],
            "soporte":   c["tp"] + c["fn"],  # videos que realmente tienen el tag
            "sugerido":  c["tp"] + c["fp"],
            "acertados": c["tp"],
            "sobrantes": c["fp"],
            "faltantes": c["fn"],
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })
    filas.sort(key=lambda f: (f["f1"], -f["soporte"]))

    tp = sum(c["tp"] for c in por_tag.values())
    fp = sum(c["fp"] for c in por_tag.values())
    fn = sum(c["fn"] for c in por_tag.values())
    p_micro = _ratio(tp, tp + fp) or 0.0
    r_micro = _ratio(tp, tp + fn) or 0.0
    con_soporte = [f for f in filas if f["soporte"] > 0]
    globales = {
        "videos":          len(pares),
        "exactos":         exactos,
        "precision_micro": p_micro,
        "recall_micro":    r_micro,
        "f1_micro":        2 * p_micro * r_micro / (p_micro + r_micro) if p_micro + r_micro else 0.0,
        "f1_macro":        (sum(f["f1"] for f in con_soporte) / len(con_soporte)) if con_soporte else 0.0,
    }

    top_conf = sorted(
        ((nombres[s], nombres[f], n) for (s, f), n in confusiones.items()),
        key=lambda x: -x[2]
    )[:TOP_CONFUSIONES]

    return filas, globales, top_conf

# ============================================================
# REPORTE
# ============================================================

def _pct(valor):
    return "  -  " if valor is None else f"{valor * 100:5.1f}%"


def imprimir_reporte(filas, globales, confusiones, min_soporte):
    print(f"\n{'='*86}")
    print(f"  Error de clasificación por tag — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'='*86}\n")

    if not globales["videos"]:
        print("  ⚠ No hay sugerencias confirmadas para comparar.")
        return

    g = globales
    print(f"  Videos revisados      : {g['videos']}")
    print(f"  Sugerencia exacta     : {g['exactos']} ({g['exactos'] / g['videos'] * 100:.1f}%)")
    print(f"  Precisión global      : {_pct(g['precision_micro'])}   (de los tags sugeridos, cuántos se mantuvieron)")
    print(f"  Recall global         : {_pct(g['recall_micro'])}   (de los tags reales, cuántos sugirió)")
    print(f"  F1 global (micro)     : {_pct(g['f1_micro'])}   (dominado por los tags frecuentes)")
    print(f"  F1 promedio por tag   : {_pct(g['f1_macro'])}   (todos los tags pesan igual)")

    visibles = [f for f in filas if f["soporte"] >= min_soporte or f["sugerido"] >= min_soporte]
    print(f"\n  Tags ordenados de peor a mejor F1 (mostrando {len(visibles)} de {len(filas)}; "
          f"filtro --min-soporte {min_soporte})\n")
    print(f"  {'Tag':<24} {'Reales':>6} {'Sugerido':>8} {'Acierto':>7} {'Sobra':>6} {'Falta':>6} "
          f"{'Precis.':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'-'*24} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for f in visibles:
        aviso = "  ⚠ pocos datos" if f["soporte"] < MIN_SOPORTE_CONFIABLE else ""
        print(f"  {f['tag'][:24]:<24} {f['soporte']:>6} {f['sugerido']:>8} {f['acertados']:>7} "
              f"{f['sobrantes']:>6} {f['faltantes']:>6} {_pct(f['precision']):>7} "
              f"{_pct(f['recall']):>7} {_pct(f['f1']):>7}{aviso}")

    if confusiones:
        print(f"\n  Confusiones más frecuentes (sugirió el primero, pero el real era el segundo):\n")
        for sobrante, faltante, n in confusiones:
            print(f"    {n:>4}×  {sobrante}  →  {faltante}")

    print(f"\n  Cómo leerlo:")
    print(f"    · Precisión baja → sugiere el tag donde no va (sobra).")
    print(f"    · Recall bajo    → no lo sugiere donde sí va (falta). Típico de tags poco frecuentes.\n")


def exportar_csv(filas, ruta):
    campos = ["tag", "soporte", "sugerido", "acertados", "sobrantes", "faltantes",
              "precision", "recall", "f1"]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in filas:
            writer.writerow({k: ("" if fila[k] is None else
                                 round(fila[k], 4) if isinstance(fila[k], float) else fila[k])
                             for k in campos})
    print(f"  💾 CSV exportado → {ruta}")

# ============================================================
# PUNTO DE ENTRADA
# ============================================================

def medir_error_tags(desde=None, min_soporte=1, csv_path=None):
    pares = obtener_pares_sugerido_real(desde)
    filas, globales, confusiones = calcular_metricas(pares)
    imprimir_reporte(filas, globales, confusiones, min_soporte)
    if csv_path:
        exportar_csv(filas, csv_path)
    return filas, globales


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara las sugerencias de la IA (video_tag_temporal) contra los tags finales (video_tag)."
    )
    parser.add_argument("--desde", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                        help="Solo sugerencias creadas desde esta fecha (YYYY-MM-DD). "
                             "Útil para medir el efecto de un cambio en el clasificador.")
    parser.add_argument("--min-soporte", type=int, default=1,
                        help="Ocultar tags con menos de N videos reales y N sugerencias (default: 1)")
    parser.add_argument("--csv", help="Ruta para exportar las métricas por tag en CSV")
    args = parser.parse_args()

    medir_error_tags(desde=args.desde, min_soporte=args.min_soporte, csv_path=args.csv)
