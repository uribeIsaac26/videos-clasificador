import os
import argparse
import pickle
import cv2
import numpy as np
import mysql.connector
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
import gc

# ============================================================
# CONFIGURACIÓN
# ============================================================
load_dotenv()

UMBRAL_DUPLICADO     = 1.0
MIN_TAGS_COMPARTIDOS = 3

raw_path = os.getenv("BASE_PATH")
if not raw_path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, "media")

BASE_PATH    = os.path.normpath(raw_path)
MODEL_CLIP   = "openai/clip-vit-base-patch32"
MEMORIA_FILE = "memoria_multimodal_ia.pkl"
N_FRAMES     = 8

# ============================================================
# INICIALIZACIÓN DE CLIP
# ============================================================
print("Cargando CLIP...")
device         = "cuda" if torch.cuda.is_available() else "cpu"
clip_model     = CLIPModel.from_pretrained(MODEL_CLIP).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP)
clip_model.eval()
print(f"  ✓ CLIP listo en {device.upper()}")

# ============================================================
# EXTRACCIÓN DE EMBEDDINGS
# ============================================================

def extraer_frames_uniformes(ruta_video, n=N_FRAMES):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def obtener_embedding_vision(ruta_video):
    frames = extraer_frames_uniformes(ruta_video)
    if not frames:
        return None

    embeddings = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inp = clip_processor(images=pil, return_tensors="pt").to(device)

        with torch.no_grad():
            out    = clip_model.get_image_features(**inp)
            tensor = out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", out[0])
            emb    = tensor.cpu().detach().numpy().flatten()

        norma = np.linalg.norm(emb)
        embeddings.append(emb / norma if norma > 1e-8 else emb)

    promedio = np.mean(embeddings, axis=0)
    norma    = np.linalg.norm(promedio)
    return promedio / norma if norma > 1e-8 else promedio


def similitud_coseno(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

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


def obtener_videos_por_tag(tag_name):
    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)
    cursor.execute("""
        SELECT v.id, v.video_path
        FROM video v
        JOIN video_tag vt ON v.id = vt.video_id
        JOIN tag t ON vt.tag_id = t.id
        WHERE t.name = %s
    """, (tag_name,))
    videos = cursor.fetchall()
    cursor.close()
    conexion.close()
    return videos


def obtener_tags_de_video(cursor, video_id):
    cursor.execute("""
        SELECT t.name
        FROM video_tag vt
        JOIN tag t ON vt.tag_id = t.id
        WHERE vt.video_id = %s
        ORDER BY t.name
    """, (video_id,))
    return [row["name"] for row in cursor.fetchall()]


def guardar_grupo_duplicados(conexion, tag_origen, miembros):
    """
    miembros = [{"video_id": int, "similitud": float}, ...]
    El primer elemento es el líder (similitud = 1.0).
    """
    cursor = conexion.cursor()
    cursor.execute("""
        INSERT INTO video_duplicate_group (tag_origen, resuelto, date_creation)
        VALUES (%s, %s, %s)
    """, (tag_origen, 0, datetime.now()))
    group_id = cursor.lastrowid

    for m in miembros:
        cursor.execute("""
            INSERT IGNORE INTO video_duplicate_member
                (group_id, video_id, similitud, revisado, accion, date_creation)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (group_id, m["video_id"], m["similitud"], False, "PENDIENTE", datetime.now()))

    conexion.commit()
    cursor.close()
    return group_id

# ============================================================
# LÓGICA DE AGRUPACIÓN (complete linkage)
# ============================================================

def construir_grupos(videos_con_embeddings, umbral, tags_map, min_tags_compartidos):
    """
    Complete linkage: un video entra al grupo solo si, respecto a TODOS los
    miembros actuales, su similitud visual supera el umbral Y comparte al
    menos `min_tags_compartidos` tags. Esto evita grupos transitivos donde
    A≈B y B≈C pero A≠C.
    """
    n          = len(videos_con_embeddings)
    ids        = [v["id"] for v in videos_con_embeddings]
    embeddings = [v["embedding"] for v in videos_con_embeddings]

    # Matriz de similitud y matriz de "cumple criterio completo" (similitud + tags)
    matriz_sim    = np.zeros((n, n))
    matriz_valida = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            sim         = similitud_coseno(embeddings[i], embeddings[j])
            compartidos = len(tags_map[ids[i]] & tags_map[ids[j]])
            valida      = sim >= umbral and compartidos >= min_tags_compartidos

            matriz_sim[i][j] = matriz_sim[j][i] = sim
            matriz_valida[i][j] = matriz_valida[j][i] = valida

    asignado   = [False] * n
    grupos_idx = []

    for i in range(n):
        if asignado[i]:
            continue

        grupo = {i}
        for j in range(n):
            if i == j or asignado[j]:
                continue
            # j entra al grupo solo si cumple el criterio con TODOS los miembros actuales
            if all(matriz_valida[j][k] for k in grupo):
                grupo.add(j)

        if len(grupo) > 1:
            for idx in grupo:
                asignado[idx] = True
            grupos_idx.append(list(grupo))

    # Armar resultado con líder (mayor similitud promedio al resto)
    resultado = []
    for indices in grupos_idx:
        sims_promedio = []
        for i in indices:
            otros = [j for j in indices if j != i]
            prom  = np.mean([matriz_sim[i][j] for j in otros])
            sims_promedio.append(prom)

        idx_lider = indices[np.argmax(sims_promedio)]

        miembros = []
        for i in indices:
            sim_vs_lider = matriz_sim[i][idx_lider] if i != idx_lider else 1.0
            miembros.append({
                "video_id":  ids[i],
                "similitud": round(float(sim_vs_lider), 4),
                "es_lider":  i == idx_lider,
            })

        miembros.sort(key=lambda x: (not x["es_lider"], -x["similitud"]))
        resultado.append(miembros)

    return resultado

# ============================================================
# FLUJO PRINCIPAL
# ============================================================

def detectar_duplicados(tag_name, umbral=UMBRAL_DUPLICADO):
    print(f"\n{'='*60}")
    print(f"  Detector de Duplicados — tag: '{tag_name}'")
    print(f"  Umbral de similitud: {umbral*100:.0f}%")
    print(f"{'='*60}\n")

    # 1. Cargar embeddings del pkl si existe
    memoria_existente = {}
    if os.path.exists(MEMORIA_FILE):
        print(f"✅ Cargando memoria existente: {MEMORIA_FILE}")
        with open(MEMORIA_FILE, "rb") as f:
            memoria = pickle.load(f)
        for m in memoria:
            memoria_existente[m["video_path"]] = m["embedding_v"]
        print(f"   → {len(memoria_existente)} embeddings en memoria\n")
    else:
        print(f"⚠  No se encontró {MEMORIA_FILE}. Se generarán todos los embeddings.\n")

    # 2. Traer todos los videos del tag desde DB
    print(f"🔍 Buscando videos con tag '{tag_name}'...")
    videos = obtener_videos_por_tag(tag_name)
    if not videos:
        print(f"  ❌ No se encontraron videos con el tag '{tag_name}'.")
        return
    print(f"  → {len(videos)} videos encontrados.\n")

    # 3. Agrupar videos por solapamiento de tags (>= MIN_TAGS_COMPARTIDOS)
    print("🏷  Obteniendo perfil de tags por video...")
    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)

    tags_map = {}
    for video in videos:
        tags_map[video["id"]] = frozenset(obtener_tags_de_video(cursor, video["id"]))

    cursor.close()
    conexion.close()

    # Union-Find: agrupa en componentes conexas los videos que comparten al
    # menos MIN_TAGS_COMPARTIDOS tags entre sí. Esto solo decide qué videos
    # vale la pena comparar visualmente; los grupos finales de duplicados
    # los arma construir_grupos con la similitud de embeddings.
    padre = {v["id"]: v["id"] for v in videos}

    def encontrar(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(x, y):
        rx, ry = encontrar(x), encontrar(y)
        if rx != ry:
            padre[rx] = ry

    ids_videos = [v["id"] for v in videos]
    for i in range(len(ids_videos)):
        for j in range(i + 1, len(ids_videos)):
            id_i, id_j = ids_videos[i], ids_videos[j]
            if len(tags_map[id_i] & tags_map[id_j]) >= MIN_TAGS_COMPARTIDOS:
                unir(id_i, id_j)

    componentes = {}
    for video in videos:
        raiz = encontrar(video["id"])
        componentes.setdefault(raiz, []).append(video)

    # Solo interesan componentes con 2+ videos (candidatos reales a duplicado)
    candidatos       = {raiz: vids for raiz, vids in componentes.items() if len(vids) >= 2}
    total_candidatos = sum(len(v) for v in candidatos.values())
    descartados      = len(videos) - total_candidatos

    print(f"  → {len(componentes)} componente(s) de tags")
    print(f"  → {len(candidatos)} componente(s) con 2+ videos ({total_candidatos} videos a comparar)")
    print(f"  → {descartados} video(s) sin afinidad de tags (< {MIN_TAGS_COMPARTIDOS} en común), descartados\n")

    if not candidatos:
        print(f"✅ Ningún par de videos comparte al menos {MIN_TAGS_COMPARTIDOS} tags. No hay duplicados posibles.")
        return

    # 4. Por cada perfil: obtener embeddings y detectar duplicados
    conexion         = conectar_db()
    grupos_guardados = 0

    for videos_grupo in candidatos.values():
        ids_str = ", ".join(str(v["id"]) for v in videos_grupo)
        print(f"─── Componente de videos [{ids_str}]  ({len(videos_grupo)} videos)")

        videos_con_embeddings = []
        for vid in videos_grupo:
            ruta = os.path.join(BASE_PATH, vid["video_path"])
            if not os.path.exists(ruta):
                print(f"    ⚠ Omitido (archivo no encontrado): {vid['video_path']}")
                continue

            if vid["video_path"] in memoria_existente:
                emb = memoria_existente[vid["video_path"]]
                print(f"    ✓ Video {vid['id']} — desde .pkl")
            else:
                print(f"    ⚙ Video {vid['id']} — generando embedding...", end=" ")
                emb = obtener_embedding_vision(ruta)
                if emb is None:
                    print("❌ falló, omitido.")
                    continue
                print("✓")

            videos_con_embeddings.append({
                "id":         vid["id"],
                "video_path": vid["video_path"],
                "embedding":  emb,
            })
            gc.collect()

        if len(videos_con_embeddings) < 2:
            print("    ⚠ Menos de 2 embeddings válidos, se omite.\n")
            continue

        grupos = construir_grupos(videos_con_embeddings, umbral, tags_map, MIN_TAGS_COMPARTIDOS)

        if not grupos:
            print(f"    ✅ Sin duplicados con umbral {umbral*100:.0f}%\n")
            continue

        print(f"    📦 {len(grupos)} grupo(s) encontrado(s)")
        for i, miembros in enumerate(grupos, 1):
            ids_grupo = [m["video_id"] for m in miembros]
            sims      = [m["similitud"] for m in miembros if not m["es_lider"]]
            print(f"       Grupo {i}: IDs {ids_grupo} — "
                  f"similitud {min(sims)*100:.1f}%-{max(sims)*100:.1f}%")

            miembros_db = [{"video_id": m["video_id"], "similitud": m["similitud"]} for m in miembros]
            group_id    = guardar_grupo_duplicados(conexion, tag_name, miembros_db)
            print(f"       ✓ Guardado como grupo #{group_id} en DB")

            grupos_guardados += 1

        print()

    conexion.close()

    print(f"{'='*60}")
    print(f"  ✅ {grupos_guardados} grupo(s) guardados para revisión.")
    print(f"{'='*60}\n")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecta videos duplicados dentro de un tag.")
    parser.add_argument("tag", help="Nombre del tag a analizar (ej: Nature)")
    parser.add_argument(
        "--umbral",
        type=float,
        default=UMBRAL_DUPLICADO,
        help=f"Umbral de similitud visual, entre 0 y 1 (default: {UMBRAL_DUPLICADO})",
    )
    args = parser.parse_args()

    detectar_duplicados(
        tag_name=args.tag,
        umbral=args.umbral,
    )
