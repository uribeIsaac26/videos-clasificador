import os
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

# --- TAG A ANALIZAR (cambiar para pruebas) ---
TAG_BUSQUEDA = "Nature"

# --- Umbral de similitud para considerar duplicado (0.0 - 1.0) ---
UMBRAL_DUPLICADO = 0.85

# --- Rutas y modelos ---
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
    """Embedding CLIP promediado sobre N frames. Devuelve np.array (512,)"""
    frames = extraer_frames_uniformes(ruta_video)
    if not frames:
        return None

    embeddings = []
    for frame in frames:
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        inp  = clip_processor(images=pil, return_tensors="pt").to(device)

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
    """Trae todos los videos que tienen el tag indicado."""
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


def guardar_grupo_duplicados(conexion, tag_origen, miembros):
    """
    Guarda un grupo de duplicados en la DB.
    miembros = [{"video_id": int, "similitud": float}, ...]
    """
    cursor = conexion.cursor()

    # 1. Insertar el grupo
    cursor.execute("""
        INSERT INTO video_duplicate_group (tag_origen, date_creation)
        VALUES (%s, %s)
    """, (tag_origen, datetime.now()))
    group_id = cursor.lastrowid

    # 2. Insertar cada miembro del grupo
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
# LÓGICA DE AGRUPACIÓN
# ============================================================

def construir_grupos(videos_con_embeddings, umbral):
    """
    Algoritmo de agrupación por similitud (Union-Find simplificado):
    - Compara cada par de videos
    - Si su similitud >= umbral, los une en el mismo grupo
    - Devuelve lista de grupos, cada grupo es lista de {video_id, similitud}

    La similitud de cada miembro se calcula vs el video con
    mayor similitud promedio dentro del grupo (el "líder").
    """
    n       = len(videos_con_embeddings)
    ids     = [v["id"] for v in videos_con_embeddings]
    embeddings = [v["embedding"] for v in videos_con_embeddings]

    # Matriz de similitud completa
    print(f"  Calculando matriz de similitud ({n}x{n})...")
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = similitud_coseno(embeddings[i], embeddings[j])
            matriz[i][j] = sim
            matriz[j][i] = sim

    # Union-Find para agrupar
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if matriz[i][j] >= umbral:
                union(i, j)

    # Agrupar índices por raíz
    grupos_idx = {}
    for i in range(n):
        raiz = find(i)
        grupos_idx.setdefault(raiz, []).append(i)

    # Solo nos interesan grupos con más de 1 video
    grupos_duplicados = []
    for raiz, indices in grupos_idx.items():
        if len(indices) < 2:
            continue

        # El "líder" es el video con mayor similitud promedio al resto del grupo
        sims_promedio = []
        for i in indices:
            otros = [j for j in indices if j != i]
            prom  = np.mean([matriz[i][j] for j in otros])
            sims_promedio.append(prom)

        idx_lider = indices[np.argmax(sims_promedio)]

        # Construir miembros con similitud vs el líder
        miembros = []
        for i in indices:
            sim_vs_lider = matriz[i][idx_lider] if i != idx_lider else 1.0
            miembros.append({
                "video_id":  ids[i],
                "similitud": round(float(sim_vs_lider), 4),
                "es_lider":  i == idx_lider
            })

        # Ordenar: líder primero, luego por similitud descendente
        miembros.sort(key=lambda x: (not x["es_lider"], -x["similitud"]))
        grupos_duplicados.append(miembros)

    return grupos_duplicados

# ============================================================
# FLUJO PRINCIPAL
# ============================================================

def detectar_duplicados(tag_name, umbral=UMBRAL_DUPLICADO):
    print(f"\n{'='*55}")
    print(f"  Detector de Duplicados — tag: '{tag_name}'")
    print(f"  Umbral de similitud: {umbral*100:.0f}%")
    print(f"{'='*55}\n")

    # 1. Cargar memoria existente del pkl
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

    # 2. Traer videos del tag desde la DB
    print(f"🔍 Buscando videos con tag '{tag_name}'...")
    videos = obtener_videos_por_tag(tag_name)

    if not videos:
        print(f"  ❌ No se encontraron videos con el tag '{tag_name}'.")
        return

    print(f"  → {len(videos)} videos encontrados.\n")

    # 3. Obtener embedding de cada video (pkl o generado)
    videos_con_embeddings = []
    for vid in videos:
        ruta_completa = os.path.join(BASE_PATH, vid["video_path"])

        if not os.path.exists(ruta_completa):
            print(f"  ⚠ Archivo no encontrado, omitido: {vid['video_path']}")
            continue

        # ¿Ya está en la memoria?
        if vid["video_path"] in memoria_existente:
            emb = memoria_existente[vid["video_path"]]
            print(f"  ✓ Video {vid['id']} — embedding desde .pkl")
        else:
            print(f"  ⚙ Video {vid['id']} — generando embedding...", end=" ")
            emb = obtener_embedding_vision(ruta_completa)
            if emb is None:
                print("❌ falló, omitido.")
                continue
            print("✓")

        videos_con_embeddings.append({
            "id":        vid["id"],
            "video_path": vid["video_path"],
            "embedding": emb
        })
        gc.collect()

    if len(videos_con_embeddings) < 2:
        print("\n⚠  Necesitas al menos 2 videos con embeddings válidos para comparar.")
        return

    # 4. Construir grupos de duplicados
    print(f"\n🔗 Buscando grupos de duplicados (umbral {umbral*100:.0f}%)...")
    grupos = construir_grupos(videos_con_embeddings, umbral)

    if not grupos:
        print(f"\n✅ No se encontraron duplicados para el tag '{tag_name}' con umbral {umbral*100:.0f}%.")
        return

    print(f"\n📦 {len(grupos)} grupo(s) de duplicados encontrados.\n")

    # 5. Guardar en la DB
    conexion = conectar_db()
    grupos_guardados = 0

    for i, miembros in enumerate(grupos, 1):
        ids_grupo = [m["video_id"] for m in miembros]
        sims      = [m["similitud"] for m in miembros if not m["es_lider"]]
        sim_max   = max(sims) if sims else 1.0
        sim_min   = min(sims) if sims else 1.0

        print(f"  Grupo {i}: {len(miembros)} videos — "
              f"similitud {sim_min*100:.1f}% a {sim_max*100:.1f}%")
        print(f"    IDs: {ids_grupo}")

        # Guardar en DB (sin el campo es_lider, eso es solo interno)
        miembros_db = [{"video_id": m["video_id"], "similitud": m["similitud"]} for m in miembros]
        group_id    = guardar_grupo_duplicados(conexion, tag_name, miembros_db)
        print(f"    ✓ Guardado como grupo #{group_id} en DB\n")
        grupos_guardados += 1

    conexion.close()

    print(f"{'='*55}")
    print(f"  ✅ {grupos_guardados} grupo(s) guardados para revisión.")
    print(f"{'='*55}\n")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    detectar_duplicados(
        tag_name=TAG_BUSQUEDA,
        umbral=UMBRAL_DUPLICADO
    )