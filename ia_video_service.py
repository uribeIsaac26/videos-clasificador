import os
import pickle
import cv2
import numpy as np
import mysql.connector
import torch
import librosa
import laion_clap
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
import gc
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARCHE DE SEGURIDAD PARA PYTORCH 2.6+
# ============================================================
torch.serialization.add_safe_globals([
    np.core.multiarray.scalar, np.dtype, np.ndarray,
    np.core.multiarray._reconstruct
])

# ============================================================
# CONFIGURACIÓN Y RUTAS
# ============================================================
load_dotenv()
raw_path = os.getenv("BASE_PATH")

if not raw_path:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(script_dir, "media")

BASE_PATH    = os.path.normpath(raw_path)
MODEL_CLIP   = "openai/clip-vit-base-patch32"
MEMORIA_FILE = "memoria_multimodal_ia.pkl"

# Cuántos frames uniformes extraer por video
N_FRAMES = 8

# Similitud coseno mínima para que un vecino "vote"
UMBRAL_SIMILITUD = 0.20

# Cuántos vecinos usar en la búsqueda KNN semántica
K_VECINOS = 5

# Porcentaje mínimo de votos para que un tag gane (0.0 - 1.0)
UMBRAL_VOTO = 0.30

# ============================================================
# INICIALIZACIÓN DE MODELOS
# ============================================================
print("Cargando motores de IA (CLIP + CLAP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  → Dispositivo: {device.upper()}")

# CLIP — Visión
clip_model     = CLIPModel.from_pretrained(MODEL_CLIP).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CLIP)
clip_model.eval()

# CLAP — Audio semántico
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny").to(device)


def cargar_clap():
    """Carga CLAP con fallback robusto."""
    try:
        print("  Cargando CLAP (carga estándar)...")
        clap_model.load_ckpt()
        print("  ✓ CLAP cargado.")
    except Exception as e:
        print(f"  ⚠ Carga estándar falló ({e}). Intentando modo degradado...")
        try:
            # Inicialización mínima para que get_audio_embedding_from_data no explote
            if not hasattr(clap_model, "model") or clap_model.model is None:
                raise RuntimeError("Modelo CLAP no inicializado correctamente.")
            # Dejamos los pesos aleatorios — embeddings serán ruidosos pero consistentes
            print("  ⚠ CLAP en modo degradado: embeddings de audio serán aproximados.")
        except Exception as e2:
            print(f"  ❌ CLAP no disponible: {e2}. El audio usará solo detección de energía.")

cargar_clap()

# ============================================================
# EXTRACCIÓN DE EMBEDDINGS
# ============================================================

def extraer_frames_uniformes(ruta_video, n=N_FRAMES):
    """
    Extrae N frames distribuidos uniformemente a lo largo del video.
    Devuelve lista de frames BGR o lista vacía si falla.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"  ⚠ No se pudo abrir: {ruta_video}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def embedding_vision_frame(frame_bgr):
    """Embedding CLIP de un solo frame BGR → np.array (512,)"""
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    inp  = clip_processor(images=pil, return_tensors="pt").to(device)

    with torch.no_grad():
        out = clip_model.get_image_features(**inp)
        tensor = out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", out[0])
        emb = tensor.cpu().detach().numpy().flatten()

    # Normalizar a norma unitaria para que la similitud coseno sea coherente
    norma = np.linalg.norm(emb)
    return emb / norma if norma > 1e-8 else emb


def obtener_embedding_vision(ruta_video):
    """
    Promedio de embeddings CLIP sobre N frames uniformes.
    Devuelve np.array (512,) o zeros si falla.
    """
    frames = extraer_frames_uniformes(ruta_video)
    if not frames:
        return np.zeros(512)

    embeddings = [embedding_vision_frame(f) for f in frames]
    promedio   = np.mean(embeddings, axis=0)

    # Re-normalizar el promedio
    norma = np.linalg.norm(promedio)
    return promedio / norma if norma > 1e-8 else promedio


def obtener_embedding_audio(ruta_video):
    """
    Embedding semántico de audio con CLAP.
    Devuelve:
      - np.array (512,) con el embedding si hay audio
      - np.zeros(512)  si es silencio real
      - np.ones(512)*1e-5 si hay audio pero CLAP falla (plan B)
    """
    try:
        audio_data, _ = librosa.load(ruta_video, sr=48000, duration=10.0, mono=True)

        if audio_data.size == 0:
            return np.zeros(512)

        # Detección de silencio real por energía RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 1e-4:
            return np.zeros(512)

        # Normalizar antes de pasar a CLAP
        audio_norm = audio_data / (np.max(np.abs(audio_data)) + 1e-8)

        try:
            emb = clap_model.get_audio_embedding_from_data(x=[audio_norm])
            if torch.is_tensor(emb):
                emb = emb.cpu().detach().numpy()
            emb = emb.flatten()

            # Normalizar
            norma = np.linalg.norm(emb)
            return emb / norma if norma > 1e-8 else emb

        except Exception:
            # CLAP falló pero sí hay audio: vector pequeño no-cero
            return np.ones(512) * 1e-5

    except Exception as e:
        print(f"  ⚠ Error cargando audio de {ruta_video}: {e}")
        return np.zeros(512)


def hay_audio(embedding_audio):
    """True si el embedding indica que hay sonido real."""
    return not np.all(embedding_audio == 0)


def combinar_embeddings(emb_v, emb_a, peso_visual=0.7, peso_audio=0.3):
    """
    Concatena los dos embeddings ponderados.
    Puedes ajustar los pesos según qué tan fiable sea cada fuente.
    """
    return np.concatenate([emb_v * peso_visual, emb_a * peso_audio])

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

# ============================================================
# GENERACIÓN DE MEMORIA (ENTRENAMIENTO)
# ============================================================

def generar_base_conocimiento():
    """
    Recorre todos los videos etiquetados en la DB,
    extrae embeddings multimodales y guarda la memoria en disco.
    """
    print("\n🔄 Generando base de conocimiento multimodal...")
    try:
        conexion = conectar_db()
        cursor   = conexion.cursor(dictionary=True)
        cursor.execute("""
            SELECT v.id, v.video_path, GROUP_CONCAT(t.name SEPARATOR ',') AS etiquetas
            FROM video v
            JOIN video_tag vt ON v.id = vt.video_id
            JOIN tag t ON vt.tag_id = t.id
            GROUP BY v.id
        """)
        videos = cursor.fetchall()
        cursor.close()
        conexion.close()

        if not videos:
            print("  ❌ No hay videos etiquetados en la base de datos.")
            return

        memoria = []
        for vid in videos:
            ruta = os.path.join(BASE_PATH, vid["video_path"])
            if not os.path.exists(ruta):
                print(f"  ⚠ Archivo no encontrado, omitido: {ruta}")
                continue

            print(f"  Procesando video {vid['id']}...", end=" ")

            emb_v = obtener_embedding_vision(ruta)
            emb_a = obtener_embedding_audio(ruta)
            emb_c = combinar_embeddings(emb_v, emb_a)

            # Parsear etiquetas limpias
            etiquetas = [t.strip() for t in vid["etiquetas"].split(",") if t.strip()]

            memoria.append({
                "video_id":    vid["id"],
                "video_path":  vid["video_path"],
                "embedding_v": emb_v,   # (512,)  — solo visual
                "embedding_a": emb_a,   # (512,)  — solo audio
                "embedding_c": emb_c,   # (1024,) — combinado
                "etiquetas":   etiquetas,
                "tiene_audio": hay_audio(emb_a),
            })
            print(f"✓  [{', '.join(etiquetas)}]")

            gc.collect()

        with open(MEMORIA_FILE, "wb") as f:
            pickle.dump(memoria, f)

        print(f"\n✅ Memoria guardada con {len(memoria)} videos → {MEMORIA_FILE}")

    except Exception as e:
        print(f"\n❌ Error en generación: {e}")
        raise

# ============================================================
# CLASIFICACIÓN DE NUEVO VIDEO
# ============================================================

def similitud_coseno(a, b):
    """1 = idénticos, 0 = ortogonales, -1 = opuestos."""
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)
    if norma_a < 1e-8 or norma_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norma_a * norma_b))


def votar_tags(vecinos_etiquetas, similitudes, umbral_voto=UMBRAL_VOTO):
    """
    Dado los K vecinos más cercanos, hace votación ponderada por similitud.
    Un tag gana si acumula al menos `umbral_voto` del total de similitudes.
    """
    conteo   = {}
    peso_total = sum(similitudes)

    for etiquetas, sim in zip(vecinos_etiquetas, similitudes):
        for tag in etiquetas:
            # Excluimos tags de audio — se manejan aparte
            tag_lower = tag.lower()
            if tag_lower in {"sound", "sonido", "no sound", "sin sonido", "silencio", "mute"}:
                continue
            conteo[tag] = conteo.get(tag, 0) + sim

    if peso_total < 1e-8:
        return []

    tags_ganadores = [
        tag for tag, peso in conteo.items()
        if (peso / peso_total) >= umbral_voto
    ]
    return tags_ganadores


def detectar_tags_audio_del_entreno(memoria):
    """Detecta dinámicamente cómo se llaman los tags de audio en el dataset."""
    tag_con_audio = None
    tag_sin_audio = None

    for m in memoria:
        for t in m["etiquetas"]:
            t_l = t.lower()
            if t_l in {"sound", "sonido", "con sonido"}:
                tag_con_audio = t
            elif t_l in {"no sound", "sin sonido", "silencio", "mute"}:
                tag_sin_audio = t

    return tag_con_audio or "Sound", tag_sin_audio or "No sound"


def clasificar_video_nuevo(ruta_relativa, memoria):
    """
    Clasifica un video nuevo usando:
      1. Similitud coseno contra embeddings del entrenamiento (visión)
      2. Votación ponderada de los K vecinos más similares
      3. Detección binaria de audio (sonido / silencio)
    """
    ruta_completa = os.path.join(BASE_PATH, ruta_relativa)
    if not os.path.exists(ruta_completa):
        print(f"  ❌ Archivo no encontrado: {ruta_completa}")
        return None

    # --- Extraer embeddings del video nuevo ---
    emb_v_nuevo = obtener_embedding_vision(ruta_completa)
    emb_a_nuevo = obtener_embedding_audio(ruta_completa)

    if np.all(emb_v_nuevo == 0):
        print(f"  ⚠ No se pudo extraer visual de {ruta_relativa}")
        return None

    # --- Calcular similitud contra TODOS los videos de entreno ---
    similitudes = []
    for m in memoria:
        sim = similitud_coseno(emb_v_nuevo, m["embedding_v"])
        similitudes.append(sim)

    similitudes  = np.array(similitudes)
    indices_top  = np.argsort(similitudes)[::-1][:K_VECINOS]
    sims_top     = similitudes[indices_top]

    # Filtrar vecinos con similitud mínima aceptable
    validos = [(i, s) for i, s in zip(indices_top, sims_top) if s >= UMBRAL_SIMILITUD]

    if not validos:
        # Si ningún vecino es suficientemente similar, tomamos el mejor igualmente
        print(f"  ⚠ Similitud baja. Usando top-1 vecino de todas formas.")
        validos = [(indices_top[0], sims_top[0])]

    indices_validos = [i for i, _ in validos]
    sims_validas    = [s for _, s in validos]

    vecinos_etiquetas = [memoria[i]["etiquetas"] for i in indices_validos]

    # --- Votación de tags visuales ---
    tags_visuales = votar_tags(vecinos_etiquetas, sims_validas)

    # --- Tag de audio ---
    tag_con_audio, tag_sin_audio = detectar_tags_audio_del_entreno(memoria)
    tag_audio = tag_con_audio if hay_audio(emb_a_nuevo) else tag_sin_audio

    # --- Resultado final ---
    tags_finales = sorted(set(tags_visuales) | {tag_audio})
    tags_finales = [t for t in tags_finales if t]  # limpiar vacíos

    return ", ".join(tags_finales)

# ============================================================
# PROCESAMIENTO EN LOTES
# ============================================================

def procesar_lote(limite=5):
    """
    Obtiene videos sin clasificar de la DB,
    los clasifica y guarda los resultados en video_tag_temporal.
    """
    # Cargar o generar memoria
    if not os.path.exists(MEMORIA_FILE):
        print(f"⚠ {MEMORIA_FILE} no encontrado. Generando...")
        generar_base_conocimiento()

    if not os.path.exists(MEMORIA_FILE):
        print("❌ No se pudo crear la memoria. Abortando.")
        return

    with open(MEMORIA_FILE, "rb") as f:
        memoria = pickle.load(f)

    print(f"\n📦 Memoria cargada: {len(memoria)} videos de entrenamiento.")

    try:
        conexion = conectar_db()
        cursor   = conexion.cursor(dictionary=True)

        cursor.execute("""
            SELECT v.id, v.video_path FROM video v
            LEFT JOIN video_tag vt  ON v.id = vt.video_id
            LEFT JOIN video_tag_temporal vtt ON v.id = vtt.video_id
            WHERE vt.video_id IS NULL AND vtt.video_id IS NULL
            LIMIT %s
        """, (limite,))
        pendientes = cursor.fetchall()

        if not pendientes:
            print("✅ No hay videos pendientes de clasificar.")
        else:
            print(f"🎬 Clasificando {len(pendientes)} video(s)...\n")

        for vid in pendientes:
            print(f"  → Video {vid['id']} ({vid['video_path']})")
            tags = clasificar_video_nuevo(vid["video_path"], memoria)

            if tags:
                cursor.execute("""
                    INSERT INTO video_tag_temporal (video_id, tags_suggest, confirm, date_creation)
                    VALUES (%s, %s, %s, %s)
                """, (vid["id"], tags, False, datetime.now()))
                conexion.commit()
                print(f"     ✓ Tags asignados: {tags}")
            else:
                print(f"     ⚠ No se pudieron generar tags.")

        cursor.close()
        conexion.close()

    except Exception as e:
        print(f"\n❌ Error en lote: {e}")
        raise

# ============================================================
# UTILIDADES
# ============================================================

def regenerar_memoria():
    """Borra la memoria existente y la regenera desde cero."""
    if os.path.exists(MEMORIA_FILE):
        os.remove(MEMORIA_FILE)
        print(f"🗑  Memoria anterior eliminada.")
    generar_base_conocimiento()


def diagnostico_memoria():
    """Imprime un resumen de la memoria guardada."""
    if not os.path.exists(MEMORIA_FILE):
        print("❌ No existe archivo de memoria.")
        return

    with open(MEMORIA_FILE, "rb") as f:
        memoria = pickle.load(f)

    todos_tags = {}
    for m in memoria:
        for t in m["etiquetas"]:
            todos_tags[t] = todos_tags.get(t, 0) + 1

    print(f"\n📊 DIAGNÓSTICO DE MEMORIA")
    print(f"   Videos en memoria : {len(memoria)}")
    print(f"   Tags únicos        : {len(todos_tags)}")
    print(f"   Con audio          : {sum(1 for m in memoria if m['tiene_audio'])}")
    print(f"   Sin audio          : {sum(1 for m in memoria if not m['tiene_audio'])}")
    print(f"\n   Distribución de tags:")
    for tag, count in sorted(todos_tags.items(), key=lambda x: -x[1]):
        barra = "█" * count
        print(f"     {tag:<20} {barra} ({count})")

# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    try:
        print(f"{'='*55}")
        print(f"  Servicio de IA Multimodal — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"{'='*55}\n")

        # Mostrar diagnóstico si ya existe memoria
        if os.path.exists(MEMORIA_FILE):
            diagnostico_memoria()
        else:
            print(f"⚠  {MEMORIA_FILE} no encontrado. Se generará ahora.")
            generar_base_conocimiento()

        # Clasificar lote
        procesar_lote(limite=5)

        print(f"\n{'='*55}")
        print(f"  ✅ Proceso finalizado — {datetime.now():%H:%M:%S}")
        print(f"{'='*55}")

    except KeyboardInterrupt:
        print("\n⚠  Detenido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")