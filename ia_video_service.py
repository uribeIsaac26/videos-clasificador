import os
import pickle
import cv2
import av
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
    Intenta primero con OpenCV; si no obtiene frames (p.ej. videos AV1 sin
    aceleración por hardware), reintenta con PyAV, que decodifica AV1 por software.
    Devuelve lista de frames BGR o lista vacía si falla.
    """
    frames = _extraer_frames_opencv(ruta_video, n)
    if frames:
        return frames

    print(f"  ↻ OpenCV no pudo leer frames, reintentando con PyAV: {ruta_video}")
    return _extraer_frames_pyav(ruta_video, n)


def _extraer_frames_opencv(ruta_video, n):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
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


def _extraer_frames_pyav(ruta_video, n):
    """
    Extrae N frames uniformes con PyAV buscando por tiempo (no por índice).
    Para cada instante objetivo hace seek al keyframe anterior y decodifica
    hasta alcanzarlo.
    """
    try:
        with av.open(ruta_video) as container:
            if not container.streams.video:
                print(f"  ⚠ Sin stream de video: {ruta_video}")
                return []

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            if stream.duration is not None:
                duracion = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duracion = container.duration / av.time_base
            else:
                print(f"  ⚠ Duración desconocida: {ruta_video}")
                return []

            tiempos = np.linspace(0, duracion, n, endpoint=False)
            frames = []
            for t in tiempos:
                container.seek(int(t / stream.time_base), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.time is None or frame.time >= t:
                        frames.append(frame.to_ndarray(format="bgr24"))
                        break
            return frames

    except av.error.FFmpegError as e:
        print(f"  ⚠ PyAV no pudo decodificar {ruta_video}: {e}")
        return []


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

# Cada cuántos videos nuevos se guarda la memoria en disco durante la sincronización,
# para no perder el progreso si el proceso se interrumpe a mitad de una carga grande.
GUARDAR_CADA = 100


def obtener_videos_etiquetados():
    """Devuelve {video_id: {"video_path", "etiquetas"}} con todos los videos etiquetados en la DB."""
    conexion = conectar_db()
    cursor   = conexion.cursor(dictionary=True)
    cursor.execute("""
        SELECT v.id, v.video_path, GROUP_CONCAT(t.name SEPARATOR ',') AS etiquetas
        FROM video v
        JOIN video_tag vt ON v.id = vt.video_id
        JOIN tag t ON vt.tag_id = t.id
        GROUP BY v.id
    """)
    filas = cursor.fetchall()
    cursor.close()
    conexion.close()

    return {
        fila["id"]: {
            "video_path": fila["video_path"],
            "etiquetas":  [t.strip() for t in fila["etiquetas"].split(",") if t.strip()],
        }
        for fila in filas
    }


def cargar_memoria():
    """Lee la memoria de disco; lista vacía si no existe."""
    if not os.path.exists(MEMORIA_FILE):
        return []
    with open(MEMORIA_FILE, "rb") as f:
        return pickle.load(f)


def guardar_memoria(memoria):
    """
    Escribe la memoria de forma atómica (archivo temporal + rename), así un corte
    a mitad de escritura no deja el pickle corrupto ni lo ve a medias detectar_duplicados.
    """
    tmp = MEMORIA_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(memoria, f)
    os.replace(tmp, MEMORIA_FILE)


def calcular_entrada_memoria(video_id, video_path, etiquetas):
    """Extrae embeddings de un video. Devuelve el registro de memoria o None si no se pudo."""
    ruta = os.path.join(BASE_PATH, video_path)
    if not os.path.exists(ruta):
        print(f"  ⚠ Archivo no encontrado, omitido: {ruta}")
        return None

    emb_v = obtener_embedding_vision(ruta)
    if np.all(emb_v == 0):
        # No se guarda: un embedding vacío no aporta al KNN y así se reintenta en la próxima sincronización
        print(f"  ⚠ No se pudo extraer visual del video {video_id}, omitido.")
        return None

    emb_a = obtener_embedding_audio(ruta)
    return {
        "video_id":    video_id,
        "video_path":  video_path,
        "embedding_v": emb_v,                            # (512,)  — solo visual
        "embedding_a": emb_a,                            # (512,)  — solo audio
        "embedding_c": combinar_embeddings(emb_v, emb_a),  # (1024,) — combinado
        "etiquetas":   etiquetas,
        "tiene_audio": hay_audio(emb_a),
    }


def sincronizar_memoria():
    """
    Pone la memoria al día con la DB de forma incremental:
      - Videos etiquetados que no están en memoria → calcula sus embeddings.
      - Videos ya en memoria → refresca sus etiquetas (sin recalcular embeddings);
        si cambió su video_path, recalcula.
      - Videos que ya no están etiquetados en la DB → los quita.
    Devuelve la memoria sincronizada.
    """
    print("\n🔄 Sincronizando memoria con la base de datos...")
    en_db   = obtener_videos_etiquetados()
    memoria = cargar_memoria()

    actualizados = 0
    eliminados   = 0
    vigentes     = []
    for m in memoria:
        info = en_db.get(m["video_id"])
        if info is None or info["video_path"] != m["video_path"]:
            eliminados += 1  # si solo cambió la ruta, se vuelve a calcular abajo como nuevo
            continue
        if m["etiquetas"] != info["etiquetas"]:
            m["etiquetas"] = info["etiquetas"]
            actualizados += 1
        vigentes.append(m)

    ids_en_memoria = {m["video_id"] for m in vigentes}
    nuevos = [vid_id for vid_id in en_db if vid_id not in ids_en_memoria]

    print(f"  → En DB: {len(en_db)} | En memoria: {len(vigentes)} | "
          f"Nuevos: {len(nuevos)} | Tags actualizados: {actualizados} | Eliminados: {eliminados}")

    agregados = 0
    for n, vid_id in enumerate(nuevos, start=1):
        info = en_db[vid_id]
        print(f"  [{n}/{len(nuevos)}] Procesando video {vid_id}...", end=" ")
        entrada = calcular_entrada_memoria(vid_id, info["video_path"], info["etiquetas"])
        if entrada is None:
            continue
        vigentes.append(entrada)
        agregados += 1
        print(f"✓  [{', '.join(info['etiquetas'])}]")

        if agregados % GUARDAR_CADA == 0:
            guardar_memoria(vigentes)
            print(f"  💾 Progreso guardado ({len(vigentes)} videos en memoria).")
        gc.collect()

    if agregados or actualizados or eliminados or not os.path.exists(MEMORIA_FILE):
        guardar_memoria(vigentes)
        print(f"✅ Memoria actualizada: {len(vigentes)} videos → {MEMORIA_FILE}")
    else:
        print("✅ Memoria ya estaba al día.")

    return vigentes


def generar_base_conocimiento():
    """Se mantiene por compatibilidad: ahora solo sincroniza de forma incremental."""
    return sincronizar_memoria()

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
    # Antes de clasificar, incorporar los videos etiquetados/corregidos desde la última ejecución
    memoria = sincronizar_memoria()

    if not memoria:
        print("❌ La memoria está vacía. Abortando.")
        return

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
    """Borra la memoria existente y la regenera desde cero (solo necesario si cambia el modelo o la extracción)."""
    if os.path.exists(MEMORIA_FILE):
        os.remove(MEMORIA_FILE)
        print(f"🗑  Memoria anterior eliminada.")
    sincronizar_memoria()


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

        # Mostrar diagnóstico de la memoria actual (si existe)
        if os.path.exists(MEMORIA_FILE):
            diagnostico_memoria()

        # Clasificar lote (sincroniza la memoria incrementalmente antes de clasificar)
        procesar_lote(limite=5)

        print(f"\n{'='*55}")
        print(f"  ✅ Proceso finalizado — {datetime.now():%H:%M:%S}")
        print(f"{'='*55}")

    except KeyboardInterrupt:
        print("\n⚠  Detenido por el usuario.")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")