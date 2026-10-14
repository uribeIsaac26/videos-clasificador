"""
Detección de idioma hablado y de música en un video.

  - Música: AST (clasificador de sonidos entrenado en AudioSet) decide si la música es lo
    dominante. Alguien hablando con música de fondo NO cuenta como música.
  - Idioma: faster-whisper sobre los tramos con voz (VAD Silero), solo si AST también
    confirma habla o canto (evita que Whisper "invente" idioma con ruido o instrumentales).

Tags resultantes por video:
  | Audio                                   | Tags                 |
  | Ruido / sonido de fondo                 | (ninguno)            |
  | Canción instrumental                    | Music                |
  | Canción con letra                       | Music + idioma letra |
  | Persona hablando (con o sin música)     | idioma               |
"""
import numpy as np
import librosa
import torch
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# ============================================================
# CONFIGURACIÓN
# ============================================================

# tiny/base/small/medium: más grande = más preciso y más lento. "small" es buen balance en CPU.
MODELO_WHISPER = "small"

SAMPLE_RATE = 16000

# Ventanas que se analizan: inicio y puntos relativos del video, para saltar intros con música
POSICIONES_VENTANA = (0.0, 0.4, 0.75)
DURACION_VENTANA   = 30.0

# Mínimo de segundos con voz (sumando ventanas) para considerar que el video tiene habla
MIN_SEGUNDOS_VOZ = 3.0

# Probabilidad mínima del idioma ganador; por debajo no se asigna idioma
UMBRAL_IDIOMA = 0.60

# Energía RMS por debajo de la cual se considera silencio (mismo criterio que ia_video_service)
UMBRAL_SILENCIO_RMS = 1e-4

# --- Música (AST) ---
MODELO_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
DURACION_TROZO_AST = 10.0   # AST analiza ~10 s por entrada; cada ventana se parte en trozos

# Una ventana tiene "música dominante" si P(música) ≥ UMBRAL_MUSICA y además hay canto
# (P(canto) ≥ UMBRAL_CANTO) o no hay habla (P(habla) < UMBRAL_HABLA).
UMBRAL_MUSICA = 0.50
UMBRAL_CANTO  = 0.30
UMBRAL_HABLA  = 0.50

# Fracción mínima de ventanas con música dominante para etiquetar el video como Music
FRACCION_MUSICA = 0.5

# En videos SIN música, para aceptar el idioma de Whisper AST debe ver habla o canto con al
# menos esta probabilidad en alguna ventana (el VAD solo a veces confunde ruido con voz).
# En videos CON música no se exige: AST suele dar poco "canto" en canciones mezcladas, y ahí
# el filtro lo hacen el VAD (0 s de voz en instrumentales) y la confianza de Whisper.
UMBRAL_VOZ_AST = 0.30

# Índices de AudioSet (id2label del modelo AST)
IDX_MUSICA = [137]                                         # Music
IDX_HABLA  = [0, 1, 2, 3, 4, 5, 7]                         # Speech, Male/Female/Child speech, Conversation, Narration, Speech synthesizer
IDX_CANTO  = [27, 28, 32, 33, 34, 35, 36, 37, 254, 255, 266]  # Singing, Choir, ...singing, Rapping, Humming, Vocal music, A capella, Song

TAG_MUSICA = "Music"

# Códigos de idioma de Whisper → nombre del tag (en inglés, igual que el resto de tags)
IDIOMAS_WHISPER = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian",
    "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali",
    "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer",
    "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan",
    "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati",
    "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", "mt": "Maltese",
    "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Burmese", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala",
    "ha": "Hausa", "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# Tags que salen de esta detección (no del KNN visual), en minúsculas para comparar
TAGS_IDIOMA_Y_MUSICA = {n.lower() for n in IDIOMAS_WHISPER.values()} | {TAG_MUSICA.lower()}

# ============================================================
# MODELOS (carga perezosa: solo se cargan si hay audio que analizar)
# ============================================================

_device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper = None
_ast = None


def _obtener_whisper():
    global _whisper
    if _whisper is None:
        compute_type = "float16" if _device == "cuda" else "int8"
        print(f"  Cargando Whisper '{MODELO_WHISPER}' ({_device.upper()})...")
        _whisper = WhisperModel(MODELO_WHISPER, device=_device, compute_type=compute_type)
    return _whisper


def _obtener_ast():
    global _ast
    if _ast is None:
        print(f"  Cargando AST para detección de música ({_device.upper()})...")
        extractor = ASTFeatureExtractor.from_pretrained(MODELO_AST)
        modelo = ASTForAudioClassification.from_pretrained(MODELO_AST).to(_device)
        modelo.eval()
        _ast = (extractor, modelo)
    return _ast

# ============================================================
# DETECCIÓN
# ============================================================

def nombre_idioma(codigo):
    return IDIOMAS_WHISPER.get(codigo, codigo.upper())


def _cargar_ventanas(ruta_video):
    """Devuelve lista de arrays de audio mono a 16 kHz, una por ventana."""
    try:
        duracion = librosa.get_duration(path=ruta_video)
    except Exception:
        duracion = None

    if not duracion:
        # Sin duración conocida: solo se analiza el inicio
        offsets = [0.0]
    else:
        offsets = sorted({
            round(min(p * duracion, max(duracion - DURACION_VENTANA, 0.0)), 1)
            for p in POSICIONES_VENTANA
        })

    ventanas = []
    for offset in offsets:
        audio, _ = librosa.load(ruta_video, sr=SAMPLE_RATE, mono=True,
                                offset=offset, duration=DURACION_VENTANA)
        if audio.size:
            ventanas.append(audio.astype(np.float32))
    return ventanas


def _solo_voz(audio):
    """Recorta el audio a los tramos con voz según el VAD. Devuelve (audio_voz, segundos_voz)."""
    tramos = get_speech_timestamps(audio, vad_options=VadOptions())
    if not tramos:
        return np.zeros(0, dtype=np.float32), 0.0
    voz = np.concatenate([audio[t["start"]:t["end"]] for t in tramos])
    return voz, voz.size / SAMPLE_RATE


def _probabilidades_sonido(audio):
    """
    Corre AST sobre la ventana (partida en trozos de ~10 s) y devuelve
    (P(música), P(habla), P(canto)) promediadas entre trozos.
    """
    extractor, modelo = _obtener_ast()
    paso = int(DURACION_TROZO_AST * SAMPLE_RATE)
    trozos = [audio[i:i + paso] for i in range(0, len(audio), paso)]
    trozos = [t for t in trozos if t.size >= SAMPLE_RATE] or [audio]  # descarta colas < 1 s

    entradas = extractor(trozos, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(_device)
    with torch.no_grad():
        probs = torch.sigmoid(modelo(**entradas).logits).cpu().numpy()  # multi-etiqueta

    musica = probs[:, IDX_MUSICA].max(axis=1).mean()
    habla  = probs[:, IDX_HABLA].max(axis=1).mean()
    canto  = probs[:, IDX_CANTO].max(axis=1).mean()
    return float(musica), float(habla), float(canto)


def _musica_dominante(p_musica, p_habla, p_canto):
    return p_musica >= UMBRAL_MUSICA and (p_canto >= UMBRAL_CANTO or p_habla < UMBRAL_HABLA)


def _detectar_lengua(ventanas):
    """Whisper sobre los tramos con voz. Devuelve (codigo, probabilidad, segundos_voz)."""
    votos     = {}
    total_voz = 0.0
    for audio in ventanas:
        voz, segundos = _solo_voz(audio)
        if segundos < 1.0:
            continue
        _, _, probabilidades = _obtener_whisper().detect_language(audio=voz)
        for codigo, prob in probabilidades:
            votos[codigo] = votos.get(codigo, 0.0) + prob * segundos  # ponderado por segundos de voz
        total_voz += segundos

    if total_voz < MIN_SEGUNDOS_VOZ:
        return None, None, round(total_voz, 1)
    codigo, peso = max(votos.items(), key=lambda kv: kv[1])
    return codigo, peso / total_voz, round(total_voz, 1)


def detectar_idioma(ruta_video):
    """
    Detecta música e idioma hablado del video.
    Devuelve dict con "tags" (lista: Music y/o idioma, puede ir vacía) y el detalle
    de la detección para el registro.
    """
    resultado = {"tags": [], "idioma": None, "codigo": None, "probabilidad": None,
                 "segundos_voz": 0.0, "prob_musica": None, "prob_habla": None, "prob_canto": None}

    try:
        ventanas = _cargar_ventanas(ruta_video)
    except Exception as e:
        print(f"  ⚠ No se pudo leer el audio para idioma de {ruta_video}: {e}")
        return resultado

    ventanas = [v for v in ventanas if np.sqrt(np.mean(v ** 2)) >= UMBRAL_SILENCIO_RMS]
    if not ventanas:
        return resultado  # silencio: lo cubre el tag de "No sound"

    # --- Música: ¿domina en suficientes ventanas? ---
    sonidos = [_probabilidades_sonido(v) for v in ventanas]
    con_musica = sum(_musica_dominante(*p) for p in sonidos)
    es_musica = con_musica / len(sonidos) >= FRACCION_MUSICA

    resultado["prob_musica"] = round(max(p[0] for p in sonidos), 3)
    resultado["prob_habla"]  = round(max(p[1] for p in sonidos), 3)
    resultado["prob_canto"]  = round(max(p[2] for p in sonidos), 3)

    # --- Idioma (letra de la canción o habla) ---
    hay_voz = es_musica or max(resultado["prob_habla"], resultado["prob_canto"]) >= UMBRAL_VOZ_AST
    if hay_voz:
        codigo, probabilidad, segundos = _detectar_lengua(ventanas)
        resultado["segundos_voz"] = segundos
        if codigo is not None:
            resultado["codigo"]       = codigo
            resultado["probabilidad"] = round(probabilidad, 3)
            if probabilidad >= UMBRAL_IDIOMA:
                resultado["idioma"] = nombre_idioma(codigo)

    if es_musica:
        resultado["tags"].append(TAG_MUSICA)
    if resultado["idioma"]:
        resultado["tags"].append(resultado["idioma"])
    return resultado

# ============================================================
# UTILIDADES DE BASE DE DATOS
# ============================================================

def asegurar_tags_video(cursor, nombres):
    """
    Garantiza que existan en la tabla `tag` (type='VIDEO') los tags indicados.
    Necesario porque el frontend descarta las sugerencias cuyo nombre no existe como tag.
    """
    for nombre in nombres:
        cursor.execute(
            "SELECT id FROM tag WHERE LOWER(name) = LOWER(%s) AND type = 'VIDEO' LIMIT 1",
            (nombre,)
        )
        if not cursor.fetchall():  # fetchall: con cursores sin buffer, fetchone deja resultados sin leer
            cursor.execute("INSERT INTO tag (name, type) VALUES (%s, 'VIDEO')", (nombre,))
            print(f"  ➕ Tag creado: {nombre}")
