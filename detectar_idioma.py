"""
Detección del idioma hablado en un video con faster-whisper + VAD (Silero).

Resultado por video:
  - Nombre del idioma en inglés (ej. "Spanish") → hay voz y Whisper está seguro.
  - TAG_MUSICA ("Music")                        → hay sonido pero no hay voz.
  - None                                        → silencio, voz con idioma dudoso o error.

Las canciones con letra cuentan como voz: se etiquetan con el idioma de la letra.
"""
import numpy as np
import librosa
import torch
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

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
# MODELO (carga perezosa: solo se carga si realmente se detecta idioma)
# ============================================================

_modelo = None


def _obtener_modelo():
    global _modelo
    if _modelo is None:
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
        print(f"  Cargando Whisper '{MODELO_WHISPER}' ({device.upper()})...")
        _modelo = WhisperModel(MODELO_WHISPER, device=device, compute_type=compute_type)
    return _modelo

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


def detectar_idioma(ruta_video):
    """
    Detecta el idioma hablado del video.
    Devuelve dict {"tag", "codigo", "probabilidad", "segundos_voz"}; "tag" puede ser
    un idioma, TAG_MUSICA o None (ver docstring del módulo).
    """
    resultado = {"tag": None, "codigo": None, "probabilidad": None, "segundos_voz": 0.0}

    try:
        ventanas = _cargar_ventanas(ruta_video)
    except Exception as e:
        print(f"  ⚠ No se pudo leer el audio para idioma de {ruta_video}: {e}")
        return resultado

    if not ventanas or all(np.sqrt(np.mean(v ** 2)) < UMBRAL_SILENCIO_RMS for v in ventanas):
        return resultado  # silencio: lo cubre el tag de "No sound"

    # Votación ponderada por segundos de voz: cada ventana con habla aporta sus probabilidades
    votos       = {}
    total_voz   = 0.0
    modelo      = None
    for audio in ventanas:
        voz, segundos = _solo_voz(audio)
        if segundos < 1.0:
            continue
        modelo = modelo or _obtener_modelo()
        _, _, probabilidades = modelo.detect_language(audio=voz)
        for codigo, prob in probabilidades:
            votos[codigo] = votos.get(codigo, 0.0) + prob * segundos
        total_voz += segundos

    resultado["segundos_voz"] = round(total_voz, 1)

    if total_voz < MIN_SEGUNDOS_VOZ:
        resultado["tag"] = TAG_MUSICA  # hay sonido pero no habla
        return resultado

    codigo, peso = max(votos.items(), key=lambda kv: kv[1])
    probabilidad = peso / total_voz
    resultado["codigo"]       = codigo
    resultado["probabilidad"] = round(probabilidad, 3)
    if probabilidad >= UMBRAL_IDIOMA:
        resultado["tag"] = nombre_idioma(codigo)
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
