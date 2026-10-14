# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Multimodal AI video classification and duplicate detection system. It uses CLIP (vision) and CLAP (audio) embeddings to:
1. Build a knowledge base of embeddings from tagged training videos stored in MySQL
2. Classify new untagged videos via K-NN voting (K=5) with audio binary detection
3. Detect near-duplicate videos within a tag using cosine similarity + Union-Find clustering

## Running the Scripts

```bash
# Install dependencies
pip install -r requirements.txt

# Sync knowledge base incrementally and classify a batch
python ia_video_service.py  # procesar_lote() calls sincronizar_memoria() first

# Classify a batch of unclassified videos (default limit=5); syncs memory first
python -c "from ia_video_service import procesar_lote; procesar_lote(limite=10)"

# Regenerate the knowledge base from scratch (only needed if the model/extraction changes)
python -c "from ia_video_service import regenerar_memoria; regenerar_memoria()"

# Print tag distribution and audio statistics for diagnostics
python -c "from ia_video_service import diagnostico_memoria; diagnostico_memoria()"

# Measure per-tag classification error: confirmed suggestions (video_tag_temporal.confirm=1) vs final tags (video_tag)
python medir_error_tags.py
python medir_error_tags.py --desde 2026-10-01 --min-soporte 10 --csv errores.csv

# Exclude language-review suggestions (they include the real tags) from the measurement
python medir_error_tags.py --hasta 2026-10-15

# Spoken-language tags for existing videos (one-off): remove manual language tags, then suggest detected ones
python borrar_tags_idioma.py --dry-run          # default tags: Spanish English
python borrar_tags_idioma.py
python etiquetar_idioma_existentes.py --limite 20 --dry-run
python etiquetar_idioma_existentes.py           # resumable; skips videos already in salidas/resultados_idioma.csv
python etiquetar_idioma_existentes.py --corregir-music   # fix Music suggested by the old version (noise tagged as Music)

# Detect duplicates within a specific tag (tag is a CLI argument)
python detectar_duplicados.py Nature
python detectar_duplicados.py Nature --umbral 0.98
python -c "from detectar_duplicados import detectar_duplicados; detectar_duplicados('Nature', umbral=1.0)"
```

## Environment Variables (`.env` file required)

```
BASE_PATH=./media        # Root path where video files are stored
DB_HOST=...
DB_PORT=3306
DB_USER=...
DB_PASSWORD=...
DB_DATABASE=...
```

## Architecture

**Two scripts, three phases:**

- `ia_video_service.py` handles **training** (building `memoria_multimodal_ia.pkl`) and **classification** (writing predictions to `video_tag_temporal` table).
- `detectar_duplicados.py` handles **duplicate detection** (writing groups to `video_duplicate_group` and `video_duplicate_member` tables).

**Embedding pipeline:**
- Extract 8 uniformly-sampled frames → CLIP model → 512-dim visual embedding (average across frames)
- Load audio at 48kHz → CLAP model → 512-dim audio embedding
- Concatenate at 70%/30% weight → normalized 1024-dim combined embedding

**Classification (KNN voting):**
- Cosine similarity against all training embeddings
- Top-K=5 neighbors with similarity ≥ `UMBRAL_SIMILITUD` (0.20)
- Weighted vote: tag wins if it reaches `UMBRAL_VOTO` (0.30) of total weight
- Audio tags handled separately via binary detection (`hay_audio()`)
- Language and `Music` tags are excluded from KNN voting; they come from `detectar_idioma.py` instead

**Music and spoken language (`detectar_idioma.py`):**
- Analyzes 30 s windows at 0%, 40%, 75% of the video
- Music: AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, AudioSet labels Music/Speech/Singing). A window has dominant music if P(music) ≥ `UMBRAL_MUSICA` (0.5) AND (P(singing) ≥ `UMBRAL_CANTO` (0.3) OR P(speech) < `UMBRAL_HABLA` (0.5)); video gets `Music` if ≥ `FRACCION_MUSICA` (50%) of windows do. Speech over background music is NOT Music
- Language: faster-whisper (`MODELO_WHISPER = "small"`) on Silero-VAD speech segments, probabilities weighted by seconds of speech; needs ≥ `MIN_SEGUNDOS_VOZ` (3 s) and prob ≥ `UMBRAL_IDIOMA` (0.60). For non-music videos AST must also see speech/singing ≥ `UMBRAL_VOZ_AST` (0.3) — blocks Whisper hallucinating a language on noise
- Result (`"tags"` list): noise → nothing (keeps `Sound`); instrumental → `Music`; song with lyrics → `Music` + lyrics language; speech (with or without background music) → language
- New language tags are created in `tag` (type `VIDEO`) via `asegurar_tags_video()` — the frontend drops suggestions whose tag doesn't exist
- `etiquetar_idioma_existentes.py` writes suggestions to `video_tag_temporal` (confirm=0) as **current real tags + detected tag**, because confirming in the backend replaces ALL video tags with the suggestion. Overwritten temporal rows are backed up to `salidas/` first; their `date_creation` is reset, so use `medir_error_tags.py --hasta` to keep them out of error metrics
- `borrar_tags_idioma.py` only deletes `video_tag` rows (never `tag` rows — shared with images/mangas) and strips those names from pending suggestions, with a CSV backup in `salidas/` (gitignored)

**Duplicate detection:**
- Tag name is passed as a CLI positional argument (`python detectar_duplicados.py <tag>`), not hardcoded
- Videos are first grouped into connected components (Union-Find) by tag overlap: two videos are only compared visually if they share ≥ `MIN_TAGS_COMPARTIDOS` (3) tags
- Within each component, N×N cosine similarity matrix is computed over visual embeddings
- Complete-linkage clustering: a video joins a group only if it satisfies similarity ≥ `UMBRAL_DUPLICADO` (1.0) AND shares ≥ `MIN_TAGS_COMPARTIDOS` tags with EVERY current group member (avoids transitive false positives)
- Group leader = video with highest average similarity to all group members

**Persistence:**
- `memoria_multimodal_ia.pkl` — cached embeddings for all training videos (gitignored). Kept in sync incrementally by `sincronizar_memoria()` before every batch: embeds only new tagged videos, refreshes tags of existing ones (no re-embedding), drops untagged ones, recomputes if `video_path` changed. Written atomically (`.tmp` + `os.replace`), checkpointed every `GUARDAR_CADA` (100) new videos. Videos whose file is missing or whose visual embedding fails are not stored and are retried on the next sync.
- MySQL DB — source of video paths/tags and destination for classification/duplicate results

## Key Tuning Constants

| Constant | File | Default | Effect |
|---|---|---|---|
| `N_FRAMES` | both | `8` | Frames sampled per video for visual embedding |
| `K_VECINOS` | ia_video_service | `5` | Neighbors used in KNN voting |
| `UMBRAL_SIMILITUD` | ia_video_service | `0.20` | Min cosine sim to include a neighbor |
| `UMBRAL_VOTO` | ia_video_service | `0.30` | Min vote fraction for a tag to be assigned |
| `UMBRAL_DUPLICADO` | detectar_duplicados | `1.0` | Cosine sim threshold to consider two videos duplicates (overridable via `--umbral`) |
| `MIN_TAGS_COMPARTIDOS` | detectar_duplicados | `3` | Min shared tags for two videos to be compared visually at all |

## Models Used

- **CLIP**: `openai/clip-vit-base-patch32` — vision embeddings
- **CLAP**: `HTSAT-tiny` (via `laion_clap`) — audio embeddings, initialized with `enable_fusion=False`

Both models run on GPU if available, falling back to CPU automatically.
