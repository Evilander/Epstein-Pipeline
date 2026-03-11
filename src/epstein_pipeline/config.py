"""Pipeline configuration using Pydantic BaseSettings with EPSTEIN_ env prefix."""

from __future__ import annotations

import importlib.util
import os
from enum import Enum
from pathlib import Path
from typing import Literal, TypedDict, cast

from pydantic_settings import BaseSettings


class OcrBackend(str, Enum):
    """Available OCR backends, ordered by speed (fastest first)."""

    AUTO = "auto"
    PYMUPDF = "pymupdf"
    SURYA = "surya"
    OLMOCR = "olmocr"
    DOCLING = "docling"


class NerBackend(str, Enum):
    """Available NER backends."""

    SPACY = "spacy"
    GLINER = "gliner"
    BOTH = "both"


class DedupMode(str, Enum):
    """Dedup strategies — can be combined via 'all'."""

    EXACT = "exact"  # content hash + title fuzzy + Bates overlap
    MINHASH = "minhash"  # MinHash/LSH near-duplicate
    SEMANTIC = "semantic"  # embedding cosine similarity
    ALL = "all"  # exact → minhash → semantic


class PathCheck(TypedDict):
    path: str
    exists: bool
    writable: bool


class EnvFlags(TypedDict):
    neon_database_url: bool
    opensanctions_api_key: bool
    auditor_anthropic_api_key: bool
    auditor_voyage_api_key: bool
    auditor_cohere_api_key: bool


class PersonsRegistryStatus(TypedDict):
    configured_path: str
    resolved_path: str
    exists: bool
    using_bundled_fallback: bool


class RuntimeSummary(TypedDict):
    paths: dict[str, PathCheck]
    optional_dependencies: dict[str, bool]
    env: EnvFlags
    persons_registry: PersonsRegistryStatus


class Settings(BaseSettings):
    """Pipeline settings loaded from environment variables prefixed with EPSTEIN_.

    Example:
        EPSTEIN_DATA_DIR=/mnt/data
        EPSTEIN_SPACY_MODEL=en_core_web_trf
        EPSTEIN_DEDUP_THRESHOLD=0.95
        EPSTEIN_NEON_DATABASE_URL=postgresql://...
    """

    model_config = {"env_prefix": "EPSTEIN_"}

    # ── Directory paths ──────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("./.cache")
    persons_registry_path: Path = Path("./data/persons-registry.json")

    # ── General processing ───────────────────────────────────────────────
    max_workers: int = 4

    # ── OCR settings ─────────────────────────────────────────────────────
    ocr_backend: OcrBackend = OcrBackend.AUTO
    ocr_batch_size: int = 50
    ocr_confidence_threshold: float = 0.7  # flag pages below this
    ocr_fallback_chain: list[str] = ["pymupdf", "surya", "olmocr", "docling"]

    # ── NER settings ─────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_trf"  # upgraded from en_core_web_sm
    ner_backend: NerBackend = NerBackend.BOTH
    gliner_model: str = "urchade/gliner_multi_pii-v1"
    ner_confidence_threshold: float = 0.5

    # ── Dedup settings ───────────────────────────────────────────────────
    dedup_mode: DedupMode = DedupMode.ALL
    dedup_threshold: float = 0.90  # title fuzzy match threshold
    dedup_jaccard_threshold: float = 0.80  # MinHash Jaccard threshold
    dedup_semantic_threshold: float = 0.95  # embedding cosine similarity
    dedup_shingle_size: int = 5  # n-gram size for MinHash
    dedup_num_perm: int = 128  # MinHash permutation count

    # ── Embedding settings ───────────────────────────────────────────────
    embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embedding_dimensions: int = 768  # 768 full, 256 Matryoshka
    embedding_chunk_size: int = 3200  # chars (~800 tokens)
    embedding_chunk_overlap: int = 800  # chars (~200 tokens)
    embedding_batch_size: int | None = None  # None = auto-detect
    embedding_device: str | None = None  # None = auto-detect

    # ── Chunker settings ─────────────────────────────────────────────────
    chunker_mode: Literal["fixed", "semantic"] = "semantic"
    chunker_target_tokens: int = 512  # target chunk size in tokens
    chunker_min_tokens: int = 100  # minimum chunk size
    chunker_max_tokens: int = 1024  # maximum chunk size

    # ── Neon Postgres settings ───────────────────────────────────────────
    neon_database_url: str | None = None  # postgresql://...@...neon.tech/...
    neon_pool_size: int = 10
    neon_batch_size: int = 100  # rows per upsert batch

    # ── Document classifier settings ─────────────────────────────────────
    classifier_model: str = "facebook/bart-large-mnli"
    classifier_confidence_threshold: float = 0.6

    # ── Knowledge graph settings ─────────────────────────────────────────
    kg_llm_provider: str = "openai"  # "openai" or "anthropic"
    kg_llm_model: str = "gpt-4o-mini"
    kg_extract_relationships: bool = False  # LLM extraction is opt-in

    # ── Sea Doughnut import ──────────────────────────────────────────────
    sea_doughnut_dir: Path | None = None

    # ── Site sync ────────────────────────────────────────────────────────
    site_dir: Path | None = None

    # ── AI / Vision settings ─────────────────────────────────────────────
    vision_model: str = "llava"
    summarizer_provider: str = "ollama"
    summarizer_model: str = "llama3.2"

    # ── Transcription ────────────────────────────────────────────────────
    whisper_model: str = "large-v3"

    # ── OpenSanctions settings ──────────────────────────────────────────
    opensanctions_api_key: str | None = None
    opensanctions_match_threshold: float = 0.5  # minimum score to flag

    # ── Person Auditor settings ────────────────────────────────────────
    auditor_anthropic_api_key: str | None = None
    auditor_anthropic_model: str = "claude-sonnet-4-6"  # Primary model for verification
    auditor_fast_model: str = "claude-haiku-4-5-20251001"  # Quick screening/decomposition
    auditor_deep_model: str = "claude-sonnet-4-6"  # Deep analysis of flagged records
    auditor_voyage_api_key: str | None = None
    auditor_cohere_api_key: str | None = None
    auditor_use_batch_api: bool = True
    auditor_dedup_threshold: float = 0.85
    auditor_name_fuzzy_threshold: int = 85
    auditor_wikidata_rate_limit: float = 1.0  # seconds between requests
    auditor_max_claims_per_person: int = 10
    auditor_max_doc_chunks: int = 5
    auditor_severity_critical: int = 70
    auditor_severity_high: int = 40
    auditor_severity_medium: int = 20

    def ensure_dirs(self) -> None:
        """Create data, output, and cache directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_persons_registry_path(self) -> Path:
        """Return the runtime registry path, falling back to the bundled registry."""
        if self.persons_registry_path.exists():
            return self.persons_registry_path

        bundled = Path(__file__).with_name("persons-registry.json")
        if bundled.exists():
            return bundled

        return self.persons_registry_path

    def runtime_summary(self) -> RuntimeSummary:
        """Return a lightweight runtime snapshot for operator checks."""
        resolved_registry = self.resolve_persons_registry_path()
        path_checks = {
            "data_dir": _path_check(self.data_dir, is_dir=True),
            "output_dir": _path_check(self.output_dir, is_dir=True),
            "cache_dir": _path_check(self.cache_dir, is_dir=True),
            "persons_registry": _path_check(resolved_registry, is_dir=False),
        }
        optional_dependencies = {
            "spacy": _module_available("spacy"),
            "gliner": _module_available("gliner"),
            "pymupdf": _module_available("fitz"),
            "faster_whisper": _module_available("faster_whisper"),
            "sentence_transformers": _module_available("sentence_transformers"),
            "psycopg": _module_available("psycopg"),
        }
        env_flags = cast(
            EnvFlags,
            {
                "neon_database_url": bool(self.neon_database_url),
                "opensanctions_api_key": bool(self.opensanctions_api_key),
                "auditor_anthropic_api_key": bool(self.auditor_anthropic_api_key),
                "auditor_voyage_api_key": bool(self.auditor_voyage_api_key),
                "auditor_cohere_api_key": bool(self.auditor_cohere_api_key),
            },
        )

        return {
            "paths": path_checks,
            "optional_dependencies": optional_dependencies,
            "env": env_flags,
            "persons_registry": {
                "configured_path": str(self.persons_registry_path.resolve()),
                "resolved_path": str(resolved_registry.resolve()),
                "exists": resolved_registry.exists(),
                "using_bundled_fallback": resolved_registry != self.persons_registry_path,
            },
        }


def _module_available(module_name: str) -> bool:
    """Check whether an optional dependency is importable without importing it."""
    return importlib.util.find_spec(module_name) is not None


def _path_check(path: Path, *, is_dir: bool) -> PathCheck:
    """Return basic existence and writability information for a path."""
    target = path if is_dir else path.parent
    exists = path.exists()
    writable = target.exists() and os.access(target, os.W_OK)
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "writable": writable,
    }
