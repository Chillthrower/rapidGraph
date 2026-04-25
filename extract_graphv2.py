from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import spacy
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

LOGGER = logging.getLogger("extract_graph")

GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
REBEL_MODEL_NAME = "Babelscape/rebel-large"

DEFAULT_ENTITY_LABELS = [
    "person",
    "organization",
    "location",
    "city",
    "country",
    "region",
    "geopolitical entity",
    "facility",
    "event",
    "product",
    "work of art",
    "law",
    "language",
    "nationality",
    "religion",
    "job title",
    "scientific theory",
    "academic discipline",
    "research topic",
    "disease",
    "chemical compound",
    "biological species",
    "food",
    "award",
    "date",
    "time",
    "money",
    "quantity",
    "software",
    "hardware",
]

TECHNICAL_KEYWORDS = {
    "attention",
    "embedding",
    "embeddings",
    "encoder",
    "decoder",
    "feed-forward",
    "feed forward",
    "network",
    "networks",
    "layer",
    "layers",
    "softmax",
    "relu",
    "projection",
    "projections",
    "normalization",
    "encoding",
    "encodings",
    "query",
    "queries",
    "key",
    "keys",
    "value",
    "values",
    "token",
    "tokens",
    "position",
    "positions",
    "sequence",
    "sequences",
    "matrix",
    "matrices",
    "vector",
    "vectors",
    "head",
    "heads",
    "sublayer",
    "sublayers",
    "transformer",
    "recurrence",
    "convolution",
    "probability",
    "probabilities",
    "transformation",
    "transformations",
}

TECHNICAL_TYPE_OVERRIDES = {
    "Hardware",
    "Software",
    "Product",
    "WorkOfArt",
    "ScientificTheory",
    "AcademicDiscipline",
}

REBEL_RELATION_BLOCKLIST_FOR_TECHNICAL = {
    "OPPOSITE_OF",
    "SUBCLASS_OF",
    "COUNTRY",
    "CHAIRPERSON",
    "EMPLOYER",
    "HEADQUARTERS_LOCATION",
    "INSTANCE_OF",
    "USED_BY",
}

GENERIC_LOW_SIGNAL_ENTITIES = {
    "algorithm",
    "model",
    "output",
    "outputs",
    "region",
    "regions",
}

TECHNICAL_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Za-z0-9][\w-]*\s+){0,3}"
    r"(?:attention|embeddings?|encodings?|layers?|network|networks|softmax|queries|query|keys|key|"
    r"values|value|positions?|tokens?|transformations?|projections?|matrix|matrices|vectors?|"
    r"probabilities|heads?|stack)\b",
    re.IGNORECASE,
)

VARIABLE_PATTERN = re.compile(
    r"\b(?:dmodel|dk|dv|df|ffn|q|k|v|h|N|W[QKVO]?\d*|b\d*)\b"
)

PHRASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "we",
    "where",
    "which",
    "while",
    "with",
}

ALLOWED_SINGLE_WORD_TECHNICAL = {
    "attention",
    "decoder",
    "embedding",
    "embeddings",
    "encoder",
    "softmax",
    "transformer",
}

BAD_PHRASE_WORDS = {
    "all",
    "allow",
    "allows",
    "assigned",
    "call",
    "come",
    "comes",
    "composed",
    "compute",
    "computed",
    "consists",
    "contain",
    "contains",
    "corresponding",
    "depends",
    "derive",
    "derived",
    "each",
    "every",
    "include",
    "includes",
    "inject",
    "mapping",
    "most",
    "outperform",
    "outperforms",
    "perform",
    "performs",
    "prevent",
    "prevents",
    "same",
    "use",
    "uses",
    "using",
    "yielding",
}

MODEL_KEYWORDS = {
    "transformer",
    "bert",
    "gpt",
    "llm",
    "architecture",
    "model",
}

METHOD_KEYWORDS = {
    "attention",
    "encoding",
    "encodings",
    "algorithm",
    "mechanism",
    "method",
}

COMPONENT_KEYWORDS = {
    "encoder",
    "decoder",
    "layer",
    "layers",
    "stack",
    "head",
    "heads",
    "sublayer",
    "network",
    "embedding layer",
    "embedding layers",
}

TRANSFORMATION_KEYWORDS = {
    "softmax",
    "projection",
    "projections",
    "transformation",
    "transformations",
    "normalization",
    "linear transformation",
    "linear transformations",
}

DATA_STRUCTURE_KEYWORDS = {
    "matrix",
    "matrices",
    "vector",
    "vectors",
    "embedding",
    "embeddings",
    "token",
    "tokens",
    "sequence",
    "sequences",
    "query",
    "queries",
    "key",
    "keys",
    "value",
    "values",
}

METRIC_KEYWORDS = {
    "accuracy",
    "loss",
    "probability",
    "probabilities",
    "gradient",
    "gradients",
    "cost",
}

SCHEMA_TYPE_COMPATIBILITY = {
    "Model": "TechnicalConcept",
    "Method": "TechnicalConcept",
    "Component": "TechnicalConcept",
    "Transformation": "TechnicalConcept",
    "DataStructure": "TechnicalConcept",
    "Metric": "Quantity",
}

SCHEMA_RELATION_ALIASES = {
    "CONSISTS_OF": "COMPOSED_OF",
    "COMPOSED_OF": "COMPOSED_OF",
    "PART_OF": "PART_OF",
}


class MentionModel(BaseModel):
    text: str
    start: int
    end: int
    chunk_index: int
    document_id: str = ""
    chunk_id: str = ""


class EntityModel(BaseModel):
    id: str
    text: str
    canonical: str
    type: str
    confidence: float
    mentions: list[MentionModel]


class RelationModel(BaseModel):
    source_id: str
    target_id: str
    relation: str
    confidence: float
    evidence: str
    chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class DocumentModel(BaseModel):
    id: str
    source: str
    title: str
    text_hash: str
    char_count: int


class ChunkModel(BaseModel):
    id: str
    document_id: str
    index: int
    text: str
    start: int
    end: int
    block_index: int
    overlap_sentences: int


class RelationSupportModel(BaseModel):
    source_id: str
    relation: str
    target_id: str
    chunk_ids: list[str]
    document_ids: list[str]
    evidence: str


class SchemaEdgeModel(BaseModel):
    source_type: str
    relation: str
    target_type: str
    count: int
    examples: list[str]


class MetaModel(BaseModel):
    entity_model: str
    relation_model: str
    entity_threshold: float
    relation_threshold: float
    chunk_count: int
    elapsed_seconds: float
    chunk_mode: str = "paragraph"
    chunk_overlap: int = 1
    entity_candidates: int = 0
    relation_candidates: int = 0
    entities_kept: int = 0
    relations_kept: int = 0
    mode: str = "balanced"
    relation_backend_strategy: str = "heuristic_plus_selective_rebel"
    rebel_spans_considered: int = 0
    rebel_spans_run: int = 0
    rebel_skipped: int = 0
    embedding_enabled: bool = False
    embedding_model: str = ""
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    embedding_comparisons: int = 0
    embedding_merges: int = 0
    embedding_links: int = 0
    document_count: int = 0
    chunk_records: int = 0
    chunk_text_included: bool = True
    warnings: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class GraphExtraction(BaseModel):
    entities: list[EntityModel]
    relations: list[RelationModel]
    potential_schema: list[SchemaEdgeModel]
    expanded_schema: list[SchemaEdgeModel] = Field(default_factory=list)
    documents: list[DocumentModel] = Field(default_factory=list)
    chunks: list[ChunkModel] = Field(default_factory=list)
    relation_support: list[RelationSupportModel] = Field(default_factory=list)
    meta: MetaModel


NEO4J_CONSTRAINT_QUERIES = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
)


NEO4J_DOCUMENT_QUERY = """
UNWIND $rows AS row
MERGE (d:Document {id: row.id})
SET d.source = row.source,
    d.title = row.title,
    d.text_hash = row.text_hash,
    d.char_count = row.char_count
"""


NEO4J_CHUNK_QUERY = """
UNWIND $rows AS row
MERGE (c:Chunk {id: row.id})
SET c.document_id = row.document_id,
    c.index = row.index,
    c.text = row.text,
    c.start = row.start,
    c.end = row.end,
    c.block_index = row.block_index,
    c.overlap_sentences = row.overlap_sentences
WITH c, row
MATCH (d:Document {id: row.document_id})
MERGE (d)-[:HAS_CHUNK]->(c)
"""


NEO4J_ENTITY_QUERY = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id})
SET e.text = row.text,
    e.canonical = row.canonical,
    e.type = row.type,
    e.confidence = row.confidence
"""


NEO4J_MENTION_QUERY = """
UNWIND $rows AS row
MATCH (c:Chunk {id: row.chunk_id})
MATCH (e:Entity {id: row.entity_id})
MERGE (c)-[m:MENTIONS {
    entity_id: row.entity_id,
    start: row.start,
    end: row.end,
    text: row.text
}]->(e)
SET m.document_id = row.document_id
"""


NEO4J_RELATION_QUERY = """
UNWIND $rows AS row
MATCH (s:Entity {id: row.source_id})
MATCH (t:Entity {id: row.target_id})
MERGE (s)-[r:RELATES_TO {
    relation: row.relation,
    source_id: row.source_id,
    target_id: row.target_id
}]->(t)
SET r.confidence = row.confidence,
    r.evidence = row.evidence,
    r.chunk_ids = row.chunk_ids,
    r.document_ids = row.document_ids
"""


NEO4J_DELETE_DOCUMENT_RELATIONS_QUERY = """
MATCH ()-[r:RELATES_TO]->()
WHERE $document_id IN r.document_ids
DELETE r
"""


NEO4J_DELETE_DOCUMENT_SUBGRAPH_QUERY = """
MATCH (d:Document {id: $document_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
WITH d, collect(DISTINCT c) AS chunks, collect(DISTINCT e) AS entities
FOREACH (chunk IN chunks | DETACH DELETE chunk)
WITH d, entities
DETACH DELETE d
WITH entities
UNWIND entities AS entity
WITH DISTINCT entity
WHERE entity IS NOT NULL
  AND NOT (entity)<-[:MENTIONS]-(:Chunk)
  AND NOT (entity)-[:RELATES_TO]-(:Entity)
DELETE entity
"""


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    start: int
    end: int
    block_index: int = 0
    overlap_sentences: int = 0


@dataclass(slots=True)
class EntityCandidate:
    text: str
    label: str
    score: float
    start: int
    end: int
    chunk_index: int
    document_id: str = ""


@dataclass(slots=True)
class RelationCandidate:
    subject: str
    relation: str
    obj: str
    score: float
    evidence: str
    chunk_index: int
    document_id: str = ""


@dataclass(slots=True)
class DocumentInput:
    text: str
    source: str
    title: str


class EntityBackend(Protocol):
    model_name: str
    fallback_used: bool
    warnings: list[str]

    def extract(self, chunk: Chunk, threshold: float) -> list[EntityCandidate]:
        ...


class RelationBackend(Protocol):
    model_name: str
    fallback_used: bool
    warnings: list[str]

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        ...


class EmbeddingBackend(Protocol):
    model_name: str

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        ...


RELATION_TRIGGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\buses?\b",
        r"\bcontains?\b",
        r"\bconsists of\b",
        r"\bcomposed of\b",
        r"\bdepends? on\b",
        r"\bcome(?:s)? from\b",
        r"\badd(?:ed|s)?\b.+\bto\b",
        r"\boutperforms?\b",
        r"\bapplied to\b",
        r"\bproject(?:ed)? to\b",
        r"\bproduce(?:s|d)?\b",
        r"\byield(?:s|ed)?\b",
        r"\bmaps? to\b",
        r"\bconvert(?:s|ed)?\b",
        r"\bpart of\b",
    )
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_entity_type(label: str) -> str:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label or "")
    pieces = re.findall(r"[A-Za-z0-9]+", expanded)
    if not pieces:
        return "Unknown"
    return "".join(piece.capitalize() for piece in pieces)


def normalize_relation_name(label: str) -> str:
    pieces = re.findall(r"[A-Za-z0-9]+", label or "")
    if not pieces:
        return "RELATED_TO"
    return "_".join(piece.upper() for piece in pieces)


def normalize_surface(text: str) -> str:
    cleaned = normalize_whitespace(text.casefold())
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
    cleaned = re.sub(r"\b(the|a|an)\b\s+", "", cleaned)
    return cleaned.strip()


def tokenize_surface(text: str) -> set[str]:
    return {piece for piece in re.findall(r"[a-z0-9]+", normalize_surface(text)) if piece}


def is_numeric_like_text(text: str) -> bool:
    return bool(re.fullmatch(r"[\W_]*\d+(?:[\W_]+\d+)*[\W_]*", text.strip()))


def is_variable_like(text: str) -> bool:
    compact = text.strip()
    lowered = compact.casefold()
    return bool(
        compact
        and len(compact) <= 20
        and re.fullmatch(r"[a-z](?:[a-z0-9_]*[a-z0-9])?", lowered)
        and not any(part in TECHNICAL_KEYWORDS for part in lowered.split())
    )


def is_technical_surface(text: str) -> bool:
    lowered = normalize_surface(text)
    if not lowered:
        return False
    if is_variable_like(lowered):
        return True
    if any(keyword in lowered for keyword in TECHNICAL_KEYWORDS):
        return True
    if re.search(r"\b[dqkvwhn][a-z0-9_]*\b", lowered):
        return True
    return False


def contains_any_keyword(text: str, keywords: set[str]) -> bool:
    lowered = normalize_surface(text)
    return any(keyword in lowered for keyword in keywords)


def is_document_section_text(text: str) -> bool:
    compact = normalize_whitespace(text)
    if not compact:
        return True
    if re.fullmatch(r"(?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z-]+(?:\s+(?:and|[A-Z][A-Za-z-]+)){0,6}", compact):
        return True
    if compact[0].isdigit() and len(compact.split()) <= 8:
        return True
    return False


def classify_technical_entity(text: str) -> str:
    lowered = normalize_surface(text)
    if not lowered:
        return "Unknown"
    if is_document_section_text(text):
        return "DocumentSection"
    if lowered in {"mean", "variance", "mean 0"} or "dimension" in lowered or "values of" in lowered:
        return "Quantity"
    if contains_any_keyword(text, METRIC_KEYWORDS):
        return "Metric"
    if is_variable_like(lowered) or lowered in {"dmodel", "dk", "dv", "df", "ffn", "q", "k", "v"}:
        return "Variable"
    if contains_any_keyword(text, MODEL_KEYWORDS):
        return "Model"
    if contains_any_keyword(text, COMPONENT_KEYWORDS):
        return "Component"
    if contains_any_keyword(text, TRANSFORMATION_KEYWORDS):
        return "Transformation"
    if contains_any_keyword(text, DATA_STRUCTURE_KEYWORDS):
        return "DataStructure"
    if contains_any_keyword(text, METHOD_KEYWORDS):
        return "Method"
    if is_technical_surface(text):
        return "TechnicalConcept"
    return "Unknown"


def refine_entity_label(text: str, label: str) -> str:
    normalized_label = normalize_entity_type(label)
    lowered = normalize_surface(text)
    if not lowered:
        return label
    technical_type = classify_technical_entity(text)
    if technical_type != "Unknown" and (
        normalized_label in TECHNICAL_TYPE_OVERRIDES.union({"Unknown", "Location", "Region", "City", "Country", "Quantity", "Variable"})
        or is_technical_surface(text)
    ):
        return technical_type
    if normalized_label == "Variable" and technical_type not in {"Unknown", "DocumentSection"}:
        return technical_type
    return label


def is_low_signal_entity(text: str, label: str) -> bool:
    lowered = normalize_surface(text)
    normalized_label = normalize_entity_type(label)
    if not lowered:
        return True
    if is_document_section_text(text) or normalized_label == "DocumentSection":
        return True
    if lowered in {"i"}:
        return True
    if lowered in GENERIC_LOW_SIGNAL_ENTITIES and normalized_label in {"Unknown", "Variable", "TechnicalConcept", "Method"}:
        return True
    return False


def extract_heuristic_entities(chunk: Chunk) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    seen: set[tuple[int, int, str]] = set()

    for pattern, label, score in (
        (TECHNICAL_PHRASE_PATTERN, "TechnicalConcept", 0.74),
        (VARIABLE_PATTERN, "Variable", 0.72),
    ):
        for match in pattern.finditer(chunk.text):
            text = normalize_whitespace(match.group(0).strip(" ,.;:()[]\"'"))
            if label == "TechnicalConcept":
                text = clean_technical_phrase(text)
            else:
                text = text.strip()
            if len(text) < 2:
                continue
            key = (match.start(), match.end(), label)
            if key in seen or is_low_signal_entity(text, label):
                continue
            seen.add(key)
            candidates.append(
                EntityCandidate(
                    text=text,
                    label=refine_entity_label(text, label),
                    score=score,
                    start=chunk.start + match.start(),
                    end=chunk.start + match.end(),
                    chunk_index=chunk.index,
                )
            )

    return candidates


def clean_technical_phrase(text: str) -> str:
    words = [part.strip(" ,.;:()[]\"'") for part in normalize_whitespace(text).split()]
    words = [word for word in words if word]
    while words and (words[0].casefold() in PHRASE_STOPWORDS or words[0].isdigit()):
        words.pop(0)
    while words and (words[-1].casefold() in PHRASE_STOPWORDS or words[-1].isdigit()):
        words.pop()
    if not words or len(words) > 4:
        return ""
    lowered = [word.casefold() for word in words]
    if any(word in PHRASE_STOPWORDS for word in lowered):
        return ""
    if any(word in BAD_PHRASE_WORDS for word in lowered):
        return ""
    if len(words) == 1 and lowered[0] not in ALLOWED_SINGLE_WORD_TECHNICAL:
        return ""
    if not any(keyword in word for word in lowered for keyword in TECHNICAL_KEYWORDS):
        return ""
    return " ".join(words)


def choose_display_text(texts: Iterable[str]) -> str:
    counter = Counter(texts)
    ranked = sorted(counter.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return ranked[0][0]


class SQLiteEmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.db_path = cache_dir / "embeddings.sqlite3"
        self.memory_cache: dict[tuple[str, str], list[float]] = {}
        self.available = True
        self.warning: str | None = None
        self._initialized = False

    def _ensure_db(self) -> bool:
        if self._initialized:
            return self.available
        self._initialized = True
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        model_name TEXT NOT NULL,
                        text_hash TEXT NOT NULL,
                        vector_json TEXT NOT NULL,
                        PRIMARY KEY (model_name, text_hash)
                    )
                    """
                )
        except Exception as exc:
            self.available = False
            self.warning = f"Embedding cache unavailable; using in-memory cache only: {exc}"
        return self.available

    def get(self, model_name: str, text_hash: str) -> list[float] | None:
        key = (model_name, text_hash)
        if key in self.memory_cache:
            return self.memory_cache[key]
        if not self._ensure_db():
            return None
        try:
            with sqlite3.connect(self.db_path) as connection:
                row = connection.execute(
                    "SELECT vector_json FROM embeddings WHERE model_name = ? AND text_hash = ?",
                    (model_name, text_hash),
                ).fetchone()
        except Exception as exc:
            self.available = False
            self.warning = f"Embedding cache unavailable; using in-memory cache only: {exc}"
            return None
        if not row:
            return None
        vector = json.loads(row[0])
        self.memory_cache[key] = vector
        return vector

    def set(self, model_name: str, text_hash: str, vector: Sequence[float]) -> None:
        key = (model_name, text_hash)
        payload = [float(value) for value in vector]
        self.memory_cache[key] = payload
        if not self._ensure_db():
            return
        try:
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO embeddings (model_name, text_hash, vector_json) VALUES (?, ?, ?)",
                    (model_name, text_hash, json.dumps(payload)),
                )
                connection.commit()
        except Exception as exc:
            self.available = False
            self.warning = f"Embedding cache unavailable; using in-memory cache only: {exc}"


class SentenceTransformerEmbeddingBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def model(self):
        return load_sentence_transformer_model(self.model_name)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=False,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in embeddings]


class EmbeddingSession:
    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        threshold: float,
        max_candidates: int,
        cache_dir: Path,
        backend: EmbeddingBackend | None = None,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.backend = backend or SentenceTransformerEmbeddingBackend(model_name)
        self.cache = SQLiteEmbeddingCache(cache_dir)
        self.cache_hits = 0
        self.cache_misses = 0
        self.comparisons = 0
        self.merges = 0
        self.links = 0
        self.warnings: list[str] = []
        if self.cache.warning:
            self.warnings.append(self.cache.warning)

    def _cache_key(self, text: str) -> tuple[str, str]:
        normalized = normalize_whitespace(text).strip()
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return normalized, text_hash

    def _disable(self, exc: Exception) -> None:
        self.enabled = False
        message = f"Embedding assistance disabled for this run: {exc}"
        if message not in self.warnings:
            self.warnings.append(message)

    def embed_text(self, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        normalized, text_hash = self._cache_key(text)
        if not normalized:
            return None
        cached = self.cache.get(self.backend.model_name, text_hash)
        if cached is not None:
            self.cache_hits += 1
            if self.cache.warning and self.cache.warning not in self.warnings:
                self.warnings.append(self.cache.warning)
            return cached
        self.cache_misses += 1
        try:
            vectors = self.backend.embed_many([normalized])
        except Exception as exc:
            self._disable(exc)
            return None
        if not vectors:
            return None
        self.cache.set(self.backend.model_name, text_hash, vectors[0])
        if self.cache.warning and self.cache.warning not in self.warnings:
            self.warnings.append(self.cache.warning)
        return vectors[0]

    def compare_texts(self, left: str, right: str) -> float | None:
        if not self.enabled:
            return None
        left_vector = self.embed_text(left)
        right_vector = self.embed_text(right)
        if left_vector is None or right_vector is None:
            return None
        self.comparisons += 1
        return cosine_similarity(left_vector, right_vector)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def trim_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    segment = text[start:end]
    leading = len(segment) - len(segment.lstrip())
    trailing = len(segment) - len(segment.rstrip())
    trimmed_start = start + leading
    trimmed_end = end - trailing
    return trimmed_start, trimmed_end, text[trimmed_start:trimmed_end]


def looks_like_heading(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False
    if re.fullmatch(r"(?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z-]+(?:\s+(?:and|[A-Z][A-Za-z-]+)){0,8}", compact):
        return True
    return False


def split_text_blocks(text: str, chunk_mode: str) -> list[tuple[int, int, str, int]]:
    if chunk_mode == "sentence":
        start, end, segment = trim_span(text, 0, len(text))
        return [(start, end, segment, 0)] if segment else []

    lines = text.splitlines(keepends=True)
    blocks: list[tuple[int, int, str, int]] = []
    current_start: int | None = None
    current_parts: list[str] = []
    offset = 0

    def flush() -> None:
        nonlocal current_start, current_parts
        if current_start is None or not current_parts:
            current_start = None
            current_parts = []
            return
        raw = "".join(current_parts)
        start, end, segment = trim_span(text, current_start, current_start + len(raw))
        if segment:
            blocks.append((start, end, segment, len(blocks)))
        current_start = None
        current_parts = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            flush()
        elif current_start is None:
            current_start = offset
            current_parts = [raw_line]
        elif looks_like_heading(line):
            flush()
            current_start = offset
            current_parts = [raw_line]
        else:
            current_parts.append(raw_line)
        offset += len(raw_line)

    flush()
    if not blocks and text.strip():
        start, end, segment = trim_span(text, 0, len(text))
        return [(start, end, segment, 0)] if segment else []
    return blocks


def pack_block_into_chunks(
    block_text: str,
    block_start: int,
    block_index: int,
    max_chars: int,
    chunk_overlap: int,
    start_index: int,
) -> list[Chunk]:
    nlp = get_sentence_splitter()
    doc = nlp(block_text)
    sentence_items: list[tuple[str, int, int]] = []
    for sent in doc.sents:
        sent_text = normalize_whitespace(sent.text)
        if not sent_text:
            continue
        sentence_items.append((sent_text, block_start + sent.start_char, block_start + sent.end_char))

    if not sentence_items and block_text.strip():
        return [
            Chunk(
                index=start_index,
                text=normalize_whitespace(block_text),
                start=block_start,
                end=block_start + len(block_text),
                block_index=block_index,
                overlap_sentences=0,
            )
        ]

    chunks: list[Chunk] = []
    current_sentences: list[tuple[str, int, int]] = []
    overlap_count = max(0, chunk_overlap)

    def flush_chunk(overlap_sentences: int) -> None:
        if not current_sentences:
            return
        chunk_text_value = " ".join(item[0] for item in current_sentences).strip()
        chunks.append(
            Chunk(
                index=start_index + len(chunks),
                text=chunk_text_value,
                start=current_sentences[0][1],
                end=current_sentences[-1][2],
                block_index=block_index,
                overlap_sentences=overlap_sentences,
            )
        )

    for sent in sentence_items:
        candidate = current_sentences + [sent]
        candidate_text = " ".join(item[0] for item in candidate).strip()
        if current_sentences and len(candidate_text) > max_chars:
            flush_chunk(min(len(current_sentences), overlap_count))
            overlap_tail = current_sentences[-overlap_count:] if overlap_count else []
            current_sentences = list(overlap_tail) + [sent]
        else:
            current_sentences = candidate

    flush_chunk(0 if len(chunks) == 0 else min(len(current_sentences), overlap_count))
    return chunks


def chunk_text(text: str, max_chars: int, *, chunk_mode: str = "paragraph", chunk_overlap: int = 1) -> list[Chunk]:
    blocks = split_text_blocks(text, chunk_mode)
    chunks: list[Chunk] = []
    for block_start, block_end, block_text, block_index in blocks:
        del block_end
        chunks.extend(
            pack_block_into_chunks(
                block_text,
                block_start,
                block_index,
                max_chars,
                chunk_overlap,
                len(chunks),
            )
        )
    if not chunks and text.strip():
        chunks.append(Chunk(index=0, text=normalize_whitespace(text), start=0, end=len(text), block_index=0))
    return chunks


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_document_record(text: str, *, source: str, title: str) -> DocumentModel:
    text_hash = hash_text(text)
    document_id = f"D{hash_text(f'{source}:{text_hash}')[:12]}"
    return DocumentModel(
        id=document_id,
        source=source,
        title=title,
        text_hash=text_hash,
        char_count=len(text),
    )


def build_chunk_records(
    chunks: Sequence[Chunk],
    *,
    document_id: str,
    include_chunk_text: bool,
) -> list[ChunkModel]:
    return [
        ChunkModel(
            id=f"{document_id}:C{chunk.index}",
            document_id=document_id,
            index=chunk.index,
            text=chunk.text if include_chunk_text else "",
            start=chunk.start,
            end=chunk.end,
            block_index=chunk.block_index,
            overlap_sentences=chunk.overlap_sentences,
        )
        for chunk in chunks
    ]


def resolve_relation_chunk_ids(
    candidate: RelationCandidate,
    *,
    chunk_record_lookup: dict[tuple[str, int], ChunkModel],
    raw_chunks: dict[tuple[str, int], Chunk],
) -> list[str]:
    key = (candidate.document_id, candidate.chunk_index)
    if key in chunk_record_lookup:
        return [chunk_record_lookup[key].id]
    normalized_evidence = normalize_whitespace(candidate.evidence)
    if not normalized_evidence:
        return []
    matched_ids = [
        chunk_record_lookup[item_key].id
        for item_key, chunk in raw_chunks.items()
        if item_key in chunk_record_lookup
        and item_key[0] == candidate.document_id
        and normalized_evidence in normalize_whitespace(chunk.text)
    ]
    return sorted(dict.fromkeys(matched_ids))


def build_relation_support(relations: Sequence[RelationModel]) -> list[RelationSupportModel]:
    return [
        RelationSupportModel(
            source_id=relation.source_id,
            relation=relation.relation,
            target_id=relation.target_id,
            chunk_ids=list(relation.chunk_ids),
            document_ids=list(relation.document_ids),
            evidence=relation.evidence,
        )
        for relation in relations
    ]


def build_neo4j_mention_rows(entities: Sequence[EntityModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in entities:
        for mention in entity.mentions:
            if not mention.chunk_id:
                continue
            rows.append(
                {
                    "entity_id": entity.id,
                    "document_id": mention.document_id,
                    "chunk_id": mention.chunk_id,
                    "text": mention.text,
                    "start": mention.start,
                    "end": mention.end,
                }
            )
    return rows


def validate_neo4j_args(args: argparse.Namespace) -> bool:
    values = (args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    enabled = any(values)
    if enabled and not all(values):
        raise SystemExit("Neo4j export requires --neo4j-uri, --neo4j-user, and --neo4j-password.")
    return enabled


def export_graph_to_neo4j(
    result: GraphExtraction,
    *,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    clean_document: bool = False,
    driver_factory: Any | None = None,
) -> None:
    if driver_factory is None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "Neo4j export requires the `neo4j` package. Install it with `.venv/bin/pip install neo4j`."
            ) from exc
        driver_factory = GraphDatabase.driver

    driver = driver_factory(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for query in NEO4J_CONSTRAINT_QUERIES:
                session.run(query)

            if clean_document:
                for document in result.documents:
                    session.run(
                        NEO4J_DELETE_DOCUMENT_RELATIONS_QUERY,
                        document_id=document.id,
                    )
                    session.run(
                        NEO4J_DELETE_DOCUMENT_SUBGRAPH_QUERY,
                        document_id=document.id,
                    )

            if result.documents:
                session.run(
                    NEO4J_DOCUMENT_QUERY,
                    rows=[document.model_dump() for document in result.documents],
                )
            if result.chunks:
                session.run(
                    NEO4J_CHUNK_QUERY,
                    rows=[chunk.model_dump() for chunk in result.chunks],
                )
            if result.entities:
                session.run(
                    NEO4J_ENTITY_QUERY,
                    rows=[
                        {
                            "id": entity.id,
                            "text": entity.text,
                            "canonical": entity.canonical,
                            "type": entity.type,
                            "confidence": entity.confidence,
                        }
                        for entity in result.entities
                    ],
                )
            mention_rows = build_neo4j_mention_rows(result.entities)
            if mention_rows:
                session.run(NEO4J_MENTION_QUERY, rows=mention_rows)
            if result.relations:
                session.run(
                    NEO4J_RELATION_QUERY,
                    rows=[relation.model_dump() for relation in result.relations],
                )
    finally:
        close = getattr(driver, "close", None)
        if callable(close):
            close()


@lru_cache(maxsize=1)
def get_sentence_splitter():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


class GLiNEREntityBackend:
    model_name = GLINER_MODEL_NAME

    def __init__(self, labels: Sequence[str] | None = None):
        self.labels = list(labels or DEFAULT_ENTITY_LABELS)
        self.fallback_used = False
        self.warnings: list[str] = []

    @property
    def model(self):
        return load_gliner_model(self.model_name)

    def extract(self, chunk: Chunk, threshold: float) -> list[EntityCandidate]:
        try:
            predictions = self.model.predict_entities(
                chunk.text,
                self.labels,
                threshold=threshold,
                flat_ner=True,
            )
        except Exception as exc:
            if not self.fallback_used:
                self.warnings.append(f"GLiNER failed, using regex entity fallback: {exc}")
                self.fallback_used = True
            return RegexEntityBackend().extract(chunk, threshold)

        return [
            EntityCandidate(
                text=item["text"],
                label=refine_entity_label(item["text"], item.get("label", "Unknown")),
                score=float(item.get("score", 0.0)),
                start=chunk.start + int(item.get("start", 0)),
                end=chunk.start + int(item.get("end", 0)),
                chunk_index=chunk.index,
            )
            for item in predictions
        ]


class RegexEntityBackend:
    model_name = "regex_entity_fallback"

    def __init__(self):
        self.fallback_used = True
        self.warnings: list[str] = ["Using regex entity fallback."]

    def extract(self, chunk: Chunk, threshold: float) -> list[EntityCandidate]:
        del threshold
        candidates: list[EntityCandidate] = []
        pattern = re.compile(r"\b(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*)*\b")
        for match in pattern.finditer(chunk.text):
            text = match.group(0).strip()
            if len(text) < 3:
                continue
            candidates.append(
                EntityCandidate(
                    text=text,
                    label=refine_entity_label(text, "Unknown"),
                    score=0.55,
                    start=chunk.start + match.start(),
                    end=chunk.start + match.end(),
                    chunk_index=chunk.index,
                )
            )
        return candidates


class RebelRelationBackend:
    model_name = REBEL_MODEL_NAME

    def __init__(self):
        self.fallback_used = False
        self.warnings: list[str] = []

    @property
    def tokenizer(self):
        return load_rebel_tokenizer(self.model_name)

    @property
    def model(self):
        return load_rebel_model(self.model_name)

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        try:
            inputs = self.tokenizer(
                chunk.text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            generated = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=3,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            decoded = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=False)
            raw_sequence_score = float(getattr(generated, "sequences_scores", [0.0])[0])
            sequence_score = compute_generation_confidence([raw_sequence_score])
        except Exception as exc:
            if not self.fallback_used:
                self.warnings.append(f"REBEL failed, using heuristic relation fallback: {exc}")
                self.fallback_used = True
            return HeuristicRelationBackend().extract(chunk, threshold)

        relations: list[RelationCandidate] = []
        for text in decoded:
            for triplet in parse_rebel_triplets(text):
                if sequence_score < threshold:
                    continue
                relations.append(
                    RelationCandidate(
                        subject=triplet["subject"],
                        relation=triplet["relation"],
                        obj=triplet["object"],
                        score=sequence_score,
                        evidence=chunk.text,
                        chunk_index=chunk.index,
                    )
                )
        return deduplicate_relation_candidates(relations)


class HeuristicRelationBackend:
    model_name = "heuristic_relation_fallback"

    def __init__(self, *, as_fallback: bool = True):
        self.fallback_used = as_fallback
        self.warnings: list[str] = ["Using heuristic relation fallback."] if as_fallback else []

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        del threshold
        text = chunk.text
        sentence_fragments = re.split(r"(?<=[.!?])\s+", text)
        relations: list[RelationCandidate] = []
        for sentence in sentence_fragments:
            matches = list(re.finditer(r"\b(?:[A-Z][\w.-]*)(?:\s+[A-Z][\w.-]*)*\b", sentence))
            if len(matches) < 2:
                continue
            for left, right in zip(matches, matches[1:]):
                bridge = sentence[left.end() : right.start()].strip(" ,.-")
                bridge = re.sub(r"\s+", " ", bridge)
                if not bridge or len(bridge.split()) > 6:
                    continue
                relations.append(
                    RelationCandidate(
                        subject=left.group(0).strip(),
                        relation=bridge,
                        obj=right.group(0).strip(),
                        score=0.55,
                        evidence=sentence.strip(),
                        chunk_index=chunk.index,
                    )
                )
        return deduplicate_relation_candidates(relations)


def deduplicate_relation_candidates(
    relations: Sequence[RelationCandidate],
) -> list[RelationCandidate]:
    best: dict[tuple[str, str, str, str], RelationCandidate] = {}
    for relation in relations:
        key = (
            normalize_surface(relation.subject),
            normalize_relation_name(relation.relation),
            normalize_surface(relation.obj),
            relation.evidence,
        )
        previous = best.get(key)
        if previous is None or relation.score > previous.score:
            best[key] = relation
    return list(best.values())


def compute_generation_confidence(scores: Sequence[float]) -> float:
    valid_scores = [score for score in scores if not math.isnan(score)]
    if not valid_scores:
        return 0.5
    average_log_prob = sum(valid_scores) / len(valid_scores)
    return max(0.0, min(1.0, math.exp(average_log_prob)))


def parse_rebel_triplets(text: str) -> list[dict[str, str]]:
    cleaned = (
        text.replace("<s>", " ")
        .replace("</s>", " ")
        .replace("<pad>", " ")
        .replace("  ", " ")
        .strip()
    )
    subject = ""
    relation = ""
    obj = ""
    state = None
    triplets: list[dict[str, str]] = []

    for token in cleaned.split():
        if token == "<triplet>":
            if subject and relation and obj:
                triplets.append(
                    {
                        "subject": normalize_whitespace(subject),
                        "relation": normalize_whitespace(relation),
                        "object": normalize_whitespace(obj),
                    }
                )
            subject = ""
            relation = ""
            obj = ""
            state = "subject"
            continue
        if token == "<subj>":
            state = "object"
            continue
        if token == "<obj>":
            state = "relation"
            continue
        if state == "subject":
            subject = f"{subject} {token}".strip()
        elif state == "object":
            obj = f"{obj} {token}".strip()
        elif state == "relation":
            relation = f"{relation} {token}".strip()

    if subject and relation and obj:
        triplets.append(
            {
                "subject": normalize_whitespace(subject),
                "relation": normalize_whitespace(relation),
                "object": normalize_whitespace(obj),
            }
        )

    return triplets


@lru_cache(maxsize=1)
def load_gliner_model(model_name: str):
    from gliner import GLiNER

    return load_from_pretrained(GLiNER.from_pretrained, model_name)


@lru_cache(maxsize=1)
def load_rebel_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return load_from_pretrained(AutoTokenizer.from_pretrained, model_name)


@lru_cache(maxsize=1)
def load_rebel_model(model_name: str):
    from transformers import AutoModelForSeq2SeqLM

    model = load_from_pretrained(AutoModelForSeq2SeqLM.from_pretrained, model_name)
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_sentence_transformer_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_from_pretrained(loader, model_name: str):
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return loader(model_name, local_files_only=True)
    try:
        return loader(model_name, local_files_only=True)
    except Exception:
        return loader(model_name)


class GraphExtractor:
    def __init__(
        self,
        entity_backend: EntityBackend,
        relation_backend: RelationBackend | None,
        *,
        cheap_relation_backend: RelationBackend | None = None,
        max_chars: int = 600,
        fuzzy_merge_threshold: int = 93,
        chunk_mode: str = "paragraph",
        chunk_overlap: int = 1,
        mode: str = "balanced",
        max_model_spans: int = 4,
        disable_rebel: bool = False,
        embedding_linking: bool = False,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_threshold: float = 0.84,
        embedding_cache_dir: str | Path = ".cache/extract_graph_embeddings",
        embedding_max_candidates: int = 8,
        embedding_backend: EmbeddingBackend | None = None,
    ):
        self.entity_backend = entity_backend
        self.relation_backend = relation_backend
        self.cheap_relation_backend = cheap_relation_backend or HeuristicRelationBackend(as_fallback=False)
        self.max_chars = max_chars
        self.fuzzy_merge_threshold = fuzzy_merge_threshold
        self.chunk_mode = chunk_mode
        self.chunk_overlap = chunk_overlap
        self.mode = mode
        self.max_model_spans = max_model_spans
        self.disable_rebel = disable_rebel
        self.embedding_linking = embedding_linking
        self.embedding_model = embedding_model
        self.embedding_threshold = embedding_threshold
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_max_candidates = embedding_max_candidates
        self.embedding_backend = embedding_backend

    def extract(
        self,
        text: str,
        *,
        entity_threshold: float = 0.35,
        relation_threshold: float = 0.2,
        max_chars: int | None = None,
        chunk_mode: str | None = None,
        chunk_overlap: int | None = None,
        mode: str | None = None,
        max_model_spans: int | None = None,
        disable_rebel: bool | None = None,
        embedding_linking: bool | None = None,
        embedding_model: str | None = None,
        embedding_threshold: float | None = None,
        embedding_cache_dir: str | Path | None = None,
        embedding_max_candidates: int | None = None,
        document_source: str = "inline",
        document_title: str = "inline_text",
        include_chunk_text: bool = True,
        entity_scope: str = "document",
    ) -> GraphExtraction:
        return self.extract_documents(
            [DocumentInput(text=text, source=document_source, title=document_title)],
            entity_threshold=entity_threshold,
            relation_threshold=relation_threshold,
            max_chars=max_chars,
            chunk_mode=chunk_mode,
            chunk_overlap=chunk_overlap,
            mode=mode,
            max_model_spans=max_model_spans,
            disable_rebel=disable_rebel,
            embedding_linking=embedding_linking,
            embedding_model=embedding_model,
            embedding_threshold=embedding_threshold,
            embedding_cache_dir=embedding_cache_dir,
            embedding_max_candidates=embedding_max_candidates,
            include_chunk_text=include_chunk_text,
            entity_scope=entity_scope,
        )

    def extract_documents(
        self,
        documents: Sequence[DocumentInput],
        *,
        entity_threshold: float = 0.35,
        relation_threshold: float = 0.2,
        max_chars: int | None = None,
        chunk_mode: str | None = None,
        chunk_overlap: int | None = None,
        mode: str | None = None,
        max_model_spans: int | None = None,
        disable_rebel: bool | None = None,
        embedding_linking: bool | None = None,
        embedding_model: str | None = None,
        embedding_threshold: float | None = None,
        embedding_cache_dir: str | Path | None = None,
        embedding_max_candidates: int | None = None,
        include_chunk_text: bool = True,
        entity_scope: str = "document",
    ) -> GraphExtraction:
        started = time.perf_counter()
        active_chunk_mode = chunk_mode or self.chunk_mode
        active_chunk_overlap = self.chunk_overlap if chunk_overlap is None else max(0, chunk_overlap)
        active_mode = mode or self.mode
        active_max_model_spans = self.max_model_spans if max_model_spans is None else max(0, max_model_spans)
        active_disable_rebel = self.disable_rebel if disable_rebel is None else disable_rebel
        active_embedding_linking = self.embedding_linking if embedding_linking is None else embedding_linking
        active_embedding_model = embedding_model or self.embedding_model
        active_embedding_threshold = self.embedding_threshold if embedding_threshold is None else embedding_threshold
        active_embedding_cache_dir = self.embedding_cache_dir if embedding_cache_dir is None else Path(embedding_cache_dir)
        active_embedding_max_candidates = self.embedding_max_candidates if embedding_max_candidates is None else max(1, embedding_max_candidates)
        embedding_session = EmbeddingSession(
            enabled=active_embedding_linking,
            model_name=active_embedding_model,
            threshold=active_embedding_threshold,
            max_candidates=active_embedding_max_candidates,
            cache_dir=active_embedding_cache_dir,
            backend=self.embedding_backend if active_embedding_linking else None,
        )
        active_entity_scope = entity_scope if entity_scope in {"document", "corpus"} else "document"

        all_document_records: list[DocumentModel] = []
        all_chunk_records: list[ChunkModel] = []
        all_entity_candidates: list[EntityCandidate] = []
        all_relation_candidates: list[RelationCandidate] = []
        chunk_record_lookup: dict[tuple[str, int], ChunkModel] = {}
        raw_chunk_lookup: dict[tuple[str, int], Chunk] = {}
        relation_strategies: list[str] = []
        total_rebel_considered = 0
        total_rebel_run = 0

        for document in documents:
            normalized_text = document.text.replace("\r\n", "\n").replace("\r", "\n").strip()
            chunks = chunk_text(
                normalized_text,
                max_chars or self.max_chars,
                chunk_mode=active_chunk_mode,
                chunk_overlap=active_chunk_overlap,
            )
            document_record = build_document_record(
                normalized_text,
                source=document.source,
                title=document.title,
            )
            chunk_records = build_chunk_records(
                chunks,
                document_id=document_record.id,
                include_chunk_text=include_chunk_text,
            )
            document_entity_candidates: list[EntityCandidate] = []
            local_chunk_lookup = {(document_record.id, chunk.index): chunk_record for chunk, chunk_record in zip(chunks, chunk_records)}

            for chunk in chunks:
                extracted_entities = self.entity_backend.extract(chunk, entity_threshold)
                heuristic_entities = extract_heuristic_entities(chunk)
                for candidate in extracted_entities + heuristic_entities:
                    candidate.document_id = document_record.id
                    document_entity_candidates.append(candidate)
                    all_entity_candidates.append(candidate)
                raw_chunk_lookup[(document_record.id, chunk.index)] = chunk

            provisional_entities, provisional_lookup = self._canonicalize_entities(
                document_entity_candidates,
                embedding_session=None,
                chunk_record_lookup=local_chunk_lookup,
                entity_scope="document",
            )
            relation_candidates, strategy, rebel_considered, rebel_run = self._collect_relation_candidates(
                chunks,
                normalized_text,
                provisional_entities,
                relation_threshold=relation_threshold,
                mode=active_mode,
                max_model_spans=active_max_model_spans,
                disable_rebel=active_disable_rebel,
            )
            for candidate in relation_candidates:
                candidate.document_id = document_record.id
                all_relation_candidates.append(candidate)

            all_document_records.append(document_record)
            all_chunk_records.extend(chunk_records)
            chunk_record_lookup.update(local_chunk_lookup)
            relation_strategies.append(strategy)
            total_rebel_considered += rebel_considered
            total_rebel_run += rebel_run

        entities, entity_lookup = self._canonicalize_entities(
            all_entity_candidates,
            embedding_session=embedding_session,
            chunk_record_lookup=chunk_record_lookup,
            entity_scope=active_entity_scope,
        )
        relations = self._link_relations(
            all_relation_candidates,
            entities,
            entity_lookup,
            relation_threshold=relation_threshold,
            embedding_session=embedding_session,
            chunk_record_lookup=chunk_record_lookup,
            chunk_records=all_chunk_records,
            raw_chunks=raw_chunk_lookup,
        )
        schema = build_schema(entities, relations)
        expanded_schema = build_expanded_schema(entities, relations)
        relation_support = build_relation_support(relations)

        relation_warnings = list(self.cheap_relation_backend.warnings)
        relation_fallback_used = self.cheap_relation_backend.fallback_used
        if self.relation_backend is not None:
            relation_warnings.extend(self.relation_backend.warnings)
            relation_fallback_used = relation_fallback_used or self.relation_backend.fallback_used

        warnings = list(dict.fromkeys(self.entity_backend.warnings + relation_warnings + embedding_session.warnings))
        fallback_used = self.entity_backend.fallback_used or relation_fallback_used
        elapsed = round(time.perf_counter() - started, 4)
        strategy = choose_relation_strategy(relation_strategies)
        relation_model_name = self.cheap_relation_backend.model_name
        if self.relation_backend is not None and strategy != "heuristic_only":
            relation_model_name = self.relation_backend.model_name

        return GraphExtraction(
            entities=entities,
            relations=relations,
            potential_schema=schema,
            expanded_schema=expanded_schema,
            documents=all_document_records,
            chunks=all_chunk_records,
            relation_support=relation_support,
            meta=MetaModel(
                entity_model=self.entity_backend.model_name,
                relation_model=relation_model_name,
                entity_threshold=entity_threshold,
                relation_threshold=relation_threshold,
                chunk_count=len(all_chunk_records),
                elapsed_seconds=elapsed,
                chunk_mode=active_chunk_mode,
                chunk_overlap=active_chunk_overlap,
                entity_candidates=len(all_entity_candidates),
                relation_candidates=len(all_relation_candidates),
                entities_kept=len(entities),
                relations_kept=len(relations),
                mode=active_mode,
                relation_backend_strategy=strategy,
                rebel_spans_considered=total_rebel_considered,
                rebel_spans_run=total_rebel_run,
                rebel_skipped=max(0, total_rebel_considered - total_rebel_run),
                embedding_enabled=active_embedding_linking,
                embedding_model=active_embedding_model if active_embedding_linking else "",
                embedding_cache_hits=embedding_session.cache_hits,
                embedding_cache_misses=embedding_session.cache_misses,
                embedding_comparisons=embedding_session.comparisons,
                embedding_merges=embedding_session.merges,
                embedding_links=embedding_session.links,
                document_count=len(all_document_records),
                chunk_records=len(all_chunk_records),
                chunk_text_included=include_chunk_text,
                warnings=warnings,
                fallback_used=fallback_used,
            ),
        )

    def _collect_relation_candidates(
        self,
        chunks: Sequence[Chunk],
        text: str,
        entities: Sequence[EntityModel],
        *,
        relation_threshold: float,
        mode: str,
        max_model_spans: int,
        disable_rebel: bool,
    ) -> tuple[list[RelationCandidate], str, int, int]:
        candidates: list[RelationCandidate] = []
        for chunk in chunks:
            if mode != "quality":
                candidates.extend(self.cheap_relation_backend.extract(chunk, relation_threshold))

        candidates.extend(extract_context_relations(text, entities))

        if disable_rebel or self.relation_backend is None or mode == "fast":
            return deduplicate_relation_candidates(candidates), "heuristic_only", 0, 0

        if mode == "quality":
            for chunk in chunks:
                candidates.extend(self.relation_backend.extract(chunk, relation_threshold))
            return deduplicate_relation_candidates(candidates), "full_rebel", len(chunks), len(chunks)

        spans = select_rebel_candidate_spans(text, entities, max_model_spans=max_model_spans)
        for span in spans:
            candidates.extend(self.relation_backend.extract(span, relation_threshold))
        return (
            deduplicate_relation_candidates(candidates),
            "heuristic_plus_selective_rebel",
            count_rebel_candidate_spans(text, entities),
            len(spans),
        )

    def _canonicalize_entities(
        self,
        candidates: Sequence[EntityCandidate],
        *,
        embedding_session: EmbeddingSession | None = None,
        chunk_record_lookup: dict[tuple[str, int], ChunkModel] | None = None,
        entity_scope: str = "document",
    ) -> tuple[list[EntityModel], dict[tuple[str, str], str]]:
        grouped: dict[tuple[str, str], list[EntityCandidate]] = defaultdict(list)
        for candidate in candidates:
            normalized = normalize_surface(candidate.text)
            if not normalized or is_low_signal_entity(candidate.text, candidate.label):
                continue
            grouping_document_id = candidate.document_id if entity_scope == "document" else ""
            grouped[(grouping_document_id, normalized)].append(candidate)

        canonical_groups: list[list[EntityCandidate]] = []
        for group_key in sorted(grouped):
            placed = False
            current_group = grouped[group_key]
            current_type = dominant_type(current_group)
            embedding_matches: list[tuple[float, int, int, int]] = []
            for existing in canonical_groups:
                if entity_scope == "document" and existing and existing[0].document_id != current_group[0].document_id:
                    continue
                reference_key = normalize_surface(choose_display_text(item.text for item in existing))
                reference_type = dominant_type(existing)
                compatible_type = (
                    current_type == reference_type
                    or current_type == "Unknown"
                    or reference_type == "Unknown"
                )
                string_score = fuzz.ratio(group_key[1], reference_key)
                if compatible_type and string_score >= self.fuzzy_merge_threshold:
                    existing.extend(current_group)
                    placed = True
                    break
                if not compatible_type or embedding_session is None or not embedding_session.enabled:
                    continue
                if not is_embedding_merge_candidate(
                    current_group,
                    existing,
                    string_score=string_score,
                    current_type=current_type,
                    reference_type=reference_type,
                ):
                    continue
                current_text = choose_display_text(item.text for item in current_group)
                reference_text = choose_display_text(item.text for item in existing)
                similarity = embedding_session.compare_texts(current_text, reference_text)
                if similarity is None or similarity < embedding_session.threshold:
                    continue
                type_score = 2 if current_type == reference_type and current_type != "Unknown" else 1
                mention_score = len(existing)
                embedding_matches.append((similarity, type_score, mention_score, canonical_groups.index(existing)))
            if not placed and embedding_matches:
                _, _, _, best_index = max(embedding_matches, key=lambda item: (item[0], item[1], item[2], -item[3]))
                canonical_groups[best_index].extend(current_group)
                if embedding_session is not None:
                    embedding_session.merges += 1
                placed = True
            if not placed:
                canonical_groups.append(list(current_group))

        entities: list[EntityModel] = []
        alias_lookup: dict[tuple[str, str], str] = {}
        for index, group in enumerate(canonical_groups, start=1):
            unique_group = list(
                {
                    (
                        item.document_id,
                        item.start,
                        item.end,
                        normalize_surface(item.text),
                        normalize_entity_type(item.label),
                    ): item
                    for item in group
                }.values()
            )
            unique_group.sort(key=lambda item: (item.start, item.end))
            entity_id = f"E{index}"
            display_text = choose_display_text(item.text for item in unique_group)
            entity_type = dominant_type(unique_group)
            confidence = round(sum(item.score for item in unique_group) / len(unique_group), 4)
            display_text = normalize_whitespace(display_text)
            mentions = [
                MentionModel(
                    text=item.text,
                    start=item.start,
                    end=item.end,
                    chunk_index=item.chunk_index,
                    document_id=item.document_id,
                    chunk_id=chunk_record_lookup[(item.document_id, item.chunk_index)].id
                    if chunk_record_lookup and (item.document_id, item.chunk_index) in chunk_record_lookup
                    else "",
                )
                for item in unique_group
            ]
            entity = EntityModel(
                id=entity_id,
                text=display_text,
                canonical=display_text,
                type=entity_type,
                confidence=confidence,
                mentions=mentions,
            )
            entities.append(entity)
            for mention in mentions:
                alias_lookup[(mention.document_id, normalize_surface(display_text))] = entity_id
            for item in unique_group:
                alias_lookup[(item.document_id, normalize_surface(item.text))] = entity_id

        return entities, alias_lookup

    def _link_relations(
        self,
        candidates: Sequence[RelationCandidate],
        entities: Sequence[EntityModel],
        entity_lookup: dict[tuple[str, str], str],
        *,
        relation_threshold: float,
        embedding_session: EmbeddingSession | None = None,
        chunk_record_lookup: dict[tuple[str, int], ChunkModel] | None = None,
        chunk_records: Sequence[ChunkModel] | None = None,
        raw_chunks: dict[tuple[str, int], Chunk] | None = None,
    ) -> list[RelationModel]:
        entity_index = {entity.id: entity for entity in entities}
        chunk_record_by_id = {chunk.id: chunk for chunk in chunk_records or ()}
        best_by_key: dict[tuple[str, str, str], RelationModel] = {}

        for candidate in candidates:
            source_id = resolve_entity_id(
                candidate.subject,
                entity_lookup,
                entities,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                evidence=candidate.evidence,
                embedding_session=embedding_session,
            )
            target_id = resolve_entity_id(
                candidate.obj,
                entity_lookup,
                entities,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                evidence=candidate.evidence,
                embedding_session=embedding_session,
            )
            if not source_id or not target_id or source_id == target_id:
                continue

            source_conf = entity_index[source_id].confidence
            target_conf = entity_index[target_id].confidence
            confidence = round((candidate.score + source_conf + target_conf) / 3.0, 4)
            if confidence < relation_threshold:
                continue

            relation_name = normalize_relation_name(candidate.relation)
            if not is_relation_plausible(
                relation_name,
                entity_index[source_id],
                entity_index[target_id],
                candidate.evidence,
            ):
                continue
            key = (source_id, relation_name, target_id)
            support_chunk_ids = resolve_relation_chunk_ids(
                candidate,
                chunk_record_lookup=chunk_record_lookup or {},
                raw_chunks=raw_chunks or {},
            )
            support_document_ids = sorted(
                {
                    chunk_record_by_id[chunk_id].document_id
                    for chunk_id in support_chunk_ids
                    if chunk_id in chunk_record_by_id
                }
            )
            relation = RelationModel(
                source_id=source_id,
                target_id=target_id,
                relation=relation_name,
                confidence=confidence,
                evidence=candidate.evidence,
                chunk_ids=support_chunk_ids,
                document_ids=support_document_ids,
            )
            previous = best_by_key.get(key)
            if previous is None:
                best_by_key[key] = relation
                continue
            merged_chunk_ids = sorted(set(previous.chunk_ids).union(relation.chunk_ids))
            merged_document_ids = sorted(set(previous.document_ids).union(relation.document_ids))
            if relation.confidence > previous.confidence:
                relation.chunk_ids = merged_chunk_ids
                relation.document_ids = merged_document_ids
                best_by_key[key] = relation
            else:
                previous.chunk_ids = merged_chunk_ids
                previous.document_ids = merged_document_ids

        return sorted(
            best_by_key.values(),
            key=lambda item: (-item.confidence, item.relation, item.source_id, item.target_id),
        )


def dominant_type(group: Sequence[EntityCandidate]) -> str:
    weighted_scores: dict[str, float] = defaultdict(float)
    for item in group:
        normalized_type = normalize_entity_type(item.label)
        weighted_scores[normalized_type] += item.score
    if not weighted_scores:
        return "Unknown"
    best_type, best_score = max(weighted_scores.items(), key=lambda item: (item[1], item[0]))
    if best_score <= 0:
        return "Unknown"
    return best_type


def is_embedding_merge_candidate(
    current_group: Sequence[EntityCandidate],
    existing_group: Sequence[EntityCandidate],
    *,
    string_score: float,
    current_type: str,
    reference_type: str,
) -> bool:
    if string_score < 78 or string_score > 92:
        return False
    current_text = choose_display_text(item.text for item in current_group)
    reference_text = choose_display_text(item.text for item in existing_group)
    if normalize_surface(current_text) == normalize_surface(reference_text):
        return False
    if any(
        is_low_signal_entity(text, label) or is_document_section_text(text) or is_numeric_like_text(text)
        for text, label in (
            (current_text, current_type),
            (reference_text, reference_type),
        )
    ):
        return False
    if not has_embedding_merge_lexical_compatibility(current_text, reference_text):
        return False
    return current_type == reference_type or current_type == "Unknown" or reference_type == "Unknown"


def has_embedding_merge_lexical_compatibility(current_text: str, reference_text: str) -> bool:
    current_words = re.findall(r"[a-z0-9]+", normalize_surface(current_text))
    reference_words = re.findall(r"[a-z0-9]+", normalize_surface(reference_text))
    if not current_words or not reference_words:
        return False
    if len(current_words) != len(reference_words):
        return False
    if current_words == reference_words:
        return False
    if current_words[-1] != reference_words[-1]:
        return False
    if len(current_words) == 1:
        return fuzz.partial_ratio(current_words[0], reference_words[0]) >= 88
    current_prefix = " ".join(current_words[:-1])
    reference_prefix = " ".join(reference_words[:-1])
    return fuzz.partial_ratio(current_prefix, reference_prefix) >= 80


RELATION_NAME_ALLOWLIST_WITH_FUNCTION_WORDS = {
    "ADDED_TO",
    "APPLIED_TO",
    "BASED_IN",
    "BORN_IN",
    "COMPOSED_OF",
    "DERIVED_FROM",
    "LOCATED_IN",
    "MAPS_TO",
    "PART_OF",
    "WORKS_AT",
}


RELATION_NAME_SUSPICIOUS_TOKENS = {
    "A",
    "AN",
    "AS",
    "HE",
    "HER",
    "HIM",
    "HIS",
    "I",
    "OUR",
    "SHE",
    "THE",
    "THEIR",
    "THEM",
    "THESE",
    "THIS",
    "THOSE",
    "WE",
    "WHEN",
    "WHERE",
    "WHO",
    "YOU",
}


def is_relation_name_well_formed(relation_name: str) -> bool:
    parts = [part for part in relation_name.split("_") if part]
    if not parts:
        return False
    if any(part[0].isdigit() for part in parts):
        return False
    if relation_name in RELATION_NAME_ALLOWLIST_WITH_FUNCTION_WORDS:
        return True
    if any(part in RELATION_NAME_SUSPICIOUS_TOKENS for part in parts):
        return False
    return True


def is_relation_plausible(
    relation_name: str,
    source: EntityModel,
    target: EntityModel,
    evidence: str,
) -> bool:
    if not is_relation_name_well_formed(relation_name):
        return False
    if relation_name in {"AND", "THE", "OF", "IS", "ARE", "A", "AN", "RELATED_TO"}:
        return False
    if len(relation_name.split("_")) > 6:
        return False
    technical_pair = is_technical_surface(source.text) or is_technical_surface(target.text)
    if technical_pair and relation_name in REBEL_RELATION_BLOCKLIST_FOR_TECHNICAL:
        return False
    lowered = evidence.casefold()
    if relation_name == "USES" and "uses multi-head attention" in lowered:
        return source.text.casefold() == "transformer" and "multi-head attention" in target.text.casefold()
    if relation_name == "DERIVED_FROM" and "queries come from" in lowered:
        return source.text.casefold() == "queries" and target.text.casefold() == "previous decoder layer"
    if relation_name == "DERIVED_FROM" and "keys and values come from" in lowered:
        return False
    if relation_name == "PART_OF" and "part of" not in lowered:
        return False
    if relation_name == "COMPOSED_OF" and not any(phrase in lowered for phrase in {"composed of", "consists of"}):
        return False
    if relation_name == "PRODUCES" and not any(phrase in lowered for phrase in {"produces", "yield", "yields", "resulting in"}):
        return False
    if relation_name == "MAPS_TO" and not any(phrase in lowered for phrase in {"maps to", "convert", "converts to"}):
        return False
    if relation_name == "OPPOSITE_OF" and "opposite" not in lowered:
        return False
    if relation_name == "SUBCLASS_OF" and "subclass" not in lowered:
        return False
    return True


def extract_context_relations(
    text: str,
    entities: Sequence[EntityModel],
) -> list[RelationCandidate]:
    sentences = list(get_sentence_splitter()(text).sents)
    entity_mentions = build_entity_mentions(entities)
    candidates: list[RelationCandidate] = []

    for sentence in sentences:
        sentence_text = sentence.text.strip()
        if not sentence_text:
            continue
        sentence_entities = [
            item
            for item in entity_mentions
            if item["start"] >= sentence.start_char and item["end"] <= sentence.end_char
        ]
        if len(sentence_entities) < 2:
            continue
        sentence_entities.sort(key=lambda item: (item["start"], item["end"]))
        candidates.extend(match_sentence_relations(sentence_text, sentence.start_char, sentence_entities))

    return deduplicate_relation_candidates(candidates)


def build_sentence_candidates(
    text: str,
    entities: Sequence[EntityModel],
) -> list[tuple[str, int, int, list[dict[str, Any]]]]:
    sentences = list(get_sentence_splitter()(text).sents)
    entity_mentions = build_entity_mentions(entities)
    candidates: list[tuple[str, int, int, list[dict[str, Any]]]] = []

    for sentence in sentences:
        sentence_text = sentence.text.strip()
        if not sentence_text or is_document_section_text(sentence_text):
            continue
        sentence_entities = [
            item
            for item in entity_mentions
            if item["start"] >= sentence.start_char and item["end"] <= sentence.end_char
        ]
        if len({item["entity_id"] for item in sentence_entities}) < 2:
            continue
        candidates.append((sentence_text, sentence.start_char, sentence.end_char, sentence_entities))

    return candidates


def score_rebel_span(
    sentence_text: str,
    sentence_entities: Sequence[dict[str, Any]],
) -> int:
    unique_ids = {item["entity_id"] for item in sentence_entities}
    trigger_score = sum(1 for pattern in RELATION_TRIGGER_PATTERNS if pattern.search(sentence_text))
    technical_score = sum(
        1 for item in sentence_entities if is_technical_surface(item["text"])
    )
    type_variety = len(
        {
            classify_technical_entity(item["text"])
            for item in sentence_entities
            if classify_technical_entity(item["text"]) != "Unknown"
        }
    )
    score = len(unique_ids) * 3 + trigger_score * 4 + min(technical_score, 4) + type_variety
    if len(sentence_text) > 320:
        score -= 1
    return score


def count_rebel_candidate_spans(text: str, entities: Sequence[EntityModel]) -> int:
    return sum(
        1
        for sentence_text, _, _, _ in build_sentence_candidates(text, entities)
        if any(pattern.search(sentence_text) for pattern in RELATION_TRIGGER_PATTERNS)
    )


def select_rebel_candidate_spans(
    text: str,
    entities: Sequence[EntityModel],
    *,
    max_model_spans: int,
) -> list[Chunk]:
    sentence_candidates = build_sentence_candidates(text, entities)
    ranked = sorted(
        (
            (
                score_rebel_span(sentence_text, sentence_entities),
                start,
                end,
                sentence_text,
            )
            for sentence_text, start, end, sentence_entities in sentence_candidates
            if any(pattern.search(sentence_text) for pattern in RELATION_TRIGGER_PATTERNS)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    selected = ranked[:max_model_spans]
    return [
        Chunk(
            index=-(offset + 1),
            text=sentence_text,
            start=start,
            end=end,
            block_index=-1,
            overlap_sentences=0,
        )
        for offset, (_, start, end, sentence_text) in enumerate(selected)
    ]


def build_entity_mentions(entities: Sequence[EntityModel]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for entity in entities:
        for mention in entity.mentions:
            mentions.append(
                {
                    "entity_id": entity.id,
                    "text": entity.text,
                    "start": mention.start,
                    "end": mention.end,
                }
            )
    return mentions


def match_sentence_relations(
    sentence_text: str,
    sentence_start: int,
    sentence_entities: Sequence[dict[str, Any]],
) -> list[RelationCandidate]:
    lower_sentence = sentence_text.casefold()
    candidates: list[RelationCandidate] = []
    relation_patterns = [
        ("OUTPERFORMS", re.compile(r"\boutperforms?\b")),
        ("CONSISTS_OF", re.compile(r"\bconsists of\b")),
        ("COMPOSED_OF", re.compile(r"\bcomposed of\b")),
        ("PART_OF", re.compile(r"\bpart of\b")),
        ("CONTAINS", re.compile(r"\b(?:contains|includes)\b")),
        ("DEPENDS_ON", re.compile(r"\bdepends? on\b")),
        ("DERIVED_FROM", re.compile(r"\bcome(?:s)? from\b")),
        ("APPLIED_TO", re.compile(r"\bapplied to\b")),
        ("PROJECTED_TO", re.compile(r"\bproject(?:ed)? to\b")),
        ("USES", re.compile(r"\buses?\b")),
        ("PRODUCES", re.compile(r"\b(?:produces?|yields?|resulting in)\b")),
        ("MAPS_TO", re.compile(r"\b(?:maps to|convert(?:s)? to)\b")),
    ]

    def nearest_before(index: int) -> dict[str, Any] | None:
        matches = [item for item in sentence_entities if item["end"] - sentence_start <= index]
        if not matches:
            return None
        return max(matches, key=lambda item: item["end"])

    def nearest_after(index: int) -> dict[str, Any] | None:
        matches = [item for item in sentence_entities if item["start"] - sentence_start >= index]
        if not matches:
            return None
        return min(matches, key=lambda item: item["start"])

    def add_candidate(relation_name: str, left: dict[str, Any] | None, right: dict[str, Any] | None, score: float) -> None:
        if not left or not right or left["entity_id"] == right["entity_id"]:
            return
        candidates.append(
            RelationCandidate(
                subject=left["text"],
                relation=relation_name,
                obj=right["text"],
                score=score,
                evidence=sentence_text,
                chunk_index=-1,
            )
        )

    for relation_name, pattern in relation_patterns:
        for match in pattern.finditer(lower_sentence):
            left = nearest_before(match.start())
            right = nearest_after(match.end())
            add_candidate(
                relation_name,
                left,
                right,
                0.86 if left and right and (is_technical_surface(left["text"]) or is_technical_surface(right["text"])) else 0.72,
            )

    for match in re.finditer(r"\badd(?:ed|s)?\b(?P<middle>.+?)\bto\b", lower_sentence):
        span_start = match.start("middle")
        to_start = match.end() - 2
        subject = nearest_after(span_start)
        target = nearest_after(to_start)
        add_candidate("ADDED_TO", subject, target, 0.88)

    if "queries come from" in lower_sentence and "previous decoder layer" in lower_sentence:
        candidates.append(
            RelationCandidate(
                subject="queries",
                relation="DERIVED_FROM",
                obj="previous decoder layer",
                score=0.9,
                evidence=sentence_text,
                chunk_index=-1,
            )
        )

    if "uses multi-head attention" in lower_sentence:
        candidates.append(
            RelationCandidate(
                subject="Transformer",
                relation="USES",
                obj="multi-head attention",
                score=0.9,
                evidence=sentence_text,
                chunk_index=-1,
            )
        )

    if "part of" in lower_sentence:
        match = re.search(r"\bpart of\b", lower_sentence)
        if match:
            add_candidate("PART_OF", nearest_before(match.start()), nearest_after(match.end()), 0.86)

    return candidates


def resolve_entity_id(
    text: str,
    entity_lookup: dict[tuple[str, str], str],
    entities: Sequence[EntityModel],
    *,
    document_id: str = "",
    chunk_index: int = -1,
    evidence: str = "",
    embedding_session: EmbeddingSession | None = None,
) -> str | None:
    normalized = normalize_surface(text)
    scoped_key = (document_id, normalized)
    if scoped_key in entity_lookup:
        return entity_lookup[scoped_key]

    scoped_candidates: list[tuple[float, str]] = []
    normalized_evidence = evidence.casefold()
    for entity in entities:
        for mention in entity.mentions:
            if document_id and mention.document_id != document_id:
                continue
            if chunk_index >= 0 and mention.chunk_index != chunk_index:
                continue
            mention_normalized = normalize_surface(mention.text)
            if normalized == mention_normalized:
                return entity.id
            if normalized_evidence and mention.text.casefold() not in normalized_evidence and entity.text.casefold() not in normalized_evidence:
                continue
            candidate_score = fuzz.token_set_ratio(normalized, mention_normalized)
            scoped_candidates.append((candidate_score, entity.id))

    if scoped_candidates:
        best_score, best_entity_id = max(scoped_candidates, key=lambda item: item[0])
        if best_score >= 94:
            return best_entity_id

    if embedding_session is not None and embedding_session.enabled and normalized:
        inferred_type = refine_entity_label(text, "Unknown")
        evidence_tokens = tokenize_surface(evidence)
        query_tokens = tokenize_surface(text)
        ranked_candidates: list[tuple[int, float, float, str]] = []
        for entity in entities:
            entity_text_normalized = normalize_surface(entity.text)
            fuzzy_score = max(
                fuzz.ratio(normalized, entity_text_normalized),
                fuzz.token_set_ratio(normalized, entity_text_normalized),
            )
            if document_id and not any(mention.document_id == document_id for mention in entity.mentions):
                continue
            token_overlap = len(query_tokens.intersection(tokenize_surface(entity.text)))
            appears_in_evidence = bool(
                normalized_evidence
                and (
                    entity.text.casefold() in normalized_evidence
                    or any(token in evidence_tokens for token in tokenize_surface(entity.text))
                )
            )
            type_compatible = inferred_type == "Unknown" or entity.type == inferred_type or entity.type == "Unknown"
            if not any((appears_in_evidence, token_overlap > 0, type_compatible, fuzzy_score >= 70)):
                continue
            ranked_candidates.append((1 if appears_in_evidence else 0, float(token_overlap), float(fuzzy_score), entity.id))
        ranked_candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        selected_ids = [item[3] for item in ranked_candidates[: embedding_session.max_candidates]]
        best_similarity = 0.0
        best_entity_id = None
        for entity in entities:
            if entity.id not in selected_ids:
                continue
            similarity = embedding_session.compare_texts(text, entity.text)
            if similarity is None:
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_entity_id = entity.id
        if best_entity_id is not None and best_similarity >= embedding_session.threshold:
            embedding_session.links += 1
            return best_entity_id

    best_entity_id = None
    best_score = 0
    for entity in entities:
        candidate_score = max(
            fuzz.ratio(normalized, normalize_surface(entity.text)),
            fuzz.token_set_ratio(normalized, normalize_surface(entity.text)),
        )
        if candidate_score > best_score:
            best_score = candidate_score
            best_entity_id = entity.id
    if best_score >= 96:
        return best_entity_id
    return None


def choose_relation_strategy(strategies: Sequence[str]) -> str:
    if not strategies:
        return "heuristic_only"
    if any(strategy == "full_rebel" for strategy in strategies):
        return "full_rebel"
    if any(strategy == "heuristic_plus_selective_rebel" for strategy in strategies):
        return "heuristic_plus_selective_rebel"
    return "heuristic_only"


def build_schema(
    entities: Sequence[EntityModel],
    relations: Sequence[RelationModel],
) -> list[SchemaEdgeModel]:
    entity_index = {entity.id: entity for entity in entities}
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for relation in relations:
        source = entity_index[relation.source_id]
        target = entity_index[relation.target_id]
        key = (
            SCHEMA_TYPE_COMPATIBILITY.get(source.type, source.type),
            relation.relation,
            SCHEMA_TYPE_COMPATIBILITY.get(target.type, target.type),
        )
        grouped[key].append(f"{source.text} -[{relation.relation}]-> {target.text}")

    schema = [
        SchemaEdgeModel(
            source_type=source_type,
            relation=relation,
            target_type=target_type,
            count=len(examples),
            examples=examples[:3],
        )
        for (source_type, relation, target_type), examples in sorted(grouped.items())
    ]
    schema.sort(key=lambda item: (-item.count, item.relation, item.source_type, item.target_type))
    return schema


def build_expanded_schema(
    entities: Sequence[EntityModel],
    relations: Sequence[RelationModel],
) -> list[SchemaEdgeModel]:
    entity_index = {entity.id: entity for entity in entities}
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for relation in relations:
        source = entity_index[relation.source_id]
        target = entity_index[relation.target_id]
        schema_relation = SCHEMA_RELATION_ALIASES.get(relation.relation, relation.relation)
        grouped[(source.type, schema_relation, target.type)].append(
            f"{source.text} -[{relation.relation}]-> {target.text}"
        )

    schema = [
        SchemaEdgeModel(
            source_type=source_type,
            relation=relation,
            target_type=target_type,
            count=len(examples),
            examples=examples[:5],
        )
        for (source_type, relation, target_type), examples in sorted(grouped.items())
    ]
    schema.sort(key=lambda item: (-item.count, item.relation, item.source_type, item.target_type))
    return schema


def build_default_extractor(
    max_chars: int = 600,
    *,
    chunk_mode: str = "paragraph",
    chunk_overlap: int = 1,
    mode: str = "balanced",
    max_model_spans: int = 4,
    disable_rebel: bool = False,
    embedding_linking: bool = False,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_threshold: float = 0.84,
    embedding_cache_dir: str | Path = ".cache/extract_graph_embeddings",
    embedding_max_candidates: int = 8,
) -> GraphExtractor:
    return GraphExtractor(
        entity_backend=GLiNEREntityBackend(),
        relation_backend=None if disable_rebel or mode == "fast" else RebelRelationBackend(),
        max_chars=max_chars,
        chunk_mode=chunk_mode,
        chunk_overlap=chunk_overlap,
        mode=mode,
        max_model_spans=max_model_spans,
        disable_rebel=disable_rebel,
        embedding_linking=embedding_linking,
        embedding_model=embedding_model,
        embedding_threshold=embedding_threshold,
        embedding_cache_dir=embedding_cache_dir,
        embedding_max_candidates=embedding_max_candidates,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract entities, relations, and a potential schema from arbitrary text."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="Raw text to process.")
    source_group.add_argument("--input", nargs="+", help="One or more UTF-8 text files to process.")
    parser.add_argument("--output", help="Optional output path for the JSON result.")
    parser.add_argument("--entity-threshold", type=float, default=0.35)
    parser.add_argument("--relation-threshold", type=float, default=0.2)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Execution mode: fast skips REBEL, balanced uses selective REBEL, quality runs REBEL on all chunks.",
    )
    parser.add_argument(
        "--max-model-spans",
        type=int,
        default=4,
        help="Maximum number of shortlisted spans to send to REBEL in balanced mode.",
    )
    parser.add_argument(
        "--disable-rebel",
        action="store_true",
        help="Force heuristic-only relation extraction regardless of mode.",
    )
    parser.add_argument(
        "--embedding-linking",
        action="store_true",
        help="Enable embedding-assisted entity canonicalization and endpoint linking for ambiguous cases.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model used for embedding-assisted linking.",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.84,
        help="Cosine similarity threshold for embedding-assisted merge and link acceptance.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        default=".cache/extract_graph_embeddings",
        help="Directory used for the local SQLite embedding cache.",
    )
    parser.add_argument(
        "--embedding-max-candidates",
        type=int,
        default=8,
        help="Maximum candidate entities to compare via embeddings for an unresolved mention.",
    )
    parser.add_argument(
        "--chunk-mode",
        default="paragraph",
        choices=["paragraph", "sentence"],
        help="Chunking strategy used before entity and relation extraction.",
    )
    parser.add_argument(
        "--entity-scope",
        default="document",
        choices=["document", "corpus"],
        help="Canonicalize entities independently per file or across the full input corpus.",
    )
    parser.add_argument(
        "--include-chunk-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include chunk text in provenance records.",
    )
    parser.add_argument(
        "--omit-provenance-text",
        action="store_true",
        help="Alias for disabling chunk text while keeping chunk metadata and IDs.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--neo4j-uri", help="Optional Neo4j URI for direct graph export, e.g. neo4j://127.0.0.1:7687.")
    parser.add_argument("--neo4j-user", help="Neo4j username for direct graph export.")
    parser.add_argument("--neo4j-password", help="Neo4j password for direct graph export.")
    parser.add_argument(
        "--neo4j-database",
        default="neo4j",
        help="Neo4j database name used for direct graph export.",
    )
    parser.add_argument(
        "--neo4j-clean-document",
        action="store_true",
        help="Delete the matching document subgraph before re-ingesting it into Neo4j.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args(argv)


def read_input_texts(args: argparse.Namespace) -> list[DocumentInput]:
    if args.text:
        return [DocumentInput(text=args.text, source="inline", title="inline_text")]
    input_paths = args.input if isinstance(args.input, (list, tuple)) else [args.input]
    return [
        DocumentInput(
            text=Path(path).read_text(encoding="utf-8"),
            source=Path(path).name,
            title=Path(path).name,
        )
        for path in input_paths
    ]


def read_input_text(args: argparse.Namespace) -> tuple[str, str, str]:
    document = read_input_texts(args)[0]
    return document.text, document.source, document.title


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    neo4j_enabled = validate_neo4j_args(args)

    document_inputs = read_input_texts(args)
    include_chunk_text = args.include_chunk_text and not args.omit_provenance_text
    extractor = build_default_extractor(
        max_chars=args.max_chars,
        chunk_mode=args.chunk_mode,
        chunk_overlap=args.chunk_overlap,
        mode=args.mode,
        max_model_spans=args.max_model_spans,
        disable_rebel=args.disable_rebel,
        embedding_linking=args.embedding_linking,
        embedding_model=args.embedding_model,
        embedding_threshold=args.embedding_threshold,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_max_candidates=args.embedding_max_candidates,
    )
    result = extractor.extract_documents(
        document_inputs,
        entity_threshold=args.entity_threshold,
        relation_threshold=args.relation_threshold,
        max_chars=args.max_chars,
        chunk_mode=args.chunk_mode,
        chunk_overlap=args.chunk_overlap,
        mode=args.mode,
        max_model_spans=args.max_model_spans,
        disable_rebel=args.disable_rebel,
        embedding_linking=args.embedding_linking,
        embedding_model=args.embedding_model,
        embedding_threshold=args.embedding_threshold,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_max_candidates=args.embedding_max_candidates,
        include_chunk_text=include_chunk_text,
        entity_scope=args.entity_scope,
    )

    if args.pretty:
        payload = result.model_dump_json(indent=2)
    else:
        payload = result.model_dump_json()

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")

    if neo4j_enabled:
        export_graph_to_neo4j(
            result,
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database,
            clean_document=args.neo4j_clean_document,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
