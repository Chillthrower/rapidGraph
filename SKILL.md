# rapidGraph Project Skill

This file is an agent-facing guide to the `rapidGraph` codebase. It is intended to help future agents understand what the project does, where the logic lives, how the extraction pipeline works, how the package is published, and what to change for common tasks.

## Purpose

`rapidGraph` is a local-first, open-domain text-to-graph extractor.

It accepts:

- inline text via CLI
- one or more text files via CLI
- programmatic text/documents via Python API

It produces structured JSON with:

- `entities`
- `relations`
- `potential_schema`
- `expanded_schema`
- provenance-aware `documents`
- provenance-aware `chunks`
- `relation_support`
- `meta`

It also supports:

- CPU-aware execution modes
- optional embedding-assisted linking
- optional direct Neo4j export
- optional GraphRAG question answering over Neo4j vector indexes and Ollama
- multi-file corpus ingestion

The public distribution name is `rapidGraph`.
The import package is `rapidgraph`.
The installed CLI command is `rapidgraph`.

## High-Level Architecture

The project currently has one main implementation module and a few wrappers:

- [rapidgraph/core.py](/Users/sadyanth/Desktop/RAG/ml_proj/rapidgraph/core.py)
  - main implementation
  - data models
  - chunking
  - extraction pipeline
  - canonicalization
  - relation linking
  - schema generation
  - Neo4j export
  - CLI parsing and runtime entrypoint
- [rapidgraph/graphrag.py](/Users/sadyanth/Desktop/RAG/ml_proj/rapidgraph/graphrag.py)
  - Neo4j vector retrieval
  - Ollama LLM provider
  - GraphRAG answer orchestration
- [rapidgraph/cli.py](/Users/sadyanth/Desktop/RAG/ml_proj/rapidgraph/cli.py)
  - thin lazy-import CLI wrapper
  - exists so `rapidgraph --help` does not eagerly import the heavy ML stack
- [rapidgraph/__init__.py](/Users/sadyanth/Desktop/RAG/ml_proj/rapidgraph/__init__.py)
  - re-exports the public package API from `core.py`
  - exposes `__version__`
- [extract_graph.py](/Users/sadyanth/Desktop/RAG/ml_proj/extract_graph.py)
  - compatibility shim for older usage
  - delegates to the packaged implementation

There is also a legacy side-path:

- [extract_graphv2.py](/Users/sadyanth/Desktop/RAG/ml_proj/extract_graphv2.py)
- [tests/test_extract_graphv2.py](/Users/sadyanth/Desktop/RAG/ml_proj/tests/test_extract_graphv2.py)

This legacy file is separate from the packaged `rapidgraph` implementation. Do not confuse it with the current package unless the user explicitly asks to modify that legacy path.

## File Map

Important files:

- [pyproject.toml](/Users/sadyanth/Desktop/RAG/ml_proj/pyproject.toml)
  - package metadata
  - dependencies
  - console script entrypoint
- [README.md](/Users/sadyanth/Desktop/RAG/ml_proj/README.md)
  - public documentation
- [LICENSE](/Users/sadyanth/Desktop/RAG/ml_proj/LICENSE)
  - MIT license
- [.github/workflows/publish.yml](/Users/sadyanth/Desktop/RAG/ml_proj/.github/workflows/publish.yml)
  - real PyPI publish workflow
- [.github/workflows/publish-testpypi.yml](/Users/sadyanth/Desktop/RAG/ml_proj/.github/workflows/publish-testpypi.yml)
  - TestPyPI publish workflow
- [tests/test_extract_graph.py](/Users/sadyanth/Desktop/RAG/ml_proj/tests/test_extract_graph.py)
  - main regression suite for current implementation

## Installation and Setup

Agents should know both the public install path and the local development install path.

### Install from PyPI

Base package:

```bash
pip install rapidGraph
```

With Neo4j support:

```bash
pip install "rapidGraph[neo4j]"
```

With embedding support:

```bash
pip install "rapidGraph[embeddings]"
```

With GraphRAG support:

```bash
pip install "rapidGraph[graphrag]"
```

With development tooling:

```bash
pip install "rapidGraph[dev]"
```

With everything:

```bash
pip install "rapidGraph[neo4j,embeddings,graphrag,dev]"
```

### Install from source

From the repo root:

```bash
pip install .
```

With extras:

```bash
pip install ".[neo4j]"
pip install ".[embeddings]"
pip install ".[graphrag]"
pip install ".[dev]"
pip install ".[neo4j,embeddings,graphrag,dev]"
```

### Editable install for active development

```bash
pip install -e ".[neo4j,embeddings,graphrag,dev]"
```

### TestPyPI install

If an agent needs to verify a staged release from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rapidGraph
```

### Notes for agents

- The package name on indexes is `rapidGraph`.
- The import package is `rapidgraph`.
- The CLI command is `rapidgraph`.
- Neo4j helper usage requires the `neo4j` extra or an equivalent direct install of the `neo4j` package.
- Embedding-assisted linking requires the `sentence-transformers` dependency, exposed through the `embeddings` extra.
- GraphRAG ask mode requires the `graphrag` extra, which includes Neo4j, sentence-transformers, and requests.

## Public Runtime Entry Points

### CLI

Main CLI is exposed through:

- `rapidgraph`
- `python extract_graph.py`

The installed console script points to:

- `rapidgraph.cli:main`

The lazy wrapper eventually calls:

- `rapidgraph.core.main(argv)`

### Python API

The most important public entry points are re-exported from `rapidgraph`.

Key public API functions and classes:

- `build_default_extractor(...)`
- `GraphExtractor`
- `DocumentInput`
- `GraphExtraction`
- `EntityModel`
- `RelationModel`
- `SchemaEdgeModel`
- `export_graph_to_neo4j(...)`
- `Neo4jVectorRetriever`
- `OllamaLLM`
- `GraphRAGClient`

## Core Data Models

Pydantic output models are defined in [rapidgraph/core.py](/Users/sadyanth/Desktop/RAG/ml_proj/rapidgraph/core.py).

Main models:

- `MentionModel`
- `EntityModel`
- `RelationModel`
- `DocumentModel`
- `ChunkModel`
- `RelationSupportModel`
- `SchemaEdgeModel`
- `MetaModel`
- `GraphExtraction`

Internal working dataclasses / records:

- `Chunk`
- `EntityCandidate`
- `RelationCandidate`
- `DocumentInput`

### Output Contract

The top-level JSON contract is:

- `entities`
- `relations`
- `potential_schema`
- `expanded_schema`
- `documents`
- `chunks`
- `relation_support`
- `meta`

This contract should be treated as stable unless the user explicitly requests a breaking change.

## Extraction Pipeline

The main pipeline is implemented in:

- `GraphExtractor.extract(...)`
- `GraphExtractor.extract_documents(...)`

Execution flow:

1. Normalize document text.
2. Chunk the text with `chunk_text(...)`.
3. Build public `DocumentModel` and `ChunkModel` records.
4. Extract entity candidates:
   - backend model extraction
   - heuristic entity extraction
5. Provisional per-document canonicalization for local relation work.
6. Extract relation candidates:
   - heuristic backend
   - context relation extraction
   - optional REBEL backend
7. Canonicalize final entities:
   - per-document or corpus-wide depending on `entity_scope`
   - optional embedding-assisted rescue for borderline merges
8. Link relations to canonical entity IDs.
9. Build:
   - `potential_schema`
   - `expanded_schema`
   - `relation_support`
10. Emit final `GraphExtraction`.

## Important Implementation Areas

### 1. Normalization helpers

Key helper functions:

- `normalize_whitespace`
- `normalize_entity_type`
- `normalize_relation_name`
- `normalize_surface`
- `tokenize_surface`

These functions are heavily relied upon by canonicalization, relation naming, and schema generation. Changes here can have wide ripple effects.

### 2. Technical entity refinement

Technical typing is improved by:

- `classify_technical_entity`
- `refine_entity_label`
- `is_low_signal_entity`
- `clean_technical_phrase`

These functions are responsible for reducing generic or noisy technical entities and refining broad labels into more meaningful ones such as:

- `Model`
- `Method`
- `Component`
- `Transformation`
- `DataStructure`
- `Metric`

When improving typing quality, start here.

### 3. Chunking

Chunking is implemented in:

- `split_text_blocks`
- `pack_block_into_chunks`
- `chunk_text`

Current supported modes:

- `paragraph`
- `sentence`

Important facts:

- chunk overlap is sentence-based
- paragraph mode is the current default
- chunking strongly affects relation quality and schema richness

When users ask why schema counts are lower or why some relations are missing, chunking is often part of the answer.

### 4. Entity backends

Current entity backend classes:

- `GLiNEREntityBackend`
- `RegexEntityBackend`

Default construction uses:

- `GLiNEREntityBackend()`

Fallback behavior matters:

- the backend can accumulate warnings
- `fallback_used` is surfaced through `meta`

### 5. Relation backends

Current relation backend classes:

- `RebelRelationBackend`
- `HeuristicRelationBackend`

Default orchestration depends on mode.

Mode behavior:

- `fast`
  - no REBEL
  - heuristic-only
- `balanced`
  - heuristic everywhere
  - selective REBEL on top-ranked spans
- `quality`
  - REBEL on all chunks

Core relation orchestration lives in:

- `GraphExtractor._collect_relation_candidates`
- `extract_context_relations`
- `select_rebel_candidate_spans`
- `count_rebel_candidate_spans`
- `score_rebel_span`

### 6. Embedding-assisted rescue

Embedding support exists for ambiguous cases only.

Key classes and helpers:

- `SQLiteEmbeddingCache`
- `SentenceTransformerEmbeddingBackend`
- `EmbeddingSession`
- `cosine_similarity`
- `is_embedding_merge_candidate`
- `has_embedding_merge_lexical_compatibility`

Important design choice:

- embeddings are not the primary extraction mechanism
- they are an opt-in rescue path for borderline merge/linking cases

### 7. Canonicalization and relation linking

Important functions:

- `GraphExtractor._canonicalize_entities`
- `resolve_entity_id`
- `match_sentence_relations`
- `build_entity_mentions`

Important behavior:

- exact and local linking is preferred
- fuzzy matching is fallback
- embedding rescue is last-resort when enabled
- `entity_scope` controls whether cross-document merging is allowed

### 8. Schema generation

Key functions:

- `build_schema`
- `build_expanded_schema`

Important behavior:

- `potential_schema` is strict and compatibility-oriented
- `expanded_schema` is richer and often larger
- schema rows are generated from final linked relations, not from raw candidates

### 9. Provenance support

Key functions:

- `build_document_record`
- `build_chunk_records`
- `resolve_relation_chunk_ids`
- `build_relation_support`

Important behavior:

- `MentionModel` includes `document_id` and `chunk_id`
- `RelationModel` includes `chunk_ids` and `document_ids`
- `relation_support` is derived from final relations

This provenance layer exists specifically to make future retrieval and graph-backed QA practical.

### 10. Neo4j export

Key functions:

- `validate_neo4j_args`
- `export_graph_to_neo4j`
- `build_neo4j_mention_rows`

Neo4j queries are embedded in constants:

- `NEO4J_CONSTRAINT_QUERIES`
- `NEO4J_DOCUMENT_QUERY`
- `NEO4J_CHUNK_QUERY`
- `NEO4J_ENTITY_QUERY`
- `NEO4J_MENTION_QUERY`
- `NEO4J_RELATION_QUERY`
- delete queries for clean re-ingest

Important Neo4j model fact:

- semantic relations like `IS_BASED_IN` or `USES` are stored as `r.relation` on `RELATES_TO`
- Neo4j Browser therefore shows many `RELATES_TO` edges rather than separate relationship types per semantic relation

### 11. GraphRAG ask layer

GraphRAG implementation lives in:

- `rapidgraph/graphrag.py`

Key classes:

- `Neo4jVectorRetriever`
- `OllamaLLM`
- `GraphRAGClient`
- `RetrievedChunk`
- `RetrievedFact`
- `GraphRAGAnswer`

Important behavior:

- `Chunk` is the vector retrieval unit.
- Neo4j vector indexes are the only HNSW backend in v1.
- Ollama is the only LLM provider in v1.
- The retriever queries `db.index.vector.queryNodes(...)`, expands to mentioned entities, and collects nearby `RELATES_TO` facts.
- `rapidgraph ask` is routed before the extraction CLI parser, so extraction flags remain backward compatible.

## Public CLI Surface

CLI parsing lives in:

- `parse_args(...)`

Main supported flags:

- `--text`
- `--input`
- `--output`
- `--entity-threshold`
- `--relation-threshold`
- `--max-chars`
- `--chunk-overlap`
- `--mode`
- `--max-model-spans`
- `--disable-rebel`
- `--embedding-linking`
- `--embedding-model`
- `--embedding-threshold`
- `--embedding-cache-dir`
- `--embedding-max-candidates`
- `--chunk-mode`
- `--entity-scope`
- `--include-chunk-text` / `--no-include-chunk-text`
- `--omit-provenance-text`
- `--pretty`
- `--neo4j-uri`
- `--neo4j-user`
- `--neo4j-password`
- `--neo4j-database`
- `--neo4j-clean-document`
- `--neo4j-embed-chunks`
- `--neo4j-create-vector-index`
- `--neo4j-vector-index-name`
- `--neo4j-embedding-property`
- `--chunk-embedding-model`
- `--log-level`

Ask-mode flags:

- `rapidgraph ask`
- `--question`
- `--top-k`
- `--graph-depth`
- `--max-facts`
- `--ollama-host`
- `--ollama-model`
- Neo4j connection and vector index flags

Defaults worth remembering:

- `mode="balanced"`
- `max_chars=600`
- `chunk_mode="paragraph"`
- `chunk_overlap=1`
- `entity_scope="document"`
- `entity_threshold=0.35`
- `relation_threshold=0.2`
- `max_model_spans=4`
- `embedding_threshold=0.84`
- `embedding_max_candidates=8`

## Packaging and Publishing

Packaging is configured in:

- [pyproject.toml](/Users/sadyanth/Desktop/RAG/ml_proj/pyproject.toml)

Important metadata:

- distribution name: `rapidGraph`
- import package: `rapidgraph`
- console script: `rapidgraph = rapidgraph.cli:main`

Base dependencies:

- `gliner`
- `pydantic`
- `rapidfuzz`
- `spacy`
- `torch`
- `transformers`

Optional extras:

- `neo4j`
- `embeddings`
- `graphrag`
- `dev`

Publish workflows:

- real PyPI: `.github/workflows/publish.yml`
- TestPyPI: `.github/workflows/publish-testpypi.yml`

These use trusted publishing via GitHub Actions.

## Tests

Main current test suite:

- [tests/test_extract_graph.py](/Users/sadyanth/Desktop/RAG/ml_proj/tests/test_extract_graph.py)

This suite covers:

- business/news extraction shape
- technical/scientific extraction shape
- mixed-domain handling
- deduplication
- CLI behavior
- chunking
- mode-specific relation behavior
- schema generation
- embedding-assisted behavior
- provenance behavior

Legacy separate test suite:

- [tests/test_extract_graphv2.py](/Users/sadyanth/Desktop/RAG/ml_proj/tests/test_extract_graphv2.py)

This is not the main packaged pipeline. Only edit it if the user explicitly asks for that legacy path.

Useful commands:

```bash
pytest -q tests/test_extract_graph.py
python -m build
python -m twine check dist/*
```

## Common Tasks and Where to Edit

### Add a new CLI flag

Edit:

- `parse_args(...)` in `rapidgraph/core.py`
- `main(...)` in `rapidgraph/core.py`
- maybe `build_default_extractor(...)`
- README examples if user-facing
- tests in `tests/test_extract_graph.py`

### Improve chunking

Edit:

- `split_text_blocks`
- `pack_block_into_chunks`
- `chunk_text`

Also verify:

- provenance correctness
- chunk counts in `meta`
- overlap behavior

### Improve entity typing

Edit:

- `classify_technical_entity`
- `refine_entity_label`
- `is_low_signal_entity`
- relevant keyword sets and overrides

### Improve relation quality

Edit:

- `HeuristicRelationBackend`
- `extract_context_relations`
- `is_relation_plausible`
- `match_sentence_relations`
- REBEL span selection helpers

### Change canonicalization behavior

Edit:

- `GraphExtractor._canonicalize_entities`
- `resolve_entity_id`
- embedding helper gating logic

This is a high-risk area because it affects:

- entity counts
- relation linking
- schema aggregation
- provenance consistency

### Extend Neo4j export

Edit:

- `export_graph_to_neo4j`
- Neo4j query constants
- maybe README Neo4j section

Be careful to preserve:

- id stability
- clean re-ingest behavior
- compatibility with current graph shape

### Extend GraphRAG ask mode

Edit:

- `rapidgraph/graphrag.py`
- `ask_main` and `parse_ask_args` in `rapidgraph/core.py`
- README GraphRAG section
- tests in `tests/test_extract_graph.py`

Preserve:

- `rapidgraph ask` JSON output shape
- lazy imports for optional dependencies
- Neo4j vector index defaults
- Ollama-only v1 provider scope unless explicitly requested

### Change JSON output contract

Edit:

- Pydantic models
- extraction assembly
- tests
- README

This should be treated as a breaking change unless explicitly requested.

## Common Commands

### Run on inline text

```bash
rapidgraph --text "Google is based in California." --mode fast --pretty
```

### Run on files

```bash
rapidgraph --input input.txt input2.txt --mode balanced --entity-scope corpus --pretty
```

### Run with Neo4j export

```bash
rapidgraph \
  --input input.txt \
  --mode quality \
  --neo4j-uri neo4j://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password 12345678 \
  --neo4j-database neo4j \
  --neo4j-clean-document \
  --neo4j-embed-chunks \
  --neo4j-create-vector-index
```

### Ask the graph with Ollama

```bash
rapidgraph ask \
  --question "What does the graph say about attention?" \
  --neo4j-uri neo4j://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password 12345678 \
  --neo4j-database neo4j \
  --ollama-model llama3.2 \
  --pretty
```

### Python API

```python
from rapidgraph import build_default_extractor

extractor = build_default_extractor(mode="balanced")
result = extractor.extract("Google is based in California.")
print(result.model_dump())
```

### Python GraphRAG API

```python
from rapidgraph import GraphRAGClient, Neo4jVectorRetriever, OllamaLLM

retriever = Neo4jVectorRetriever(
    uri="neo4j://127.0.0.1:7687",
    user="neo4j",
    password="12345678",
)
llm = OllamaLLM(model="llama3.2")
answer = GraphRAGClient(retriever=retriever, llm=llm).ask("What does the graph say?")
print(answer.model_dump())
```

## Important Project-Specific Gotchas

### 1. `rapidgraph/cli.py` must stay lightweight

The CLI wrapper intentionally lazy-imports `core.py`. Do not reintroduce eager heavy imports there unless you want `rapidgraph --help` to require the ML stack at import time.

### 2. `extract_graph.py` is a compatibility shim

Tests still import this root-level shim. Breaking or removing it can break compatibility and tests.

### 3. Real extraction is in `rapidgraph/core.py`

Do not spread primary implementation logic into multiple top-level files unless there is a strong reason.

### 4. `extract_graphv2.py` is legacy

It has its own tests and should not be treated as the current packaged pipeline.

### 5. Neo4j relation semantics are properties, not relationship types

This is intentional and must be understood before “fixing” it.

### 6. GraphRAG requires embedded chunks

`rapidgraph ask` expects `Chunk` nodes to have embeddings and a Neo4j vector index. Users should export with `--neo4j-embed-chunks --neo4j-create-vector-index` before asking questions.

### 7. Schema counts depend on final linked relations

Users may ask why `potential_schema` is smaller than expected. The reason is usually:

- stricter type grouping
- relation filtering
- lost context at chunk boundaries
- relation linking failures
- duplicate relations collapsing into one edge

### 8. First-run model downloads are normal

When testing in a clean environment, Hugging Face-backed model fetches may happen. That is not necessarily a bug.

## How to Approach Changes Safely

When making changes:

1. identify whether the change affects:
   - extraction quality
   - JSON contract
   - CLI contract
   - package distribution
   - Neo4j export
2. update tests or add new tests
3. preserve backward compatibility unless the user explicitly requests otherwise
4. prefer additive changes for output fields
5. avoid modifying legacy files unless requested

## Recommended Validation Flow

For most code changes:

1. run:

```bash
pytest -q tests/test_extract_graph.py
```

2. if packaging changed:

```bash
python -m build
python -m twine check dist/*
```

3. if CLI changed:

```bash
rapidgraph --help
```

4. if extraction changed:

```bash
rapidgraph --text "Google is based in California." --mode fast --pretty
```

5. if Neo4j export changed:

- verify graph shape in Neo4j
- verify clean re-ingest

## Agent Guidance Summary

If you are an agent working on this project:

- treat `rapidgraph/core.py` as the source of truth
- keep the CLI and JSON surface stable unless the user asks for a breaking change
- update README when adding user-facing features
- update tests whenever extraction behavior or flags change
- do not confuse `extract_graphv2.py` with the main packaged implementation
- remember that provenance and Neo4j compatibility are deliberate design goals, not incidental details
