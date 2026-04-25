# rapidGraph

`rapidGraph` is a local-first, open-domain text-to-graph extractor for arbitrary text. It reads inline text or one or more `.txt` files and produces a structured graph-oriented JSON payload with:

- `entities`
- `relations`
- `potential_schema`
- `expanded_schema`
- `documents`
- `chunks`
- `relation_support`
- `meta`

The package is designed for:

- entity and relation extraction across general, technical, scientific, and mixed-domain text
- CPU-friendly local execution
- provenance-aware graph construction
- future GraphRAG / RAG workflows
- optional Neo4j ingestion

The public package name is `rapidGraph`, the import package is `rapidgraph`, and the installed CLI command is `rapidgraph`.

## Table of Contents

- [What rapidGraph Does](#what-rapidgraph-does)
- [Key Capabilities](#key-capabilities)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Execution Modes](#execution-modes)
- [Input Model](#input-model)
- [Output Model](#output-model)
- [CLI Reference](#cli-reference)
- [Recommended Flag Combinations](#recommended-flag-combinations)
- [Neo4j Export Model](#neo4j-export-model)
- [Python Library Usage](#python-library-usage)
- [Performance and Practical Notes](#performance-and-practical-notes)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Publishing](#publishing)
- [License](#license)

## What rapidGraph Does

At a high level, `rapidGraph` takes arbitrary text and turns it into a graph-friendly representation.

The pipeline:

1. normalizes input text
2. splits text into chunked spans
3. extracts entity candidates
4. extracts relation candidates
5. canonicalizes duplicate or near-duplicate mentions
6. links relation endpoints to canonical entities
7. infers schema patterns from the accepted graph edges
8. stores document and chunk provenance so every entity mention and relation can be traced back to source text

The extractor is open-domain and best-effort. It does not rely on a fixed business-only ontology. If typing confidence is weak, it keeps entities as `Unknown` rather than discarding them.

## Key Capabilities

- Open-domain entity extraction
- Open-domain relation extraction
- Schema inference from extracted graph edges
- Provenance-aware output using `documents`, `chunks`, and `relation_support`
- Multi-file corpus ingestion in one run
- Two entity canonicalization scopes:
  - `document`
  - `corpus`
- Three execution modes:
  - `fast`
  - `balanced`
  - `quality`
- Optional embedding-assisted entity merging and relation endpoint linking
- Optional Neo4j export
- Backward-compatible `potential_schema` plus richer `expanded_schema`

## Installation

### Install from PyPI

```bash
pip install rapidGraph
```

### Install with optional extras

Neo4j support:

```bash
pip install "rapidGraph[neo4j]"
```

Embedding-assisted linking:

```bash
pip install "rapidGraph[embeddings]"
```

Development tooling:

```bash
pip install "rapidGraph[dev]"
```

Everything:

```bash
pip install "rapidGraph[neo4j,embeddings,dev]"
```

### Install from source

```bash
pip install .
```

Or with extras:

```bash
pip install ".[neo4j,embeddings,dev]"
```

## Quick Start

Show CLI help:

```bash
rapidgraph --help
```

Extract from inline text:

```bash
rapidgraph --text "Google is based in California." --pretty
```

Extract from a file:

```bash
rapidgraph --input input.txt --pretty
```

Extract from multiple files:

```bash
rapidgraph --input input.txt input2.txt --pretty
```

Write JSON to a file:

```bash
rapidgraph --input input.txt --output graph.json --pretty
```

The repo-root compatibility shim also works:

```bash
python extract_graph.py --input input.txt --pretty
```

## How the Pipeline Works

### 1. Text normalization

The input is normalized for whitespace and line ending consistency before extraction begins.

### 2. Chunking

The extractor splits text into chunks before model inference. Chunking exists because relation and entity models work better on bounded spans than on arbitrarily long documents.

Two chunking strategies are available:

- `paragraph`
  - default
  - respects paragraph and block boundaries first
  - better for preserving local structure
- `sentence`
  - simpler sentence packing
  - useful for experimentation or tighter chunk control

Optional overlap preserves context across chunk boundaries.

### 3. Entity extraction

Entities are primarily extracted with GLiNER, with heuristic fallback and supplemental heuristics used where useful.

### 4. Relation extraction

Relations come from a combination of:

- heuristic relation extraction
- context-based relation patterns
- optional REBEL relation extraction

The amount of REBEL usage depends on `--mode`.

### 5. Canonicalization

Mentions are merged into canonical entities using:

- normalized string matching
- fuzzy matching
- optional embedding-assisted rescue for borderline cases

### 6. Relation linking

Relation endpoints are linked back to canonical entity IDs using:

- exact and local mention-aware matching first
- fuzzy matching second
- optional embedding-assisted rescue last

### 7. Schema generation

Two schema views are produced:

- `potential_schema`
  - strict compatibility view
  - grouped by `(source_type, relation, target_type)`
- `expanded_schema`
  - richer view using more refined type groupings
  - keeps more semantic detail

### 8. Provenance capture

Each mention and accepted relation can be traced to:

- a `document`
- one or more `chunks`
- representative evidence text

This is what makes the model usable later for retrieval or graph-backed answer generation.

## Execution Modes

`rapidGraph` supports three runtime modes.

### `fast`

Best for:

- CPU-only quick passes
- rapid iteration
- rough graph drafts

Behavior:

- uses GLiNER plus heuristics
- does not run REBEL
- lowest startup cost
- lowest relation recall of the three modes

Example:

```bash
rapidgraph --input input.txt --mode fast --pretty
```

### `balanced`

This is the default mode.

Best for:

- most local CPU runs
- practical relation quality without paying the full REBEL cost

Behavior:

- runs heuristic relations everywhere
- runs REBEL only on shortlisted high-value spans
- usually the best speed/quality tradeoff

Example:

```bash
rapidgraph --input input.txt --mode balanced --pretty
```

### `quality`

Best for:

- slower offline analysis
- smaller corpora
- maximum relation recall

Behavior:

- runs REBEL across all chunks
- highest model cost
- typically the slowest mode

Example:

```bash
rapidgraph --input input.txt --mode quality --pretty
```

## Input Model

The CLI accepts exactly one of:

- `--text "..."` for inline text
- `--input file1.txt [file2.txt ...]` for file input

`--text` and `--input` are mutually exclusive.

Multi-file ingestion produces a single combined JSON result with multiple `documents` and `chunks`.

## Output Model

The extractor returns a single JSON object.

### `entities`

Each entity contains:

- `id`
- `text`
- `canonical`
- `type`
- `confidence`
- `mentions`

Each mention contains:

- `text`
- `start`
- `end`
- `chunk_index`
- `document_id`
- `chunk_id`

### `relations`

Each relation contains:

- `source_id`
- `target_id`
- `relation`
- `confidence`
- `evidence`
- `chunk_ids`
- `document_ids`

### `potential_schema`

Strict schema aggregation. This preserves backward compatibility and groups edges by:

- `source_type`
- `relation`
- `target_type`

### `expanded_schema`

Richer schema aggregation that retains more type detail and gives a broader schema view than `potential_schema`.

### `documents`

One row per input document:

- `id`
- `source`
- `title`
- `text_hash`
- `char_count`

### `chunks`

One row per extraction chunk:

- `id`
- `document_id`
- `index`
- `text`
- `start`
- `end`
- `block_index`
- `overlap_sentences`

### `relation_support`

One row per final accepted relation edge with merged provenance:

- `source_id`
- `relation`
- `target_id`
- `chunk_ids`
- `document_ids`
- `evidence`

### `meta`

Contains execution metadata such as:

- model names
- thresholds
- chunk count
- elapsed time
- mode
- relation backend strategy
- REBEL usage counts
- embedding usage counts
- warning list
- fallback indicator

### Example output shape

```json
{
  "entities": [
    {
      "id": "E1",
      "text": "Google",
      "canonical": "Google",
      "type": "Organization",
      "confidence": 0.91,
      "mentions": [
        {
          "text": "Google",
          "start": 0,
          "end": 6,
          "chunk_index": 0,
          "document_id": "D1",
          "chunk_id": "D1:C0"
        }
      ]
    }
  ],
  "relations": [
    {
      "source_id": "E1",
      "target_id": "E2",
      "relation": "IS_BASED_IN",
      "confidence": 0.78,
      "evidence": "Google is based in California.",
      "chunk_ids": ["D1:C0"],
      "document_ids": ["D1"]
    }
  ],
  "potential_schema": [],
  "expanded_schema": [],
  "documents": [],
  "chunks": [],
  "relation_support": [],
  "meta": {}
}
```

## CLI Reference

### Input and output

#### `--text TEXT`

Inline text to process.

Example:

```bash
rapidgraph --text "Transformer uses attention." --pretty
```

#### `--input INPUT [INPUT ...]`

One or more UTF-8 text files.

Examples:

```bash
rapidgraph --input input.txt
rapidgraph --input input.txt input2.txt
```

#### `--output OUTPUT`

Write JSON to a file instead of stdout.

Example:

```bash
rapidgraph --input input.txt --output graph.json --pretty
```

#### `--pretty`

Pretty-print JSON output.

### Thresholds and chunking

#### `--entity-threshold`

Default: `0.35`

Minimum confidence for keeping entity candidates.

#### `--relation-threshold`

Default: `0.2`

Minimum confidence for keeping relation candidates.

#### `--max-chars`

Default: `600`

Approximate chunk size budget.

Higher values:

- preserve more context
- may improve some relations
- increase compute cost

#### `--chunk-mode {paragraph,sentence}`

Default: `paragraph`

Controls chunk construction.

#### `--chunk-overlap`

Default: `1`

Number of overlapping sentences preserved between neighboring chunks.

Example:

```bash
rapidgraph --input input.txt --chunk-overlap 2
```

### Runtime mode and relation strategy

#### `--mode {fast,balanced,quality}`

Default: `balanced`

Controls the speed/quality tradeoff.

#### `--max-model-spans`

Default: `4`

Balanced-mode only. Caps how many shortlisted spans go through REBEL.

Example:

```bash
rapidgraph --input input.txt --mode balanced --max-model-spans 6
```

#### `--disable-rebel`

Force heuristic-only relation extraction regardless of mode.

Example:

```bash
rapidgraph --input input.txt --mode quality --disable-rebel
```

### Entity canonicalization scope

#### `--entity-scope {document,corpus}`

Default: `document`

Controls whether compatible entities can merge across files.

Use `document` when:

- files are independent
- names may be ambiguous across documents
- you want safer graph boundaries

Use `corpus` when:

- the files describe the same topic or domain
- you want one merged entity layer across the corpus
- you are building a shared graph for Neo4j or GraphRAG

Examples:

```bash
rapidgraph --input input.txt input2.txt --entity-scope document
rapidgraph --input input.txt input2.txt --entity-scope corpus
```

### Embedding-assisted linking

These flags are optional. They are not enabled by default.

#### `--embedding-linking`

Enable embedding-assisted rescue for ambiguous entity merges and unresolved relation endpoints.

#### `--embedding-model`

Default:

```text
sentence-transformers/all-MiniLM-L6-v2
```

#### `--embedding-threshold`

Default: `0.84`

Cosine similarity threshold for accepting embedding-assisted merge or link candidates.

#### `--embedding-cache-dir`

Default:

```text
.cache/extract_graph_embeddings
```

Embedding vectors are cached locally in SQLite form.

#### `--embedding-max-candidates`

Default: `8`

Maximum number of candidates considered in embedding-assisted linking for an unresolved mention.

Example:

```bash
rapidgraph \
  --input input.txt input2.txt \
  --entity-scope corpus \
  --embedding-linking \
  --embedding-threshold 0.84 \
  --embedding-max-candidates 8 \
  --pretty
```

### Provenance controls

#### `--include-chunk-text`

Default: enabled

Include chunk text inside the `chunks` array.

#### `--no-include-chunk-text`

Omit chunk text while keeping chunk metadata.

#### `--omit-provenance-text`

Alias for omitting chunk text while preserving chunk IDs and provenance structure.

Examples:

```bash
rapidgraph --input input.txt --no-include-chunk-text
rapidgraph --input input.txt --omit-provenance-text
```

### Neo4j export

These flags are optional. Without them, the CLI only prints or writes JSON.

#### `--neo4j-uri`

Neo4j URI, for example:

```text
neo4j://127.0.0.1:7687
```

#### `--neo4j-user`

Neo4j username.

#### `--neo4j-password`

Neo4j password.

#### `--neo4j-database`

Default: `neo4j`

Neo4j database name.

#### `--neo4j-clean-document`

Deletes matching document subgraphs before re-ingesting them.

Useful when rerunning the same files and you do not want duplicate document/chunk subgraphs.

Example:

```bash
rapidgraph \
  --input input.txt input2.txt \
  --mode quality \
  --entity-scope corpus \
  --neo4j-uri neo4j://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password 12345678 \
  --neo4j-database neo4j \
  --neo4j-clean-document
```

### Logging

#### `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`

Default: `WARNING`

Controls CLI log verbosity.

Example:

```bash
rapidgraph --input input.txt --log-level DEBUG
```

## Recommended Flag Combinations

### Fastest CPU pass

```bash
rapidgraph --input input.txt --mode fast --pretty
```

### Best default for most users

```bash
rapidgraph --input input.txt --mode balanced --pretty
```

### Higher-recall single document run

```bash
rapidgraph --input input.txt --mode quality --chunk-overlap 2 --pretty
```

### Multi-file corpus merge

```bash
rapidgraph \
  --input input.txt input2.txt \
  --mode balanced \
  --entity-scope corpus \
  --pretty
```

### Multi-file corpus with stronger ambiguous-link rescue

```bash
rapidgraph \
  --input input.txt input2.txt \
  --mode balanced \
  --entity-scope corpus \
  --embedding-linking \
  --pretty
```

### Smaller provenance payload

```bash
rapidgraph \
  --input input.txt \
  --omit-provenance-text \
  --pretty
```

### Neo4j ingestion with document replacement

```bash
rapidgraph \
  --input input.txt input2.txt \
  --mode quality \
  --entity-scope corpus \
  --neo4j-uri neo4j://127.0.0.1:7687 \
  --neo4j-user neo4j \
  --neo4j-password 12345678 \
  --neo4j-database neo4j \
  --neo4j-clean-document
```

## Neo4j Export Model

When Neo4j export is enabled, the graph currently uses:

Node labels:

- `Document`
- `Chunk`
- `Entity`

Relationship types:

- `HAS_CHUNK`
- `MENTIONS`
- `RELATES_TO`

Important detail:

- the semantic edge label such as `IS_BASED_IN`, `USES`, or `DERIVED_FROM` is stored as a property on `RELATES_TO`
- this is why Neo4j Browser may show many `RELATES_TO` relationships while the actual semantic relation name is visible in `r.relation`

Example query:

```cypher
MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
RETURN s.text, r.relation, t.text, r.evidence
ORDER BY r.relation
```

## Python Library Usage

### Basic usage

```python
from rapidgraph import DocumentInput, build_default_extractor

extractor = build_default_extractor(mode="balanced")

result = extractor.extract_documents(
    [
        DocumentInput(
            text="Google is based in California.",
            source="one.txt",
            title="one.txt",
        ),
        DocumentInput(
            text="Sundar Pichai leads Google.",
            source="two.txt",
            title="two.txt",
        ),
    ],
    entity_scope="corpus",
)

print(result.model_dump())
```

### Single-document usage

```python
from rapidgraph import build_default_extractor

extractor = build_default_extractor(mode="fast")
result = extractor.extract(
    "Transformer uses multi-head attention.",
    include_chunk_text=True,
)

print(result.model_dump_json(indent=2))
```

### Configurable extractor construction

```python
from rapidgraph import build_default_extractor

extractor = build_default_extractor(
    max_chars=800,
    chunk_mode="paragraph",
    chunk_overlap=2,
    mode="balanced",
    max_model_spans=6,
    embedding_linking=True,
)
```

### Python library usage with Neo4j export

This example extracts a graph in Python and then writes it directly to Neo4j.

```python
from rapidgraph import build_default_extractor, export_graph_to_neo4j

extractor = build_default_extractor(
    mode="balanced",
    chunk_mode="paragraph",
    chunk_overlap=1,
)

result = extractor.extract(
    """
    Google is based in California.
    Sundar Pichai leads Google.
    """,
    entity_scope="document",
    include_chunk_text=True,
)

export_graph_to_neo4j(
    result,
    uri="neo4j://127.0.0.1:7687",
    user="neo4j",
    password="12345678",
    database="neo4j",
    clean_document=True,
)
```

If you want to use the Neo4j helper, install the extra first:

```bash
pip install "rapidGraph[neo4j]"
```

## Performance and Practical Notes

### CPU expectations

- `fast` is the cheapest mode
- `balanced` is usually the best practical CPU choice
- `quality` may be significantly slower because it runs REBEL on every chunk

### First-run cost

The first run may be slower because model weights may need to be loaded or downloaded.

### Hugging Face access

Some backends download models from the Hugging Face Hub if they are not already present locally.

Optional environment variables:

- `HF_TOKEN`
  - useful for higher rate limits
- `HF_HUB_OFFLINE=1`
  - useful if models are already cached locally and you want fully offline behavior

### Why chunking matters

Chunking directly affects:

- relation recall
- context preservation
- runtime
- schema richness

Too-small chunks can lose relation context. Too-large chunks can increase noise and runtime. `paragraph` mode with a small overlap is a good default.

## Troubleshooting

### `rapidgraph` command not found

Make sure the package is installed in the active environment:

```bash
pip install rapidGraph
```

### Slow first run

Expected if models are being downloaded or loaded for the first time.

### Hugging Face warnings

Warnings about unauthenticated requests are not fatal. Set `HF_TOKEN` if you want authenticated Hub access.

### Real PyPI vs TestPyPI

To install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rapidGraph
```

For the public release:

```bash
pip install rapidGraph
```

### Neo4j shows only `RELATES_TO`

That is expected. The semantic relationship name is stored in the `relation` property, not as a separate Neo4j relationship type.

### Why `expanded_schema` can be larger than `potential_schema`

`potential_schema` is intentionally strict and compatibility-focused. `expanded_schema` preserves finer type detail and therefore often contains more rows.

## Development

Install development dependencies:

```bash
pip install ".[dev]"
```

Run the main test suite:

```bash
pytest -q tests/test_extract_graph.py
```

Build the package:

```bash
python -m build
```

Validate package metadata:

```bash
python -m twine check dist/*
```

## Publishing

### TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rapidGraph
```

### Real PyPI

```bash
pip install rapidGraph
```

This repository includes GitHub Actions workflows for:

- TestPyPI publishing
- real PyPI publishing

## License

MIT
