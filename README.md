# rapidGraph

`rapidGraph` is a local-first, open-domain text-to-graph extractor for arbitrary text. It turns raw text files or inline text into structured JSON containing:

- `entities`
- `relations`
- `potential_schema`
- `expanded_schema`
- provenance-aware `documents`, `chunks`, and `relation_support`

It is designed for:

- general entity and relation extraction across business, technical, scientific, and mixed-topic text
- CPU-friendly local runs with selectable quality modes
- provenance-aware graph building for future RAG or GraphRAG pipelines
- optional direct Neo4j ingestion

The public distribution name is `rapidGraph`, the Python import package is `rapidgraph`, and the installed CLI command is `rapidgraph`.

## What It Does

At a high level, `rapidGraph`:

1. normalizes raw text
2. splits it into chunked spans
3. extracts entity candidates
4. extracts relation candidates
5. canonicalizes duplicate or near-duplicate entity mentions
6. links relation endpoints back to canonical entities
7. infers schema patterns from the final graph
8. preserves chunk/document provenance for downstream graph and retrieval use

The extractor is open-domain best effort. It does not enforce a fixed ontology and keeps `Unknown` types when typing confidence is weak.

## Core Features

- Open-domain entity extraction
- Open-domain relation extraction
- Schema inference from observed graph edges
- Provenance-aware output with `documents`, `chunks`, and relation support records
- Multi-file corpus ingestion in one run
- Two canonicalization scopes:
  - `document`: keep each file independent
  - `corpus`: merge compatible entities across files
- Three CPU-aware execution modes:
  - `fast`
  - `balanced`
  - `quality`
- Optional embedding-assisted canonicalization and linking
- Optional Neo4j export

## Install

Install from source:

```bash
pip install .
```

Install with optional extras:

```bash
pip install ".[neo4j]"
pip install ".[embeddings]"
pip install ".[dev]"
pip install ".[neo4j,embeddings,dev]"
```

After publishing to PyPI, users will be able to install with:

```bash
pip install rapidGraph
```

PyPI extras will work the same way:

```bash
pip install "rapidGraph[neo4j]"
pip install "rapidGraph[embeddings]"
pip install "rapidGraph[dev]"
```

## CLI Quick Start

Show help:

```bash
rapidgraph --help
```

Process inline text:

```bash
rapidgraph --text "Google is based in California." --pretty
```

Process one file:

```bash
rapidgraph --input input.txt --pretty
```

Process multiple files:

```bash
rapidgraph --input input.txt input2.txt --pretty
```

Write output to JSON:

```bash
rapidgraph --input input.txt --output graph.json --pretty
```

The repo-root compatibility command still works:

```bash
python extract_graph.py --input input.txt --pretty
```

## Execution Modes

`rapidGraph` supports three relation extraction modes.

### `fast`

Best for:

- CPU-only quick passes
- bulk experiments
- basic graph drafts

Behavior:

- uses GLiNER and heuristics
- does not run REBEL
- fastest startup and lowest CPU cost

### `balanced`

This is the default mode.

Best for:

- normal CPU usage
- better relation quality without full model cost

Behavior:

- runs heuristics everywhere
- runs REBEL only on shortlisted high-value spans
- usually the best tradeoff

### `quality`

Best for:

- maximum relation recall
- slower offline analysis
- smaller corpora where quality matters more than throughput

Behavior:

- runs REBEL across all chunks
- highest model cost

## Input Model

The CLI accepts either:

- `--text "..."` for inline text
- `--input file1.txt [file2.txt ...]` for one or more text files

`--text` and `--input` are mutually exclusive.

## Output Model

The extractor returns one combined JSON object with these top-level fields.

### `entities`

Each entity includes:

- `id`
- `text`
- `canonical`
- `type`
- `confidence`
- `mentions`

Each mention includes:

- `text`
- `start`
- `end`
- `chunk_index`
- `document_id`
- `chunk_id`

### `relations`

Each relation includes:

- `source_id`
- `target_id`
- `relation`
- `confidence`
- `evidence`
- `chunk_ids`
- `document_ids`

### `potential_schema`

Strict schema aggregation using:

- `(source_type, relation, target_type)`

This is the backward-compatible schema view.

### `expanded_schema`

Richer schema aggregation using finer-grained normalized types and more examples.

### `documents`

One document row per input source:

- `id`
- `source`
- `title`
- `text_hash`
- `char_count`

### `chunks`

Each chunk includes:

- `id`
- `document_id`
- `index`
- `text` unless omitted
- `start`
- `end`
- `block_index`
- `overlap_sentences`

### `relation_support`

One row per final relation edge with merged provenance:

- `source_id`
- `relation`
- `target_id`
- `chunk_ids`
- `document_ids`
- `evidence`

### `meta`

Includes model names, thresholds, chunk counts, mode, embedding stats, relation backend stats, warnings, and processing time.

## Flag Reference

### Input and Output Flags

#### `--text TEXT`

Inline text input.

Example:

```bash
rapidgraph --text "Transformer uses self-attention." --pretty
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

## Quality and Runtime Flags

#### `--mode {fast,balanced,quality}`

Controls the CPU and quality tradeoff.

Examples:

```bash
rapidgraph --input input.txt --mode fast
rapidgraph --input input.txt --mode balanced
rapidgraph --input input.txt --mode quality
```

#### `--disable-rebel`

Forces heuristic-only relation extraction even if the mode would otherwise use REBEL.

Example:

```bash
rapidgraph --input input.txt --mode quality --disable-rebel
```

#### `--max-model-spans MAX_MODEL_SPANS`

Only used meaningfully in `balanced` mode. Caps the number of shortlisted spans sent to REBEL.

Example:

```bash
rapidgraph --input input.txt --mode balanced --max-model-spans 6
```

## Extraction Threshold Flags

#### `--entity-threshold ENTITY_THRESHOLD`

Minimum confidence used to keep entity candidates.

Example:

```bash
rapidgraph --input input.txt --entity-threshold 0.45
```

#### `--relation-threshold RELATION_THRESHOLD`

Minimum confidence used to keep relations.

Example:

```bash
rapidgraph --input input.txt --relation-threshold 0.3
```

#### `--max-chars MAX_CHARS`

Chunk size budget. Larger values preserve more context but cost more runtime.

Example:

```bash
rapidgraph --input input.txt --max-chars 1400
```

## Chunking Flags

#### `--chunk-mode {paragraph,sentence}`

Controls chunk construction.

- `paragraph`: structure-aware paragraph-first chunking
- `sentence`: simpler sentence packing

Example:

```bash
rapidgraph --input input.txt --chunk-mode paragraph
rapidgraph --input input.txt --chunk-mode sentence
```

#### `--chunk-overlap CHUNK_OVERLAP`

Sentence overlap between neighboring chunks. Higher values preserve context across chunk boundaries but increase redundancy.

Example:

```bash
rapidgraph --input input.txt --chunk-overlap 2
```

## Multi-File and Canonicalization Flags

#### `--entity-scope {document,corpus}`

Controls how entities are canonicalized across multiple files.

- `document`: identical entities in different files stay separate
- `corpus`: compatible entities can merge across files

Examples:

```bash
rapidgraph --input input.txt input2.txt --entity-scope document
rapidgraph --input input.txt input2.txt --entity-scope corpus
```

Use `document` when:

- document-local provenance matters most
- names are ambiguous across files
- you want a safer default

Use `corpus` when:

- the files are about a shared topic
- you want a consolidated graph across the corpus
- you plan to export one merged graph to Neo4j

## Provenance Flags

#### `--include-chunk-text`

Include full chunk text in the `chunks` array. This is the default.

#### `--no-include-chunk-text`

Keep chunk records but omit chunk text.

#### `--omit-provenance-text`

Alias for omitting chunk text while preserving chunk IDs and metadata.

Examples:

```bash
rapidgraph --input input.txt --no-include-chunk-text
rapidgraph --input input.txt --omit-provenance-text
```

## Embedding-Assisted Linking Flags

These are opt-in. They are not enabled by default.

#### `--embedding-linking`

Enable embedding-assisted rescue for ambiguous entity merges and unresolved relation endpoints.

#### `--embedding-model EMBEDDING_MODEL`

Sentence embedding model to use. Default:

```text
sentence-transformers/all-MiniLM-L6-v2
```

#### `--embedding-threshold EMBEDDING_THRESHOLD`

Cosine similarity threshold for accepting embedding-based merges or links.

#### `--embedding-cache-dir EMBEDDING_CACHE_DIR`

Local cache directory for embedding vectors.

#### `--embedding-max-candidates EMBEDDING_MAX_CANDIDATES`

Caps the candidate pool used during embedding-assisted linking.

Examples:

```bash
rapidgraph \
  --input input.txt \
  --embedding-linking \
  --embedding-threshold 0.84 \
  --embedding-cache-dir .cache/extract_graph_embeddings
```

```bash
rapidgraph \
  --input input.txt input2.txt \
  --entity-scope corpus \
  --embedding-linking \
  --embedding-max-candidates 8
```

## Neo4j Flags

These flags are optional. If omitted, the extractor only emits JSON.

#### `--neo4j-uri NEO4J_URI`

Neo4j URI such as:

```text
neo4j://127.0.0.1:7687
```

#### `--neo4j-user NEO4J_USER`

Neo4j username.

#### `--neo4j-password NEO4J_PASSWORD`

Neo4j password.

#### `--neo4j-database NEO4J_DATABASE`

Target Neo4j database name.

#### `--neo4j-clean-document`

Delete matching document subgraphs before re-ingesting them. Useful when rerunning the same document set.

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

## Logging Flag

#### `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`

Controls CLI log verbosity.

Example:

```bash
rapidgraph --input input.txt --log-level DEBUG
```

## Recommended Flag Combinations

### Quick CPU pass

```bash
rapidgraph --input input.txt --mode fast --pretty
```

### Best default for most users

```bash
rapidgraph --input input.txt --mode balanced --pretty
```

### Higher recall on one document

```bash
rapidgraph --input input.txt --mode quality --chunk-overlap 2 --pretty
```

### Multi-file corpus graph

```bash
rapidgraph \
  --input input.txt input2.txt \
  --mode balanced \
  --entity-scope corpus \
  --pretty
```

### Multi-file corpus with stronger cross-file merging

```bash
rapidgraph \
  --input input.txt input2.txt \
  --mode balanced \
  --entity-scope corpus \
  --embedding-linking \
  --pretty
```

### Lean provenance payload

```bash
rapidgraph \
  --input input.txt \
  --omit-provenance-text \
  --pretty
```

### Neo4j export with replacement of existing document graph

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

## Python Library Usage

Basic usage:

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
            text="Google hired Sundar Pichai.",
            source="two.txt",
            title="two.txt",
        ),
    ],
    entity_scope="corpus",
)

print(result.model_dump())
```

## Neo4j Graph Shape

When Neo4j export is enabled, the graph is designed to remain compatible with future GraphRAG workflows.

Current node labels:

- `Document`
- `Chunk`
- `Entity`

Current relationship types:

- `HAS_CHUNK`
- `MENTIONS`
- `RELATES_TO`

The semantic relation name is stored as a property on `RELATES_TO`, which is why Neo4j Browser shows one relationship type while preserving relation semantics in properties.

## Packaging

Build distributions:

```bash
python -m build
```

Validate package metadata:

```bash
python -m twine check dist/*
```

Install from a built wheel:

```bash
pip install dist/rapidgraph-0.1.0-py3-none-any.whl
```

## Publishing to PyPI

Create a PyPI account, generate an API token, then upload:

```bash
python -m twine upload dist/*
```

If the `rapidGraph` name is accepted on PyPI, users will be able to install with:

```bash
pip install rapidGraph
```

## Development

Install dev dependencies:

```bash
pip install ".[dev]"
```

Run tests:

```bash
pytest -q tests/test_extract_graph.py
```

Build the package:

```bash
python -m build
```

## License

MIT
