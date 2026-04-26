from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extract_graph
import rapidgraph.core as rapidgraph_core
from rapidgraph import graphrag
from extract_graph import (
    Chunk,
    DocumentInput,
    EntityCandidate,
    EntityModel,
    GraphExtraction,
    GraphExtractor,
    MetaModel,
    MentionModel,
    RelationCandidate,
    RelationModel,
    SchemaEdgeModel,
    build_expanded_schema,
    build_schema,
    count_rebel_candidate_spans,
    chunk_text,
    extract_context_relations,
    extract_heuristic_entities,
    EmbeddingSession,
    is_relation_plausible,
    refine_entity_label,
    resolve_entity_id,
    select_rebel_candidate_spans,
)


class FakeEntityBackend:
    model_name = "fake_entity_backend"
    fallback_used = False
    warnings: list[str] = []

    def extract(self, chunk: Chunk, threshold: float) -> list[EntityCandidate]:
        del threshold
        candidates: list[EntityCandidate] = []
        known = {
            "Sundar Pichai": "person",
            "Google": "organization",
            "California": "location",
            "CRISPR": "research topic",
            "DNA": "chemical compound",
            "E. coli": "biological species",
            "Apollo 11": "event",
            "Moon": "location",
            "Neil Armstrong": "person",
            "OpenAI": "organization",
            "Sam Altman": "person",
            "San Francisco": "city",
        }
        for surface, label in known.items():
            for match_start in find_all_occurrences(chunk.text, surface):
                candidates.append(
                    EntityCandidate(
                        text=surface,
                        label=label,
                        score=0.9,
                        start=chunk.start + match_start,
                        end=chunk.start + match_start + len(surface),
                        chunk_index=chunk.index,
                    )
                )
        return candidates


class FakeRelationBackend:
    model_name = "fake_relation_backend"
    fallback_used = False
    warnings: list[str] = []

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        del threshold
        relations: list[RelationCandidate] = []
        text = chunk.text
        if "Sundar Pichai" in text and "Google" in text:
            relations.append(
                RelationCandidate(
                    subject="Sundar Pichai",
                    relation="chief executive officer",
                    obj="Google",
                    score=0.88,
                    evidence=text,
                    chunk_index=chunk.index,
                )
            )
        if "Google" in text and "California" in text:
            relations.append(
                RelationCandidate(
                    subject="Google",
                    relation="based in",
                    obj="California",
                    score=0.86,
                    evidence=text,
                    chunk_index=chunk.index,
                )
            )
        if "CRISPR" in text and "DNA" in text:
            relations.append(
                RelationCandidate(
                    subject="CRISPR",
                    relation="edits",
                    obj="DNA",
                    score=0.81,
                    evidence=text,
                    chunk_index=chunk.index,
                )
            )
        if "CRISPR" in text and "E. coli" in text:
            relations.append(
                RelationCandidate(
                    subject="CRISPR",
                    relation="tested in",
                    obj="E. coli",
                    score=0.79,
                    evidence=text,
                    chunk_index=chunk.index,
                )
            )
        if "Apollo 11" in text and "Moon" in text:
            relations.append(
                RelationCandidate(
                    subject="Apollo 11",
                    relation="landed on",
                    obj="Moon",
                    score=0.83,
                    evidence=text,
                    chunk_index=chunk.index,
                )
            )
        if "Neil Armstrong" in text and "Apollo 11" in text:
            relations.append(
                RelationCandidate(
                    subject="Neil Armstrong",
                    relation="commanded",
                    obj="Apollo 11",
                    score=0.8,
                    evidence=text,
                    chunk_index=chunk.index,
                )
            )
        return relations


class CountingRelationBackend:
    model_name = "counting_relation_backend"
    fallback_used = False
    warnings: list[str] = []

    def __init__(self):
        self.calls: list[str] = []

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        del threshold
        self.calls.append(chunk.text)
        return []


class FakeEmbeddingBackend:
    def __init__(self, mapping: dict[tuple[str, str], float] | None = None):
        self.model_name = "fake_embedding_backend"
        self.mapping = mapping or {}
        self.calls = 0

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            score = None
            for (left, right), value in self.mapping.items():
                if text in {left, right}:
                    score = value
                    break
            base = float(sum(ord(char) for char in text) % 97)
            vectors.append([base, score if score is not None else base / 10.0 + 1.0])
        return vectors


def build_fake_extractor() -> GraphExtractor:
    return GraphExtractor(FakeEntityBackend(), FakeRelationBackend(), max_chars=120, mode="quality")


class StubExtractorForCli:
    def __init__(self, fake_result: GraphExtraction, captured: dict[str, object]):
        self.fake_result = fake_result
        self.captured = captured

    def extract_documents(self, documents, **kwargs) -> GraphExtraction:
        self.captured["documents"] = documents
        self.captured.update(kwargs)
        return self.fake_result


def find_all_occurrences(text: str, surface: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        match = text.find(surface, offset)
        if match < 0:
            return starts
        starts.append(match)
        offset = match + len(surface)


def test_business_style_extraction():
    extractor = build_fake_extractor()
    result = extractor.extract(
        "Sundar Pichai is the CEO of Google. Google operates from California."
    )

    assert any(entity.text == "Sundar Pichai" for entity in result.entities)
    assert any(relation.relation == "CHIEF_EXECUTIVE_OFFICER" for relation in result.relations)
    assert any(edge.relation == "BASED_IN" for edge in result.potential_schema)


def test_scientific_text_produces_output():
    extractor = build_fake_extractor()
    result = extractor.extract("CRISPR edits DNA and was tested in E. coli.")

    assert len(result.entities) >= 3
    assert any(relation.relation == "EDITS" for relation in result.relations)
    assert any(edge.source_type == "ResearchTopic" for edge in result.potential_schema)


def test_mixed_topic_keeps_unknown_type_when_needed():
    extractor = GraphExtractor(FakeEntityBackend(), FakeRelationBackend(), max_chars=120, mode="quality")
    result = extractor.extract("Apollo 11 reached the Moon while Neil Armstrong led the mission.")

    assert any(entity.text == "Apollo 11" for entity in result.entities)
    assert any(edge.relation == "LANDED_ON" for edge in result.potential_schema)


def test_entity_deduplication_merges_repeated_mentions():
    extractor = build_fake_extractor()
    result = extractor.extract("Google hired people. Google grew quickly in California.")

    google_entities = [entity for entity in result.entities if entity.text == "Google"]
    assert len(google_entities) == 1
    assert len(google_entities[0].mentions) == 2


def test_cli_stdout_and_output_file(monkeypatch, capsys, tmp_path: Path):
    fake_result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[SchemaEdgeModel(source_type="A", relation="R", target_type="B", count=1, examples=["x"])],
        expanded_schema=[SchemaEdgeModel(source_type="A1", relation="R1", target_type="B1", count=1, examples=["y"])],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        extract_graph,
        "build_default_extractor",
        lambda max_chars=600, **kwargs: StubExtractorForCli(fake_result, captured),
    )

    exit_code = extract_graph.main(["--text", "hello world", "--max-chars", "321"])
    assert exit_code == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["meta"]["entity_model"] == "fake"
    assert captured["documents"][0] == DocumentInput(text="hello world", source="inline", title="inline_text")
    assert captured["max_chars"] == 321
    assert captured["mode"] == "balanced"

    output_path = tmp_path / "graph.json"
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello world", encoding="utf-8")
    exit_code = extract_graph.main(
        ["--input", str(input_path), "--output", str(output_path), "--pretty", "--max-chars", "321"]
    )
    assert exit_code == 0
    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload["potential_schema"][0]["relation"] == "R"
    assert written_payload["expanded_schema"][0]["relation"] == "R1"
    assert written_payload["documents"] == []
    assert written_payload["chunks"] == []
    assert written_payload["relation_support"] == []


def test_cli_forwards_mode_and_disable_rebel(monkeypatch, capsys):
    fake_result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )

    captured: dict[str, object] = {}

    def fake_builder(max_chars=600, **kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForCli(fake_result, captured)

    monkeypatch.setattr(extract_graph, "build_default_extractor", fake_builder)
    exit_code = extract_graph.main(["--text", "hello world", "--mode", "fast", "--disable-rebel", "--max-model-spans", "2"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["meta"]["entity_model"] == "fake"
    assert captured["builder_kwargs"]["mode"] == "fast"
    assert captured["builder_kwargs"]["disable_rebel"] is True
    assert captured["documents"][0].text == "hello world"
    assert captured["mode"] == "fast"
    assert captured["disable_rebel"] is True
    assert captured["max_model_spans"] == 2


def test_cli_forwards_embedding_flags(monkeypatch, capsys):
    fake_result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )
    captured: dict[str, object] = {}

    def fake_builder(max_chars=600, **kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForCli(fake_result, captured)

    monkeypatch.setattr(extract_graph, "build_default_extractor", fake_builder)
    exit_code = extract_graph.main(
        [
            "--text",
            "hello world",
            "--embedding-linking",
            "--embedding-model",
            "fake-model",
            "--embedding-threshold",
            "0.9",
            "--embedding-cache-dir",
            "/tmp/embed-cache",
            "--embedding-max-candidates",
            "3",
        ]
    )
    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured["builder_kwargs"]["embedding_linking"] is True
    assert captured["builder_kwargs"]["embedding_model"] == "fake-model"
    assert captured["builder_kwargs"]["embedding_threshold"] == 0.9
    assert captured["builder_kwargs"]["embedding_cache_dir"] == "/tmp/embed-cache"
    assert captured["builder_kwargs"]["embedding_max_candidates"] == 3
    assert captured["embedding_linking"] is True
    assert captured["embedding_model"] == "fake-model"
    assert captured["embedding_threshold"] == 0.9
    assert captured["embedding_cache_dir"] == "/tmp/embed-cache"
    assert captured["embedding_max_candidates"] == 3


def test_cli_omit_provenance_text_flag(monkeypatch, capsys):
    fake_result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )
    captured: dict[str, object] = {}

    def fake_builder(max_chars=600, **kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForCli(fake_result, captured)

    monkeypatch.setattr(extract_graph, "build_default_extractor", fake_builder)
    exit_code = extract_graph.main(["--text", "hello world", "--omit-provenance-text"])
    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured["include_chunk_text"] is False


def test_cli_forwards_neo4j_flags(monkeypatch, capsys):
    fake_result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )
    captured: dict[str, object] = {}

    def fake_builder(max_chars=600, **kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForCli(fake_result, captured)

    def fake_export(result: GraphExtraction, **kwargs) -> None:
        captured["neo4j_result"] = result
        captured["neo4j_kwargs"] = kwargs

    monkeypatch.setattr(extract_graph, "build_default_extractor", fake_builder)
    monkeypatch.setattr(extract_graph, "export_graph_to_neo4j", fake_export)
    exit_code = extract_graph.main(
        [
            "--text",
            "hello world",
            "--neo4j-uri",
            "neo4j://127.0.0.1:7687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "secret",
            "--neo4j-database",
            "graphrag",
            "--neo4j-clean-document",
        ]
    )
    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured["neo4j_result"] is fake_result
    assert captured["neo4j_kwargs"] == {
        "uri": "neo4j://127.0.0.1:7687",
        "user": "neo4j",
        "password": "secret",
        "database": "graphrag",
        "clean_document": True,
        "embed_chunks": False,
        "create_vector_index": False,
        "vector_index_name": "rapidgraph_chunk_embedding",
        "embedding_property": "embedding",
        "chunk_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_partial_neo4j_flags_raise_system_exit():
    with pytest.raises(SystemExit, match="Neo4j export requires"):
        extract_graph.main(["--text", "hello world", "--neo4j-uri", "neo4j://127.0.0.1:7687"])


def test_vector_index_creation_requires_chunk_embedding_flag():
    with pytest.raises(SystemExit, match="requires --neo4j-embed-chunks"):
        extract_graph.main(
            [
                "--text",
                "hello world",
                "--neo4j-uri",
                "neo4j://127.0.0.1:7687",
                "--neo4j-user",
                "neo4j",
                "--neo4j-password",
                "secret",
                "--neo4j-create-vector-index",
            ]
        )


def test_cli_reads_multiple_input_files_and_forwards_entity_scope(monkeypatch, capsys, tmp_path: Path):
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("Google is based in California.", encoding="utf-8")
    second.write_text("Google hired Sundar Pichai.", encoding="utf-8")
    fake_result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=2,
            elapsed_seconds=0.01,
        ),
    )
    captured: dict[str, object] = {}

    def fake_builder(max_chars=600, **kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForCli(fake_result, captured)

    monkeypatch.setattr(extract_graph, "build_default_extractor", fake_builder)
    exit_code = extract_graph.main(
        ["--input", str(first), str(second), "--entity-scope", "corpus"]
    )
    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    documents = captured["documents"]
    assert len(documents) == 2
    assert documents[0].source == "one.txt"
    assert documents[1].source == "two.txt"
    assert captured["entity_scope"] == "corpus"


def test_context_relations_capture_technical_phrases():
    result = extract_context_relations(
        "Additive attention outperforms dot product attention. We add positional encodings to input embeddings.",
        [
            type("Entity", (), {"id": "E1", "text": "Additive attention", "mentions": [type("Mention", (), {"start": 0, "end": 18})()]}),
            type("Entity", (), {"id": "E2", "text": "dot product attention", "mentions": [type("Mention", (), {"start": 31, "end": 52})()]}),
            type("Entity", (), {"id": "E3", "text": "positional encodings", "mentions": [type("Mention", (), {"start": 62, "end": 82})()]}),
            type("Entity", (), {"id": "E4", "text": "input embeddings", "mentions": [type("Mention", (), {"start": 86, "end": 102})()]}),
        ],
    )

    relation_names = {item.relation for item in result}
    assert "OUTPERFORMS" in relation_names
    assert "ADDED_TO" in relation_names


def test_relation_plausibility_rejects_bad_technical_rebel_edges():
    source = type("EntityModelStub", (), {"text": "Additive attention", "type": "TechnicalConcept"})()
    target = type("EntityModelStub", (), {"text": "Dot-product attention", "type": "TechnicalConcept"})()

    assert not is_relation_plausible(
        "OPPOSITE_OF",
        source,
        target,
        "Additive attention outperforms dot product attention.",
    )
    assert is_relation_plausible(
        "OUTPERFORMS",
        source,
        target,
        "Additive attention outperforms dot product attention.",
    )


def test_chunk_text_paragraph_mode_preserves_boundaries():
    text = "Alpha builds systems.\n\nBeta studies models."
    chunks = chunk_text(text, max_chars=200, chunk_mode="paragraph", chunk_overlap=0)

    assert len(chunks) == 2
    assert chunks[0].block_index == 0
    assert chunks[1].block_index == 1
    assert chunks[0].text == "Alpha builds systems."
    assert chunks[1].text == "Beta studies models."


def test_chunk_text_overlap_repeats_boundary_sentence():
    text = "Sentence one. Sentence two is longer. Sentence three."
    chunks = chunk_text(text, max_chars=40, chunk_mode="sentence", chunk_overlap=1)

    assert len(chunks) >= 2
    assert "Sentence two is longer." in chunks[0].text
    assert "Sentence two is longer." in chunks[1].text


def test_provenance_records_are_emitted_with_stable_ids():
    extractor = build_fake_extractor()
    result = extractor.extract("Google is based in California.")

    assert len(result.documents) == 1
    assert result.documents[0].id.startswith("D")
    assert result.documents[0].source == "inline"
    assert result.documents[0].title == "inline_text"
    assert len(result.chunks) == result.meta.chunk_count
    assert all(chunk.id.startswith(f"{result.documents[0].id}:C") for chunk in result.chunks)
    assert result.meta.document_count == 1
    assert result.meta.chunk_records == len(result.chunks)
    assert result.meta.chunk_text_included is True


def test_mentions_include_document_and_chunk_ids():
    extractor = build_fake_extractor()
    result = extractor.extract("Google is based in California.")
    chunk_ids = {chunk.id for chunk in result.chunks}

    for entity in result.entities:
        for mention in entity.mentions:
            assert mention.document_id == result.documents[0].id
            assert mention.chunk_id in chunk_ids


def test_relations_include_chunk_and_document_ids():
    extractor = build_fake_extractor()
    result = extractor.extract("Google is based in California.")

    assert result.relations
    for relation in result.relations:
        assert relation.document_ids == [result.documents[0].id]
        assert relation.chunk_ids


def test_relation_support_aligns_with_final_relations():
    extractor = build_fake_extractor()
    result = extractor.extract("Google is based in California.")
    support_keys = {(item.source_id, item.relation, item.target_id) for item in result.relation_support}
    relation_keys = {(item.source_id, item.relation, item.target_id) for item in result.relations}

    assert support_keys == relation_keys


class DuplicateRelationBackend:
    model_name = "duplicate_relation_backend"
    fallback_used = False
    warnings: list[str] = []

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        del threshold
        relations: list[RelationCandidate] = []
        if "Google" in chunk.text and "California" in chunk.text:
            relations.append(
                RelationCandidate(
                    subject="Google",
                    relation="based in",
                    obj="California",
                    score=0.8,
                    evidence=chunk.text,
                    chunk_index=chunk.index,
                )
            )
        return relations


def test_repeated_relation_support_collapses_to_one_relation_with_multiple_chunks():
    extractor = GraphExtractor(FakeEntityBackend(), DuplicateRelationBackend(), max_chars=40, mode="quality")
    result = extractor.extract("Google is based in California. Google expanded in California.")

    based_in = [relation for relation in result.relations if relation.relation == "BASED_IN"]
    assert len(based_in) == 1
    assert len(based_in[0].chunk_ids) >= 2
    assert len(result.relation_support) == len(result.relations)


def test_file_input_sets_document_source_and_title(tmp_path: Path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text("Google is based in California.", encoding="utf-8")

    text, source, title = extract_graph.read_input_text(
        type("Args", (), {"text": None, "input": str(input_path)})()
    )

    assert text == "Google is based in California."
    assert source == "sample.txt"
    assert title == "sample.txt"


def test_omit_provenance_text_keeps_chunks_without_text():
    extractor = build_fake_extractor()
    result = extractor.extract("Google is based in California.", include_chunk_text=False)

    assert result.chunks
    assert all(chunk.text == "" for chunk in result.chunks)
    assert result.meta.chunk_text_included is False


def test_multifile_document_scope_keeps_same_entity_separate():
    extractor = build_fake_extractor()
    result = extractor.extract_documents(
        [
            DocumentInput(text="Google is based in California.", source="one.txt", title="one.txt"),
            DocumentInput(text="Google hired people in California.", source="two.txt", title="two.txt"),
        ],
        entity_scope="document",
    )

    google_entities = [entity for entity in result.entities if entity.text == "Google"]
    assert len(result.documents) == 2
    assert len(google_entities) == 2
    assert {mention.document_id for entity in google_entities for mention in entity.mentions} == {
        result.documents[0].id,
        result.documents[1].id,
    }


def test_multifile_corpus_scope_merges_same_entity_across_documents():
    extractor = build_fake_extractor()
    result = extractor.extract_documents(
        [
            DocumentInput(text="Google is based in California.", source="one.txt", title="one.txt"),
            DocumentInput(text="Google hired people in California.", source="two.txt", title="two.txt"),
        ],
        entity_scope="corpus",
    )

    google_entities = [entity for entity in result.entities if entity.text == "Google"]
    assert len(result.documents) == 2
    assert len(google_entities) == 1
    assert {mention.document_id for mention in google_entities[0].mentions} == {
        result.documents[0].id,
        result.documents[1].id,
    }


def test_refine_entity_label_produces_specific_technical_types():
    assert refine_entity_label("Transformer", "Unknown") == "Model"
    assert refine_entity_label("encoder", "Unknown") == "Component"
    assert refine_entity_label("softmax", "Unknown") == "Transformation"
    assert refine_entity_label("embeddings", "Unknown") == "DataStructure"
    assert refine_entity_label("accuracy", "Unknown") == "Metric"


def test_heuristic_entities_reject_document_sections():
    chunk = Chunk(index=0, text="3.4 Embeddings and Softmax\nLearned embeddings improve results.", start=0, end=61)
    entities = extract_heuristic_entities(chunk)

    assert all(entity.text != "3.4 Embeddings and Softmax" for entity in entities)
    assert any("embeddings" in entity.text.casefold() for entity in entities)


def test_resolve_entity_id_prefers_sentence_local_mentions():
    entities = [
        EntityModel(
            id="E1",
            text="Attention",
            canonical="Attention",
            type="Method",
            confidence=0.8,
            mentions=[MentionModel(text="Attention", start=0, end=9, chunk_index=0)],
        ),
        EntityModel(
            id="E2",
            text="Multi-Head Attention",
            canonical="Multi-Head Attention",
            type="Method",
            confidence=0.9,
            mentions=[MentionModel(text="Multi-Head Attention", start=20, end=40, chunk_index=1)],
        ),
    ]
    entity_lookup = {extract_graph.normalize_surface(entity.text): entity.id for entity in entities}

    resolved = resolve_entity_id(
        "multi-head attention",
        entity_lookup,
        entities,
        chunk_index=1,
        evidence="The Transformer uses multi-head attention.",
    )

    assert resolved == "E2"


def test_context_relations_capture_part_of_direction():
    result = extract_context_relations(
        "The encoder stack is part of the Transformer.",
        [
            type("Entity", (), {"id": "E1", "text": "encoder stack", "mentions": [type("Mention", (), {"start": 4, "end": 17})()]}),
            type("Entity", (), {"id": "E2", "text": "Transformer", "mentions": [type("Mention", (), {"start": 34, "end": 45})()]}),
        ],
    )

    assert any(item.relation == "PART_OF" and item.subject == "encoder stack" and item.obj == "Transformer" for item in result)


def test_relation_plausibility_rejects_generic_derived_from_for_encoder_output():
    source = type("EntityModelStub", (), {"text": "memory keys", "type": "Component"})()
    target = type("EntityModelStub", (), {"text": "encoder", "type": "Component"})()

    assert not is_relation_plausible(
        "DERIVED_FROM",
        source,
        target,
        "The memory keys and values come from the output of the encoder.",
    )


def test_relation_plausibility_rejects_malformed_rebel_relation_names():
    source = type("EntityModelStub", (), {"text": "attention", "type": "Method"})()
    target = type("EntityModelStub", (), {"text": "Transformer", "type": "Model"})()

    assert not is_relation_plausible("2_LEFT", source, target, "Multi-head attention has 2 left branches.")
    assert not is_relation_plausible("WHERE_HEADI", source, target, "where headi is a projected head.")
    assert not is_relation_plausible(
        "COMPUTE_THE_MATRIX_OF_OUTPUTS_AS",
        source,
        target,
        "We compute the matrix of outputs as attention.",
    )


def test_expanded_schema_is_richer_than_compat_schema():
    entities = [
        EntityModel(
            id="E1",
            text="Transformer",
            canonical="Transformer",
            type="Model",
            confidence=0.9,
            mentions=[MentionModel(text="Transformer", start=0, end=11, chunk_index=0)],
        ),
        EntityModel(
            id="E2",
            text="encoder",
            canonical="encoder",
            type="Component",
            confidence=0.9,
            mentions=[MentionModel(text="encoder", start=20, end=27, chunk_index=0)],
        ),
        EntityModel(
            id="E3",
            text="Multi-Head Attention",
            canonical="Multi-Head Attention",
            type="Method",
            confidence=0.9,
            mentions=[MentionModel(text="Multi-Head Attention", start=30, end=50, chunk_index=0)],
        ),
    ]
    relations = [
        RelationModel(source_id="E1", target_id="E2", relation="USES", confidence=0.9, evidence="x"),
        RelationModel(source_id="E1", target_id="E3", relation="USES", confidence=0.9, evidence="y"),
    ]

    potential = build_schema(entities, relations)
    expanded = build_expanded_schema(entities, relations)

    assert len(potential) == 1
    assert len(expanded) == 2


def test_fast_mode_does_not_call_rebel_backend():
    backend = CountingRelationBackend()
    extractor = GraphExtractor(
        FakeEntityBackend(),
        backend,
        max_chars=120,
        mode="fast",
    )

    extractor.extract("Sundar Pichai leads Google in California.")

    assert backend.calls == []


def test_balanced_mode_only_runs_rebel_on_shortlisted_spans():
    backend = CountingRelationBackend()
    extractor = GraphExtractor(
        FakeEntityBackend(),
        backend,
        max_chars=120,
        mode="balanced",
        max_model_spans=1,
    )

    extractor.extract(
        "Google uses OpenAI in California. "
        "Neil Armstrong uses Apollo 11 on the Moon."
    )

    assert len(backend.calls) == 1


def test_quality_mode_routes_all_chunks_through_rebel():
    backend = CountingRelationBackend()
    extractor = GraphExtractor(
        FakeEntityBackend(),
        backend,
        max_chars=40,
        mode="quality",
    )

    result = extractor.extract(
        "Sundar Pichai leads Google. Google is in California. Neil Armstrong commanded Apollo 11."
    )

    assert len(backend.calls) == result.meta.chunk_count
    assert result.meta.relation_backend_strategy == "full_rebel"


def test_rebel_span_selection_skips_low_signal_and_caps_count():
    text = (
        "Intro heading.\n\n"
        "Google uses OpenAI in California. "
        "Neil Armstrong uses Apollo 11 on the Moon. "
        "OpenAI uses Sam Altman in San Francisco."
    )
    extractor = build_fake_extractor()
    entities = extractor.extract(text).entities

    selected = select_rebel_candidate_spans(text, entities, max_model_spans=2)

    assert count_rebel_candidate_spans(text, entities) >= 2
    assert len(selected) == 2
    assert all("heading" not in chunk.text.casefold() for chunk in selected)


class BorderlineEntityBackend:
    model_name = "borderline_entity_backend"
    fallback_used = False
    warnings: list[str] = []

    def extract(self, chunk: Chunk, threshold: float) -> list[EntityCandidate]:
        del threshold, chunk
        return [
            EntityCandidate(text="encoder layer", label="Unknown", score=0.9, start=0, end=13, chunk_index=0),
            EntityCandidate(text="encoding layer", label="Unknown", score=0.9, start=25, end=39, chunk_index=0),
        ]


class SimilarityEmbeddingBackend:
    def __init__(self, similarities: dict[frozenset[str], float]):
        self.model_name = "similarity_embedding_backend"
        self.similarities = similarities
        self.lookup: dict[str, list[float]] = {}
        self.calls = 0

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        for left_right, similarity in self.similarities.items():
            left, right = tuple(left_right)
            self.lookup[left] = [1.0, 0.0]
            self.lookup[right] = [similarity, (1.0 - similarity**2) ** 0.5 if similarity < 1.0 else 0.0]
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self.lookup.get(text, [0.0, 1.0]))
        return vectors


def build_embedding_session(tmp_path: Path, backend, *, enabled: bool = True, max_candidates: int = 8) -> EmbeddingSession:
    return EmbeddingSession(
        enabled=enabled,
        model_name=backend.model_name,
        threshold=0.84,
        max_candidates=max_candidates,
        cache_dir=tmp_path / "embed-cache",
        backend=backend,
    )


def test_embedding_path_is_not_used_when_flag_is_off(tmp_path: Path):
    backend = SimilarityEmbeddingBackend({frozenset({"encoder layer", "encoding layer"}): 0.92})
    extractor = GraphExtractor(
        BorderlineEntityBackend(),
        None,
        embedding_linking=False,
        embedding_backend=backend,
    )

    result = extractor.extract("ignored text")

    assert len(result.entities) == 2
    assert backend.calls == 0
    assert not result.meta.embedding_enabled


def test_embedding_model_loading_is_deferred_until_ambiguous_case(tmp_path: Path):
    backend = SimilarityEmbeddingBackend({frozenset({"encoder layer", "encoding layer"}): 0.92})
    extractor = GraphExtractor(
        FakeEntityBackend(),
        None,
        embedding_linking=True,
        embedding_backend=backend,
        embedding_cache_dir=tmp_path / "embed-cache",
    )

    extractor.extract("Google uses OpenAI in California.")

    assert backend.calls == 0


def test_borderline_duplicate_mentions_merge_with_embeddings(tmp_path: Path):
    backend = SimilarityEmbeddingBackend({frozenset({"encoder layer", "encoding layer"}): 0.92})
    extractor = GraphExtractor(
        BorderlineEntityBackend(),
        None,
        embedding_linking=True,
        embedding_backend=backend,
        embedding_cache_dir=tmp_path / "embed-cache",
    )

    result = extractor.extract("ignored text")

    assert len(result.entities) == 1
    assert result.meta.embedding_merges == 1
    assert backend.calls > 0


def test_unrelated_terms_do_not_merge_despite_embedding_pressure(tmp_path: Path):
    backend = SimilarityEmbeddingBackend({frozenset({"encoder layer", "decoder stack"}): 0.97})
    session = build_embedding_session(tmp_path, backend)
    current_group = [EntityCandidate(text="encoder layer", label="Component", score=0.9, start=0, end=13, chunk_index=0)]
    existing_group = [EntityCandidate(text="decoder stack", label="Component", score=0.9, start=20, end=33, chunk_index=0)]

    assert not extract_graph.is_embedding_merge_candidate(
        current_group,
        existing_group,
        string_score=55,
        current_type="Component",
        reference_type="Component",
    )
    assert backend.calls == 0
    assert session.comparisons == 0


def test_embedding_merge_gate_rejects_extra_modifier_technical_terms():
    current_group = [EntityCandidate(text="Dot-product attention", label="Method", score=0.9, start=0, end=21, chunk_index=0)]
    existing_group = [EntityCandidate(text="Scaled Dot-Product Attention", label="Method", score=0.9, start=25, end=53, chunk_index=0)]

    assert not extract_graph.is_embedding_merge_candidate(
        current_group,
        existing_group,
        string_score=82,
        current_type="Method",
        reference_type="Method",
    )


def test_endpoint_linking_uses_embeddings_only_after_fuzzy_fails(tmp_path: Path):
    backend = SimilarityEmbeddingBackend({frozenset({"search company", "Google"}): 0.91})
    session = build_embedding_session(tmp_path, backend)
    entities = [
        EntityModel(
            id="E1",
            text="Google",
            canonical="Google",
            type="Organization",
            confidence=0.9,
            mentions=[MentionModel(text="Google", start=0, end=6, chunk_index=0)],
        )
    ]
    entity_lookup = {extract_graph.normalize_surface(entity.text): entity.id for entity in entities}

    resolved = resolve_entity_id(
        "search company",
        entity_lookup,
        entities,
        evidence="The search company released a product.",
        embedding_session=session,
    )

    assert resolved == "E1"
    assert session.links == 1
    assert backend.calls > 0


def test_embedding_max_candidates_caps_comparisons(tmp_path: Path):
    backend = SimilarityEmbeddingBackend(
        {
            frozenset({"query layer", "query component 0"}): 0.85,
            frozenset({"query layer", "query component 1"}): 0.86,
            frozenset({"query layer", "query component 2"}): 0.87,
            frozenset({"query layer", "query component 3"}): 0.88,
        }
    )
    session = build_embedding_session(tmp_path, backend, max_candidates=2)
    entities = [
        EntityModel(
            id=f"E{index}",
            text=f"query component {index}",
            canonical=f"query component {index}",
            type="Component",
            confidence=0.8,
            mentions=[MentionModel(text=f"query component {index}", start=index * 10, end=index * 10 + 17, chunk_index=0)],
        )
        for index in range(4)
    ]
    entity_lookup = {extract_graph.normalize_surface(entity.text): entity.id for entity in entities}

    resolve_entity_id(
        "query layer",
        entity_lookup,
        entities,
        evidence="query layer interacts with query component 0 and query component 1.",
        embedding_session=session,
    )

    assert session.comparisons == 2


class LinkingRelationBackend:
    model_name = "linking_relation_backend"
    fallback_used = False
    warnings: list[str] = []

    def extract(self, chunk: Chunk, threshold: float) -> list[RelationCandidate]:
        del threshold
        return [
            RelationCandidate(
                subject="search company",
                relation="based in",
                obj="California",
                score=0.8,
                evidence=chunk.text,
                chunk_index=chunk.index,
            )
        ]


def test_embedding_cache_hit_and_miss_counters_populate(tmp_path: Path):
    backend = SimilarityEmbeddingBackend({frozenset({"search company", "Google"}): 0.91})
    extractor = GraphExtractor(
        FakeEntityBackend(),
        LinkingRelationBackend(),
        embedding_linking=True,
        embedding_backend=backend,
        embedding_cache_dir=tmp_path / "embed-cache",
    )

    first = extractor.extract("The search company expanded in California.")
    second = extractor.extract("The search company expanded in California.")

    assert first.meta.embedding_cache_misses > 0
    assert second.meta.embedding_cache_hits > 0


class FakeNeo4jSession:
    def __init__(self, *, database: str):
        self.database = database
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str, **params) -> None:
        self.calls.append((query, params))


class FakeNeo4jDriver:
    def __init__(self):
        self.closed = False
        self.sessions: list[FakeNeo4jSession] = []

    def session(self, *, database: str):
        session = FakeNeo4jSession(database=database)
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


def test_export_graph_to_neo4j_writes_expected_batches():
    result = GraphExtraction(
        entities=[
            EntityModel(
                id="E1",
                text="Google",
                canonical="Google",
                type="Organization",
                confidence=0.95,
                mentions=[
                    MentionModel(
                        text="Google",
                        start=0,
                        end=6,
                        chunk_index=0,
                        document_id="D1",
                        chunk_id="D1:C0",
                    )
                ],
            ),
            EntityModel(
                id="E2",
                text="California",
                canonical="California",
                type="Location",
                confidence=0.9,
                mentions=[
                    MentionModel(
                        text="California",
                        start=20,
                        end=30,
                        chunk_index=0,
                        document_id="D1",
                        chunk_id="D1:C0",
                    )
                ],
            ),
        ],
        relations=[
            RelationModel(
                source_id="E1",
                target_id="E2",
                relation="BASED_IN",
                confidence=0.88,
                evidence="Google is based in California.",
                chunk_ids=["D1:C0"],
                document_ids=["D1"],
            )
        ],
        potential_schema=[],
        expanded_schema=[],
        documents=[
            extract_graph.DocumentModel(
                id="D1",
                source="inline",
                title="inline_text",
                text_hash="abc",
                char_count=31,
            )
        ],
        chunks=[
            extract_graph.ChunkModel(
                id="D1:C0",
                document_id="D1",
                index=0,
                text="Google is based in California.",
                start=0,
                end=31,
                block_index=0,
                overlap_sentences=0,
            )
        ],
        relation_support=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )
    fake_driver = FakeNeo4jDriver()

    extract_graph.export_graph_to_neo4j(
        result,
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="secret",
        database="graphrag",
        driver_factory=lambda uri, auth: fake_driver,
    )

    assert fake_driver.closed is True
    assert len(fake_driver.sessions) == 1
    session = fake_driver.sessions[0]
    assert session.database == "graphrag"
    assert len(session.calls) == 8
    assert session.calls[3][1]["rows"][0]["id"] == "D1"
    assert session.calls[4][1]["rows"][0]["id"] == "D1:C0"
    assert session.calls[5][1]["rows"][0]["id"] == "E1"
    assert session.calls[6][1]["rows"][0]["entity_id"] == "E1"
    assert session.calls[7][1]["rows"][0]["relation"] == "BASED_IN"


def test_export_graph_to_neo4j_cleans_document_before_reingest():
    result = GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        documents=[
            extract_graph.DocumentModel(
                id="D1",
                source="inline",
                title="inline_text",
                text_hash="abc",
                char_count=31,
            )
        ],
        chunks=[],
        relation_support=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=0,
            elapsed_seconds=0.01,
        ),
    )
    fake_driver = FakeNeo4jDriver()

    extract_graph.export_graph_to_neo4j(
        result,
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="secret",
        database="graphrag",
        clean_document=True,
        driver_factory=lambda uri, auth: fake_driver,
    )

    session = fake_driver.sessions[0]
    assert len(session.calls) == 6
    assert session.calls[3][0].strip() == extract_graph.NEO4J_DELETE_DOCUMENT_RELATIONS_QUERY.strip()
    assert session.calls[3][1] == {"document_id": "D1"}
    assert session.calls[4][0].strip() == extract_graph.NEO4J_DELETE_DOCUMENT_SUBGRAPH_QUERY.strip()
    assert session.calls[4][1] == {"document_id": "D1"}
    assert session.calls[5][0].strip() == extract_graph.NEO4J_DOCUMENT_QUERY.strip()


class StaticEmbeddingBackend:
    model_name = "static_embedding_backend"

    def __init__(self, vector: list[float] | None = None):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.calls: list[list[str]] = []

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [self.vector for _ in texts]


def build_minimal_graph_for_neo4j_export() -> GraphExtraction:
    return GraphExtraction(
        entities=[],
        relations=[],
        potential_schema=[],
        expanded_schema=[],
        documents=[
            extract_graph.DocumentModel(
                id="D1",
                source="inline",
                title="inline_text",
                text_hash="abc",
                char_count=31,
            )
        ],
        chunks=[
            extract_graph.ChunkModel(
                id="D1:C0",
                document_id="D1",
                index=0,
                text="Google is based in California.",
                start=0,
                end=31,
                block_index=0,
                overlap_sentences=0,
            )
        ],
        relation_support=[],
        meta=MetaModel(
            entity_model="fake",
            relation_model="fake",
            entity_threshold=0.1,
            relation_threshold=0.2,
            chunk_count=1,
            elapsed_seconds=0.01,
        ),
    )


def test_export_graph_to_neo4j_embeds_chunks_and_creates_vector_index():
    result = build_minimal_graph_for_neo4j_export()
    fake_driver = FakeNeo4jDriver()
    embedding_backend = StaticEmbeddingBackend([0.4, 0.5, 0.6])

    extract_graph.export_graph_to_neo4j(
        result,
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="secret",
        database="graphrag",
        embed_chunks=True,
        create_vector_index=True,
        vector_index_name="rapidgraph_chunk_embedding",
        embedding_property="embedding",
        chunk_embedding_model="fake-embedder",
        embedding_backend=embedding_backend,
        driver_factory=lambda uri, auth: fake_driver,
    )

    session = fake_driver.sessions[0]
    assert embedding_backend.calls == [["Google is based in California."]]
    assert len(session.calls) == 7
    assert "CREATE VECTOR INDEX `rapidgraph_chunk_embedding`" in session.calls[5][0]
    assert "c.`embedding`" in session.calls[5][0]
    assert session.calls[5][1] == {"dimension": 3, "similarity_function": "cosine"}
    assert "SET c.`embedding` = row.embedding" in session.calls[6][0]
    assert session.calls[6][1]["rows"][0]["embedding"] == [0.4, 0.5, 0.6]
    assert session.calls[6][1]["rows"][0]["embedding_model"] == "fake-embedder"


def test_export_graph_to_neo4j_does_not_embed_chunks_by_default():
    result = build_minimal_graph_for_neo4j_export()
    fake_driver = FakeNeo4jDriver()
    embedding_backend = StaticEmbeddingBackend()

    extract_graph.export_graph_to_neo4j(
        result,
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="secret",
        database="graphrag",
        embedding_backend=embedding_backend,
        driver_factory=lambda uri, auth: fake_driver,
    )

    assert embedding_backend.calls == []
    assert len(fake_driver.sessions[0].calls) == 5


class FakeGraphRAGSession:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "db.index.vector.queryNodes" in query:
            return [
                {
                    "id": "D1:C0",
                    "document_id": "D1",
                    "text": "Transformer uses attention.",
                    "embedding_model": "fake-embedder",
                    "score": 0.93,
                    "source": "input.txt",
                    "title": "input.txt",
                }
            ]
        return [
            {
                "source_id": "E1",
                "source_text": "Transformer",
                "relation": "USES",
                "target_id": "E2",
                "target_text": "attention",
                "evidence": "Transformer uses attention.",
                "chunk_ids": ["D1:C0"],
                "document_ids": ["D1"],
                "confidence": 0.88,
            }
        ]


class FakeGraphRAGDriver:
    def __init__(self, session):
        self.session_obj = session
        self.closed = False

    def session(self, *, database: str):
        self.database = database
        return self.session_obj

    def close(self):
        self.closed = True


def test_neo4j_vector_retriever_queries_index_and_expands_facts():
    session = FakeGraphRAGSession()
    driver = FakeGraphRAGDriver(session)
    backend = StaticEmbeddingBackend([0.1, 0.2])
    retriever = graphrag.Neo4jVectorRetriever(
        uri="neo4j://example",
        user="neo4j",
        password="secret",
        database="neo4j",
        embedding_model="fake-embedder",
        embedding_backend=backend,
        driver_factory=lambda uri, auth: driver,
    )

    chunks, facts, meta = retriever.retrieve(
        "How does Transformer use attention?",
        top_k=3,
        graph_depth=1,
        max_facts=7,
    )

    assert driver.closed is True
    assert len(chunks) == 1
    assert chunks[0].id == "D1:C0"
    assert facts[0].relation == "USES"
    assert meta["warnings"] == []
    assert session.calls[0][1]["index_name"] == "rapidgraph_chunk_embedding"
    assert session.calls[0][1]["top_k"] == 3
    assert session.calls[1][1]["chunk_ids"] == ["D1:C0"]
    assert session.calls[1][1]["max_facts"] == 7


def test_graphrag_client_returns_empty_answer_when_no_chunks():
    class EmptyRetriever:
        def retrieve(self, question, *, top_k, graph_depth, max_facts):
            return [], [], {"warnings": []}

    class FailingLLM:
        def generate(self, prompt):
            raise AssertionError("LLM should not be called without context")

    answer = graphrag.GraphRAGClient(retriever=EmptyRetriever(), llm=FailingLLM()).ask("What happened?")

    assert answer.answer == "No relevant graph context was found for the question."
    assert answer.sources == []
    assert answer.facts == []


def test_ollama_llm_sends_expected_payload_and_parses_response():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "Answer text"}

    class FakeHTTPClient:
        def __init__(self):
            self.calls = []

        def post(self, url, *, json, timeout):
            self.calls.append((url, json, timeout))
            return FakeResponse()

    client = FakeHTTPClient()
    llm = graphrag.OllamaLLM(model="llama3.2", host="http://ollama.local", http_client=client)

    assert llm.generate("hello") == "Answer text"
    assert client.calls[0][0] == "http://ollama.local/api/generate"
    assert client.calls[0][1] == {"model": "llama3.2", "prompt": "hello", "stream": False}


def test_ollama_llm_malformed_response_raises_clear_error():
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "missing response"}

    class FakeHTTPClient:
        def post(self, url, *, json, timeout):
            return FakeResponse()

    llm = graphrag.OllamaLLM(model="llama3.2", http_client=FakeHTTPClient())

    with pytest.raises(RuntimeError, match="string `response` field"):
        llm.generate("hello")


def test_cli_ask_parses_and_forwards_graphrag_flags(monkeypatch, capsys):
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, **kwargs):
            captured["retriever_kwargs"] = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

    class FakeClient:
        def __init__(self, *, retriever, llm):
            captured["retriever"] = retriever
            captured["llm"] = llm

        def ask(self, question, *, top_k, graph_depth, max_facts):
            captured["ask"] = {
                "question": question,
                "top_k": top_k,
                "graph_depth": graph_depth,
                "max_facts": max_facts,
            }
            return graphrag.GraphRAGAnswer(
                answer="answer",
                sources=[],
                facts=[],
                meta={"ok": True},
            )

    monkeypatch.setattr(graphrag, "Neo4jVectorRetriever", FakeRetriever)
    monkeypatch.setattr(graphrag, "OllamaLLM", FakeLLM)
    monkeypatch.setattr(graphrag, "GraphRAGClient", FakeClient)

    exit_code = extract_graph.main(
        [
            "ask",
            "--question",
            "What uses attention?",
            "--neo4j-uri",
            "neo4j://127.0.0.1:7687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "secret",
            "--neo4j-database",
            "graphrag",
            "--neo4j-vector-index-name",
            "my_index",
            "--chunk-embedding-model",
            "fake-embedder",
            "--top-k",
            "4",
            "--graph-depth",
            "1",
            "--max-facts",
            "9",
            "--ollama-host",
            "http://ollama.local",
            "--ollama-model",
            "llama3.2",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "answer"
    assert captured["retriever_kwargs"]["database"] == "graphrag"
    assert captured["retriever_kwargs"]["vector_index_name"] == "my_index"
    assert captured["retriever_kwargs"]["embedding_model"] == "fake-embedder"
    assert captured["llm_kwargs"] == {"model": "llama3.2", "host": "http://ollama.local"}
    assert captured["ask"] == {
        "question": "What uses attention?",
        "top_k": 4,
        "graph_depth": 1,
        "max_facts": 9,
    }


def test_cli_ask_requires_credentials():
    with pytest.raises(SystemExit):
        extract_graph.main(["ask", "--question", "hello", "--ollama-model", "llama3.2"])


class StubExtractorForPythonApi:
    def __init__(self, fake_result: GraphExtraction, captured: dict[str, object]):
        self.fake_result = fake_result
        self.captured = captured

    def extract(self, text, **kwargs):
        self.captured["text"] = text
        self.captured["extract_kwargs"] = kwargs
        return self.fake_result

    def extract_documents(self, documents, **kwargs):
        self.captured["documents"] = documents
        self.captured["extract_documents_kwargs"] = kwargs
        return self.fake_result


def test_extract_text_forwards_options_and_returns_graph(monkeypatch):
    result = build_minimal_graph_for_neo4j_export()
    captured: dict[str, object] = {}

    def fake_build_default_extractor(**kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForPythonApi(result, captured)

    monkeypatch.setattr(rapidgraph_core, "build_default_extractor", fake_build_default_extractor)

    actual = extract_graph.extract_text(
        "Transformer uses attention.",
        entity_threshold=0.4,
        relation_threshold=0.3,
        max_chars=700,
        chunk_mode="sentence",
        chunk_overlap=2,
        mode="fast",
        max_model_spans=3,
        disable_rebel=True,
        embedding_linking=True,
        embedding_model="fake-embedding",
        embedding_threshold=0.9,
        embedding_cache_dir=".cache/test",
        embedding_max_candidates=5,
        document_source="inline-test",
        document_title="Inline Test",
        include_chunk_text=False,
        entity_scope="corpus",
    )

    assert actual is result
    assert captured["builder_kwargs"]["mode"] == "fast"
    assert captured["builder_kwargs"]["embedding_linking"] is True
    assert captured["text"] == "Transformer uses attention."
    assert captured["extract_kwargs"]["entity_threshold"] == 0.4
    assert captured["extract_kwargs"]["relation_threshold"] == 0.3
    assert captured["extract_kwargs"]["document_source"] == "inline-test"
    assert captured["extract_kwargs"]["document_title"] == "Inline Test"
    assert captured["extract_kwargs"]["include_chunk_text"] is False
    assert captured["extract_kwargs"]["entity_scope"] == "corpus"


def test_extract_files_reads_utf8_files_and_forwards_entity_scope(monkeypatch, tmp_path: Path):
    result = build_minimal_graph_for_neo4j_export()
    captured: dict[str, object] = {}
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("Google is based in California.", encoding="utf-8")
    second.write_text("Sundar Pichai leads Google.", encoding="utf-8")

    def fake_build_default_extractor(**kwargs):
        captured["builder_kwargs"] = kwargs
        return StubExtractorForPythonApi(result, captured)

    monkeypatch.setattr(rapidgraph_core, "build_default_extractor", fake_build_default_extractor)

    actual = extract_graph.extract_files(
        [first, second],
        mode="balanced",
        max_model_spans=9,
        include_chunk_text=False,
        entity_scope="corpus",
    )

    assert actual is result
    assert captured["builder_kwargs"]["mode"] == "balanced"
    assert captured["builder_kwargs"]["max_model_spans"] == 9
    documents = captured["documents"]
    assert [document.text for document in documents] == [
        "Google is based in California.",
        "Sundar Pichai leads Google.",
    ]
    assert [document.source for document in documents] == ["one.txt", "two.txt"]
    assert [document.title for document in documents] == ["one.txt", "two.txt"]
    assert captured["extract_documents_kwargs"]["include_chunk_text"] is False
    assert captured["extract_documents_kwargs"]["entity_scope"] == "corpus"


def test_write_json_supports_compact_and_pretty_output(tmp_path: Path):
    result = build_minimal_graph_for_neo4j_export()
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"

    extract_graph.write_json(result, compact)
    extract_graph.write_json(result, pretty, pretty=True)

    compact_text = compact.read_text(encoding="utf-8")
    pretty_text = pretty.read_text(encoding="utf-8")
    assert "\n  " not in compact_text
    assert "\n  " in pretty_text
    assert json.loads(compact_text)["documents"][0]["id"] == "D1"
    assert json.loads(pretty_text)["documents"][0]["id"] == "D1"


def test_neo4j_graph_writer_delegates_to_export(monkeypatch):
    result = build_minimal_graph_for_neo4j_export()
    captured: dict[str, object] = {}
    embedding_backend = StaticEmbeddingBackend()

    def fake_export_graph_to_neo4j(graph, **kwargs):
        captured["graph"] = graph
        captured.update(kwargs)

    monkeypatch.setattr(rapidgraph_core, "export_graph_to_neo4j", fake_export_graph_to_neo4j)

    writer = extract_graph.Neo4jGraphWriter(
        uri="neo4j://example",
        user="neo4j",
        password="secret",
        database="graphrag",
        vector_index_name="custom_index",
        embedding_property="vector",
        chunk_embedding_model="fake-embedder",
        driver_factory="driver_factory",
        embedding_backend=embedding_backend,
    )
    writer.write(result, clean_document=True, embed_chunks=True, create_vector_index=True)

    assert captured["graph"] is result
    assert captured["uri"] == "neo4j://example"
    assert captured["user"] == "neo4j"
    assert captured["password"] == "secret"
    assert captured["database"] == "graphrag"
    assert captured["clean_document"] is True
    assert captured["embed_chunks"] is True
    assert captured["create_vector_index"] is True
    assert captured["vector_index_name"] == "custom_index"
    assert captured["embedding_property"] == "vector"
    assert captured["chunk_embedding_model"] == "fake-embedder"
    assert captured["driver_factory"] == "driver_factory"
    assert captured["embedding_backend"] is embedding_backend


def test_ask_neo4j_graph_builds_client_and_forwards_options(monkeypatch):
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, **kwargs):
            captured["retriever_kwargs"] = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

    class FakeClient:
        def __init__(self, *, retriever, llm):
            captured["retriever"] = retriever
            captured["llm"] = llm

        def ask(self, question, *, top_k, graph_depth, max_facts):
            captured["ask"] = {
                "question": question,
                "top_k": top_k,
                "graph_depth": graph_depth,
                "max_facts": max_facts,
            }
            return graphrag.GraphRAGAnswer(answer="answer", sources=[], facts=[], meta={})

    monkeypatch.setattr(graphrag, "Neo4jVectorRetriever", FakeRetriever)
    monkeypatch.setattr(graphrag, "OllamaLLM", FakeLLM)
    monkeypatch.setattr(graphrag, "GraphRAGClient", FakeClient)

    answer = graphrag.ask_neo4j_graph(
        "What uses attention?",
        neo4j_uri="neo4j://example",
        neo4j_user="neo4j",
        neo4j_password="secret",
        neo4j_database="graphrag",
        neo4j_vector_index_name="custom_index",
        neo4j_embedding_property="vector",
        chunk_embedding_model="fake-embedder",
        ollama_model="llama3.2",
        ollama_host="http://ollama.local",
        ollama_timeout=30.0,
        top_k=4,
        graph_depth=1,
        max_facts=8,
        driver_factory="driver_factory",
        embedding_backend="embedding_backend",
        http_client="http_client",
    )

    assert answer.answer == "answer"
    assert captured["retriever_kwargs"] == {
        "uri": "neo4j://example",
        "user": "neo4j",
        "password": "secret",
        "database": "graphrag",
        "vector_index_name": "custom_index",
        "embedding_property": "vector",
        "embedding_model": "fake-embedder",
        "driver_factory": "driver_factory",
        "embedding_backend": "embedding_backend",
    }
    assert captured["llm_kwargs"] == {
        "model": "llama3.2",
        "host": "http://ollama.local",
        "timeout": 30.0,
        "http_client": "http_client",
    }
    assert captured["ask"] == {
        "question": "What uses attention?",
        "top_k": 4,
        "graph_depth": 1,
        "max_facts": 8,
    }
