from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extract_graphv2
from test import TEXT


class StubClient:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.content}}


def test_extract_schema_normalizes_labels_and_sorts_output():
    client = StubClient(
        json.dumps(
            {
                "entities": ["person", "organization", "city"],
                "relations": ["headquartered in", "leads"],
                "potential_schema": [
                    ["person", "leads", "organization"],
                    ["organization", "headquartered in", "city"],
                    ["organization", "headquartered in", "city"],
                ],
            }
        )
    )

    result = extract_graphv2.extract_schema(TEXT, client=client)

    assert result == {
        "TEXT": TEXT,
        "ENTITIES": ["City", "Organization", "Person"],
        "RELATIONS": ["HEADQUARTERED_IN", "LEADS"],
        "POTENTIAL_SCHEMA": [
            ("Organization", "HEADQUARTERED_IN", "City"),
            ("Person", "LEADS", "Organization"),
        ],
    }


def test_schema_triples_add_missing_entity_and_relation_labels():
    payload = {
        "entities": ["Person"],
        "relations": ["leads"],
        "potential_schema": [
            ["Research Topic", "describes", "Knowledge Graph"],
        ],
    }

    result = extract_graphv2.normalize_payload("hello", payload)

    assert result == {
        "TEXT": "hello",
        "ENTITIES": ["KnowledgeGraph", "Person", "ResearchTopic"],
        "RELATIONS": ["DESCRIBES", "LEADS"],
        "POTENTIAL_SCHEMA": [("ResearchTopic", "DESCRIBES", "KnowledgeGraph")],
    }


def test_invalid_non_json_response_raises_clear_error():
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_graphv2.parse_model_response("not json")


def test_missing_required_key_raises_clear_error():
    with pytest.raises(ValueError, match="must contain 'relations'"):
        extract_graphv2.parse_model_response(json.dumps({"entities": [], "potential_schema": []}))


def test_invalid_schema_shape_raises_clear_error():
    with pytest.raises(ValueError, match="3-item string lists"):
        extract_graphv2.parse_model_response(
            json.dumps(
                {
                    "entities": ["Person"],
                    "relations": ["LEADS"],
                    "potential_schema": [["Person", "LEADS"]],
                }
            )
        )


def test_formatter_matches_test_py_shape_in_pretty_mode():
    payload = {
        "TEXT": "Alpha leads Beta.",
        "ENTITIES": ["Organization", "Person"],
        "RELATIONS": ["LEADS"],
        "POTENTIAL_SCHEMA": [("Person", "LEADS", "Organization")],
    }

    formatted = extract_graphv2.format_python_output(payload, pretty=True)

    assert 'TEXT = """\nAlpha leads Beta.\n""".strip()' in formatted
    assert "ENTITIES = [\n" in formatted
    assert '    "Organization",' in formatted
    assert '    ("Person", "LEADS", "Organization"),' in formatted
    parsed = ast.parse(formatted)
    assert parsed is not None


def test_cli_stdout_and_output_file_render_python_assignments(monkeypatch, capsys, tmp_path: Path):
    fake_result = {
        "TEXT": "hello world",
        "ENTITIES": ["Organization", "Person"],
        "RELATIONS": ["LEADS"],
        "POTENTIAL_SCHEMA": [("Person", "LEADS", "Organization")],
    }

    def fake_extract_schema(text: str, *, model: str, client=None):
        del client
        assert text == "hello world"
        assert model == "custom-model"
        return fake_result

    monkeypatch.setattr(extract_graphv2, "extract_schema", fake_extract_schema)

    exit_code = extract_graphv2.main(["--text", "hello world", "--model", "custom-model"])
    assert exit_code == 0
    stdout_text = capsys.readouterr().out
    assert 'TEXT = """\nhello world\n""".strip()' in stdout_text
    assert 'ENTITIES = ["Organization", "Person"]' in stdout_text
    assert 'POTENTIAL_SCHEMA = [("Person", "LEADS", "Organization")]' in stdout_text

    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "schema.py"
    input_path.write_text("hello world", encoding="utf-8")

    exit_code = extract_graphv2.main(
        ["--input", str(input_path), "--output", str(output_path), "--pretty", "--model", "custom-model"]
    )
    assert exit_code == 0
    written_text = output_path.read_text(encoding="utf-8")
    assert "RELATIONS = [\n" in written_text
    assert '    "LEADS",' in written_text
    assert '    ("Person", "LEADS", "Organization"),' in written_text


def test_extract_schema_passes_expected_ollama_contract():
    client = StubClient(
        json.dumps(
            {
                "entities": ["Person"],
                "relations": ["LEADS"],
                "potential_schema": [["Person", "LEADS", "Organization"]],
            }
        )
    )

    result = extract_graphv2.extract_schema("Sam Altman leads OpenAI.", model="phi3:test", client=client)

    assert result["POTENTIAL_SCHEMA"] == [("Person", "LEADS", "Organization")]
    assert client.calls[0]["model"] == "phi3:test"
    assert client.calls[0]["format"] == extract_graphv2.RESPONSE_SCHEMA
    assert client.calls[0]["stream"] is False
    assert client.calls[0]["options"] == {"temperature": 0}


def test_formatter_output_is_valid_python():
    payload = {
        "TEXT": TEXT,
        "ENTITIES": ["City", "Organization", "Person"],
        "RELATIONS": ["HEADQUARTERED_IN", "LEADS"],
        "POTENTIAL_SCHEMA": [
            ("Organization", "HEADQUARTERED_IN", "City"),
            ("Person", "LEADS", "Organization"),
        ],
    }

    formatted = extract_graphv2.format_python_output(payload, pretty=True)
    namespace: dict[str, object] = {}
    exec(formatted, namespace)

    assert namespace["TEXT"] == TEXT
    assert namespace["ENTITIES"] == ["City", "Organization", "Person"]
    assert namespace["RELATIONS"] == ["HEADQUARTERED_IN", "LEADS"]
    assert namespace["POTENTIAL_SCHEMA"] == [
        ("Organization", "HEADQUARTERED_IN", "City"),
        ("Person", "LEADS", "Organization"),
    ]
