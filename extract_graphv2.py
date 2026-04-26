from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_MODEL = "phi3:latest"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "relations": {"type": "array", "items": {"type": "string"}},
        "potential_schema": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "string"},
            },
        },
    },
    "required": ["entities", "relations", "potential_schema"],
}


def normalize_entity_label(label: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", label)
    return "".join(word[:1].upper() + word[1:].lower() for word in words)


def normalize_relation_label(label: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", label)
    return "_".join(word.upper() for word in words)


def parse_model_response(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    for key in ("entities", "relations", "potential_schema"):
        if key not in payload:
            raise ValueError(f"Model response must contain '{key}'.")
    if not isinstance(payload["potential_schema"], list) or not all(
        isinstance(item, list)
        and len(item) == 3
        and all(isinstance(value, str) for value in item)
        for item in payload["potential_schema"]
    ):
        raise ValueError("potential_schema must contain 3-item string lists.")
    return payload


def normalize_payload(text: str, payload: dict[str, Any]) -> dict[str, Any]:
    entities = {normalize_entity_label(item) for item in payload.get("entities", []) if isinstance(item, str)}
    relations = {normalize_relation_label(item) for item in payload.get("relations", []) if isinstance(item, str)}
    schema = set()

    for source, relation, target in payload.get("potential_schema", []):
        source_type = normalize_entity_label(source)
        relation_name = normalize_relation_label(relation)
        target_type = normalize_entity_label(target)
        entities.update([source_type, target_type])
        relations.add(relation_name)
        schema.add((source_type, relation_name, target_type))

    return {
        "TEXT": text,
        "ENTITIES": sorted(entities),
        "RELATIONS": sorted(relations),
        "POTENTIAL_SCHEMA": sorted(schema),
    }


def extract_schema(text: str, *, model: str = DEFAULT_MODEL, client=None) -> dict[str, Any]:
    if client is None:
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError("extract_graphv2 requires `ollama` when no client is provided.") from exc
        client = ollama

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract entity labels, relation labels, and potential schema triples from this text. "
                    "Return only JSON matching the provided schema.\n\n"
                    f"Text:\n{text}"
                ),
            }
        ],
        format=RESPONSE_SCHEMA,
        stream=False,
        options={"temperature": 0},
    )
    content = response.get("message", {}).get("content", "")
    return normalize_payload(text, parse_model_response(content))


def format_python_output(payload: dict[str, Any], *, pretty: bool = False) -> str:
    text = payload["TEXT"]
    entities = payload["ENTITIES"]
    relations = payload["RELATIONS"]
    schema = payload["POTENTIAL_SCHEMA"]
    if not pretty:
        return (
            f'TEXT = """\n{text}\n""".strip()\n\n'
            f"ENTITIES = {format_string_list(entities)}\n\n"
            f"RELATIONS = {format_string_list(relations)}\n\n"
            f"POTENTIAL_SCHEMA = {format_schema_list(schema)}\n"
        )

    lines = [f'TEXT = """\n{text}\n""".strip()', "", "ENTITIES = ["]
    lines.extend(f'    "{entity}",' for entity in entities)
    lines.extend(["]", "", "RELATIONS = ["])
    lines.extend(f'    "{relation}",' for relation in relations)
    lines.extend(["]", "", "POTENTIAL_SCHEMA = ["])
    lines.extend(f"    {format_schema_triple(triple)}," for triple in schema)
    lines.append("]")
    return "\n".join(lines) + "\n"


def format_string_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def format_schema_triple(triple: tuple[str, str, str]) -> str:
    return "(" + ", ".join(json.dumps(value) for value in triple) + ")"


def format_schema_list(values: Sequence[tuple[str, str, str]]) -> str:
    return "[" + ", ".join(format_schema_triple(value) for value in values) + "]"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a schema payload with an Ollama chat model.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", help="Raw text to process.")
    source_group.add_argument("--input", help="UTF-8 text file to process.")
    parser.add_argument("--output", help="Optional output path for Python assignment output.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--pretty", action="store_true", help="Render multiline Python assignments.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    text = args.text if args.text is not None else Path(args.input).read_text(encoding="utf-8")
    payload = extract_schema(text, model=args.model)
    output = format_python_output(payload, pretty=args.pretty)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
