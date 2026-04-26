from __future__ import annotations

from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field

from .core import (
    SentenceTransformerEmbeddingBackend,
    build_neo4j_vector_index_query,
    quote_neo4j_identifier,
)


DEFAULT_VECTOR_INDEX_NAME = "rapidgraph_chunk_embedding"
DEFAULT_EMBEDDING_PROPERTY = "embedding"
DEFAULT_CHUNK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class RetrievedChunk(BaseModel):
    id: str
    document_id: str
    text: str
    score: float
    source: str = ""
    title: str = ""
    embedding_model: str = ""


class RetrievedFact(BaseModel):
    source_id: str
    source_text: str
    relation: str
    target_id: str
    target_text: str
    evidence: str
    chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class GraphRAGAnswer(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    facts: list[RetrievedFact]
    meta: dict[str, Any] = Field(default_factory=dict)


class OllamaLLM:
    def __init__(
        self,
        *,
        model: str,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        http_client: Any | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.http_client = http_client

    def generate(self, prompt: str) -> str:
        client = self.http_client
        if client is None:
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError(
                    "Ollama GraphRAG requires the `requests` package. Install it with `pip install rapidGraph[graphrag]`."
                ) from exc
            client = requests

        try:
            response = client.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        text = payload.get("response")
        if not isinstance(text, str):
            raise RuntimeError("Ollama response did not contain a string `response` field.")
        return text.strip()


class Neo4jVectorRetriever:
    def __init__(
        self,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        vector_index_name: str = DEFAULT_VECTOR_INDEX_NAME,
        embedding_property: str = DEFAULT_EMBEDDING_PROPERTY,
        embedding_model: str = DEFAULT_CHUNK_EMBEDDING_MODEL,
        driver_factory: Any | None = None,
        embedding_backend: Any | None = None,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.vector_index_name = vector_index_name
        self.embedding_property = embedding_property
        self.embedding_model = embedding_model
        self.driver_factory = driver_factory
        self.embedding_backend = embedding_backend or SentenceTransformerEmbeddingBackend(embedding_model)

    def _driver(self):
        if self.driver_factory is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "Neo4j GraphRAG requires the `neo4j` package. Install it with `pip install rapidGraph[graphrag]`."
                ) from exc
            self.driver_factory = GraphDatabase.driver
        return self.driver_factory(self.uri, auth=(self.user, self.password))

    def ensure_vector_index(self, *, dimension: int, similarity_function: str = "cosine") -> None:
        driver = self._driver()
        try:
            with driver.session(database=self.database) as session:
                session.run(
                    build_neo4j_vector_index_query(
                        index_name=self.vector_index_name,
                        embedding_property=self.embedding_property,
                    ),
                    dimension=dimension,
                    similarity_function=similarity_function,
                )
        finally:
            close = getattr(driver, "close", None)
            if callable(close):
                close()

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        graph_depth: int = 1,
        max_facts: int = 20,
    ) -> tuple[list[RetrievedChunk], list[RetrievedFact], dict[str, Any]]:
        vectors = self.embedding_backend.embed_many([question])
        if not vectors:
            return [], [], {"warnings": ["Question embedding was empty."]}
        question_embedding = [float(value) for value in vectors[0]]
        driver = self._driver()
        try:
            with driver.session(database=self.database) as session:
                chunk_rows = list(
                    session.run(
                        build_vector_retrieval_query(),
                        index_name=self.vector_index_name,
                        top_k=max(1, top_k),
                        embedding=question_embedding,
                    )
                )
                chunks = [record_to_chunk(row) for row in chunk_rows]
                if not chunks:
                    return [], [], {"warnings": []}

                chunk_ids = [chunk.id for chunk in chunks]
                fact_rows = list(
                    session.run(
                        build_fact_expansion_query(graph_depth=graph_depth, max_facts=max_facts),
                        chunk_ids=chunk_ids,
                        max_facts=max(0, max_facts),
                    )
                )
                facts = [record_to_fact(row) for row in fact_rows]
        except Exception as exc:
            message = str(exc)
            if "index" in message.casefold() or "vector" in message.casefold():
                raise RuntimeError(
                    "Neo4j vector retrieval failed. Re-export with "
                    "`--neo4j-embed-chunks --neo4j-create-vector-index` and confirm the index name."
                ) from exc
            raise
        finally:
            close = getattr(driver, "close", None)
            if callable(close):
                close()

        warnings = []
        indexed_models = {chunk.embedding_model for chunk in chunks if chunk.embedding_model}
        if indexed_models and self.embedding_model not in indexed_models:
            warnings.append(
                f"Question embedding model `{self.embedding_model}` differs from chunk embedding model(s): "
                f"{', '.join(sorted(indexed_models))}."
            )
        return chunks, facts, {"warnings": warnings}


class GraphRAGClient:
    def __init__(self, *, retriever: Neo4jVectorRetriever, llm: LLMProvider):
        self.retriever = retriever
        self.llm = llm

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        graph_depth: int = 1,
        max_facts: int = 20,
    ) -> GraphRAGAnswer:
        chunks, facts, retrieval_meta = self.retriever.retrieve(
            question,
            top_k=top_k,
            graph_depth=graph_depth,
            max_facts=max_facts,
        )
        if not chunks:
            return GraphRAGAnswer(
                answer="No relevant graph context was found for the question.",
                sources=[],
                facts=[],
                meta={
                    "top_k": top_k,
                    "graph_depth": graph_depth,
                    "max_facts": max_facts,
                    "warnings": retrieval_meta.get("warnings", []),
                },
            )
        prompt = build_graphrag_prompt(question, chunks, facts)
        answer = self.llm.generate(prompt)
        return GraphRAGAnswer(
            answer=answer,
            sources=chunks,
            facts=facts,
            meta={
                "top_k": top_k,
                "graph_depth": graph_depth,
                "max_facts": max_facts,
                "chunk_count": len(chunks),
                "fact_count": len(facts),
                "warnings": retrieval_meta.get("warnings", []),
            },
        )


def ask_neo4j_graph(
    question: str,
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    neo4j_vector_index_name: str = DEFAULT_VECTOR_INDEX_NAME,
    neo4j_embedding_property: str = DEFAULT_EMBEDDING_PROPERTY,
    chunk_embedding_model: str = DEFAULT_CHUNK_EMBEDDING_MODEL,
    ollama_model: str,
    ollama_host: str = "http://127.0.0.1:11434",
    ollama_timeout: float = 120.0,
    top_k: int = 5,
    graph_depth: int = 1,
    max_facts: int = 20,
    driver_factory: Any | None = None,
    embedding_backend: Any | None = None,
    http_client: Any | None = None,
) -> GraphRAGAnswer:
    retriever = Neo4jVectorRetriever(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        vector_index_name=neo4j_vector_index_name,
        embedding_property=neo4j_embedding_property,
        embedding_model=chunk_embedding_model,
        driver_factory=driver_factory,
        embedding_backend=embedding_backend,
    )
    llm = OllamaLLM(
        model=ollama_model,
        host=ollama_host,
        timeout=ollama_timeout,
        http_client=http_client,
    )
    return GraphRAGClient(retriever=retriever, llm=llm).ask(
        question,
        top_k=top_k,
        graph_depth=graph_depth,
        max_facts=max_facts,
    )


def build_vector_retrieval_query() -> str:
    return """
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
YIELD node, score
OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.id AS id,
       node.document_id AS document_id,
       node.text AS text,
       node.embedding_model AS embedding_model,
       score AS score,
       d.source AS source,
       d.title AS title
ORDER BY score DESC
"""


def build_fact_expansion_query(*, graph_depth: int, max_facts: int) -> str:
    del max_facts
    if graph_depth <= 0:
        return """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.id IN $chunk_ids
RETURN e.id AS source_id,
       e.text AS source_text,
       "MENTIONED_IN" AS relation,
       e.id AS target_id,
       e.text AS target_text,
       c.text AS evidence,
       [c.id] AS chunk_ids,
       [c.document_id] AS document_ids,
       1.0 AS confidence
LIMIT $max_facts
"""
    return """
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE c.id IN $chunk_ids
MATCH (e)-[r:RELATES_TO]-(other:Entity)
RETURN startNode(r).id AS source_id,
       startNode(r).text AS source_text,
       r.relation AS relation,
       endNode(r).id AS target_id,
       endNode(r).text AS target_text,
       r.evidence AS evidence,
       r.chunk_ids AS chunk_ids,
       r.document_ids AS document_ids,
       r.confidence AS confidence
ORDER BY confidence DESC
LIMIT $max_facts
"""


def build_graphrag_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    facts: Sequence[RetrievedFact],
) -> str:
    chunk_lines = [
        f"[{chunk.id}] score={chunk.score:.4f} source={chunk.source or chunk.document_id}\n{chunk.text}"
        for chunk in chunks
    ]
    fact_lines = [
        (
            f"[{', '.join(fact.chunk_ids) or 'no_chunk'}] "
            f"{fact.source_text} -[{fact.relation}]-> {fact.target_text}. "
            f"Evidence: {fact.evidence}"
        )
        for fact in facts
    ]
    return (
        "Answer the question using only the graph context below. "
        "If the context is insufficient, say so. Include source chunk IDs when possible.\n\n"
        f"Question:\n{question}\n\n"
        "Retrieved chunks:\n"
        + ("\n\n".join(chunk_lines) if chunk_lines else "None")
        + "\n\nGraph facts:\n"
        + ("\n".join(fact_lines) if fact_lines else "None")
        + "\n\nAnswer:"
    )


def record_to_chunk(row: Any) -> RetrievedChunk:
    return RetrievedChunk(
        id=str(get_record_value(row, "id", "")),
        document_id=str(get_record_value(row, "document_id", "")),
        text=str(get_record_value(row, "text", "")),
        score=float(get_record_value(row, "score", 0.0) or 0.0),
        source=str(get_record_value(row, "source", "") or ""),
        title=str(get_record_value(row, "title", "") or ""),
        embedding_model=str(get_record_value(row, "embedding_model", "") or ""),
    )


def record_to_fact(row: Any) -> RetrievedFact:
    return RetrievedFact(
        source_id=str(get_record_value(row, "source_id", "")),
        source_text=str(get_record_value(row, "source_text", "")),
        relation=str(get_record_value(row, "relation", "")),
        target_id=str(get_record_value(row, "target_id", "")),
        target_text=str(get_record_value(row, "target_text", "")),
        evidence=str(get_record_value(row, "evidence", "") or ""),
        chunk_ids=list(get_record_value(row, "chunk_ids", []) or []),
        document_ids=list(get_record_value(row, "document_ids", []) or []),
        confidence=float(get_record_value(row, "confidence", 0.0) or 0.0),
    )


def get_record_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(key, default)
    try:
        return row[key]
    except Exception:
        return default


def validate_graphrag_identifier(identifier: str) -> str:
    return quote_neo4j_identifier(identifier)
