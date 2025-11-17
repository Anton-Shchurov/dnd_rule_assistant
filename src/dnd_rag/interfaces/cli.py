from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer
from rich import print
from rich.table import Table

from dnd_rag.core.pipelines import (
    chunk_docs_pipeline,
    parse_docs_pipeline,
    sections_from_md_pipeline,
    chunks_from_sections_pipeline,
)
from dnd_rag.core.config import DEFAULT_CONFIG_PATH
from dnd_rag.providers.vectorstore import get_client, ensure_collection, upsert_vectors
from dnd_rag.providers.embeddings import embed_texts


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("docs-parse")
def docs_parse(
    in_dir: Path = typer.Option(Path("data/raw"), "--in", help="Папка с PDF"),
    out_dir: Path = typer.Option(Path("data/processed/md"), "--out", help="Папка для Markdown"),
    ocr: bool = typer.Option(False, help="Включить OCR (обычно не требуется)"),
):
    produced = parse_docs_pipeline(in_dir, out_dir, ocr=ocr)
    for p in produced:
        print(f"[green]MD сохранён[/green]: {p}")


@app.command("docs-chunk")
def docs_chunk(
    in_dir: Path = typer.Option(Path("data/processed/md"), "--in", help="Папка с Markdown"),
    out_dir: Path = typer.Option(Path("data/processed/chunks"), "--out", help="Папка для JSONL чанков"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Путь к YAML-конфигу"),
):
    produced = chunk_docs_pipeline(in_dir, out_dir, config_path=config)
    for p in produced:
        print(f"[green]Chunks сохранены[/green]: {p}")


@app.command("sections")
def sections_cmd(
    in_dir: Path = typer.Option(Path("data/processed/md_clean"), "--in", help="Папка с Markdown"),
    out_dir: Path = typer.Option(Path("data/processed/sections"), "--out", help="Папка для JSONL секций"),
):
    produced = sections_from_md_pipeline(in_dir, out_dir)
    for p in produced:
        print(f"[green]Sections сохранены[/green]: {p}")


@app.command("chunks")
def chunks_cmd(
    in_dir: Path = typer.Option(Path("data/processed/sections"), "--in", help="Папка с JSONL секциями"),
    out_dir: Path = typer.Option(Path("data/processed/chunks"), "--out", help="Папка для JSONL чанков"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Путь к YAML-конфигу"),
):
    produced = chunks_from_sections_pipeline(in_dir, out_dir, config_path=config)
    for p in produced:
        print(f"[green]Chunks сохранены[/green]: {p}")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@app.command("docs-sample")
def docs_sample(
    in_path: Path = typer.Option(Path("data/processed/chunks"), "--in", help="Папка или файл JSONL"),
    q: str = typer.Option(..., "--q", help="Запрос"),
    k: int = typer.Option(5, "-k", help="Количество результатов"),
):
    files: List[Path] = []
    if in_path.is_dir():
        files = sorted(in_path.glob("*.jsonl"))
    else:
        files = [in_path]

    rows: List[dict] = []
    for fp in files:
        rows.extend(_read_jsonl(fp))

    q_terms = [t for t in q.lower().split() if t]
    def score(text: str) -> int:
        low = text.lower()
        return sum(low.count(t) for t in q_terms)

    rows.sort(key=lambda r: score(r.get("text", "")), reverse=True)
    top = rows[:k]

    table = Table(title=f"Top-{k} по запросу: {q}")
    table.add_column("book")
    table.add_column("chapter")
    table.add_column("section")
    table.add_column("pages")
    table.add_column("preview")

    for r in top:
        pages = (
            f"{r.get('page_start')}-{r.get('page_end')}"
            if r.get("page_start") is not None and r.get("page_end") is not None
            else ""
        )
        preview = (r.get("text", "").replace("\n", " ")[:140] + "…") if r.get("text") else ""
        table.add_row(
            str(r.get("book", "")),
            str(r.get("chapter", "")),
            str(r.get("section", "")),
            pages,
            preview,
        )

    print(table)


@app.command("init-qdrant")
def init_qdrant(
    collection: str = typer.Option("dnd_rule_assistant", "--collection", help="Имя коллекции"),
    host: str = typer.Option("localhost", "--host", envvar="QDRANT_HOST"),
    port: int = typer.Option(6333, "--port", envvar="QDRANT_PORT"),
    dim: int = typer.Option(1536, "--dim", help="Размерность эмбеддинга"),
):
    client = get_client(host=host, port=port)
    ensure_collection(client, collection, vector_size=dim)
    print(f"[green]Qdrant коллекция готова[/green]: {collection} (dim={dim})")


@app.command("index")
def index_cmd(
    chunks: List[Path] = typer.Argument(..., help="Список путей к JSONL с чанками"),
    collection: str = typer.Option("dnd_rule_assistant", "--collection", help="Имя коллекции"),
    host: str = typer.Option("localhost", "--host", envvar="QDRANT_HOST"),
    port: int = typer.Option(6333, "--port", envvar="QDRANT_PORT"),
    model: str = typer.Option("text-embedding-3-small", "--model", help="Модель эмбеддингов OpenAI"),
):
    client = get_client(host=host, port=port)
    # размерность 1536 для text-embedding-3-small
    ensure_collection(client, collection, vector_size=1536)

    # Считываем чанки
    rows: List[dict] = []
    for fp in chunks:
        rows.extend(_read_jsonl(fp))

    texts = [r.get("text", "") for r in rows]
    ids = [r.get("chunk_id", "") for r in rows]
    payloads: List[dict] = []
    for r in rows:
        payloads.append(
            {
                "chunk_id": r.get("chunk_id", ""),
                "book_title": r.get("book_title", ""),
                "chapter_title": r.get("chapter_title", ""),
                "section_path": r.get("section_path", []),
                "chunk_index": r.get("chunk_index", 0),
            }
        )

    vectors = embed_texts(texts, model=model)
    upsert_vectors(client, collection, ids=ids, vectors=vectors, payloads=payloads)

    print(f"[green]Загружено в Qdrant[/green]: {len(rows)} точек → {collection}")


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

