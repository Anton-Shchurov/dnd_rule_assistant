from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer
from rich import print
from rich.table import Table

from dnd_rag.core.pipelines import chunk_docs_pipeline, parse_docs_pipeline
from dnd_rag.core.config import DEFAULT_CONFIG_PATH


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


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

