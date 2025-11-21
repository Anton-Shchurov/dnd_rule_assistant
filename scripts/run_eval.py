from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from dnd_rag.core.config import load_ingest_config, DEFAULT_CONFIG_PATH
from dnd_rag.core.pipelines import AnswerResult, answer_query_pipeline
from dnd_rag.core.prompts import get_eval_prompt
from dnd_rag.providers.llm import LLMClient, ChatMessage

TOKEN_SPLIT_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


async def llm_eval_score(
    question: str,
    expected: str,
    predicted: str,
    llm_client: LLMClient,
    prompts_path: Path | None = None
) -> Dict[str, Any]:
    eval_prompt_tmpl = get_eval_prompt(prompts_path)
    user_content = eval_prompt_tmpl.format(
        question=question,
        expected_answer=expected,
        predicted_answer=predicted
    )
    
    messages = [ChatMessage(role="user", content=user_content)]
    response = await llm_client.generate(messages, temperature=0.0)
    
    content = response.content.strip()
    # Try to find JSON block
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
            
    return {"score": 0.0, "reasoning": f"Failed to parse JSON. Raw: {content[:100]}..."}


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if not samples:
        raise ValueError(f"Dataset {path} is empty.")
    return samples


def normalize_tokens(text: str) -> List[str]:
    tokens = [tok for tok in TOKEN_SPLIT_RE.split((text or "").lower()) if tok]
    return tokens


def squad_f1(predicted: str, reference: str) -> float:
    pred_tokens = normalize_tokens(predicted)
    ref_tokens = normalize_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    common = 0
    ref_counts: Dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    for tok in pred_tokens:
        if ref_counts.get(tok, 0) > 0:
            common += 1
            ref_counts[tok] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def chunk_recall(expected: Sequence[str], retrieved: Sequence[str]) -> float:
    expected_set = {cid for cid in expected if cid}
    if not expected_set:
        return 1.0
    retrieved_set = {cid for cid in retrieved if cid}
    if not retrieved_set:
        return 0.0
    overlap = len(expected_set & retrieved_set)
    return overlap / len(expected_set)


def gather_chunk_ids(result: AnswerResult) -> List[str]:
    return [chunk.chunk_id for chunk in result.chunks]


async def evaluate_sample(
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # Prepare LLM client for evaluation (judge)
    # We use the same model config or can override via args if needed.
    # For simplicity, we use the ingestion config model or default to gpt-4o/mini.
    cfg = load_ingest_config(args.config or DEFAULT_CONFIG_PATH)
    judge_llm = LLMClient(model=cfg.llm_model_name)

    result = await answer_query_pipeline(
        sample["question"],
        collection=args.collection,
        host=args.host,
        port=args.port,
        k=args.k,
        config_path=args.config,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        rerank=not args.no_rerank,
        include_diagnostics=True,
    )

    expected_chunks = [ref.get("chunk_id") for ref in sample.get("references", []) if ref.get("chunk_id")]
    predicted_chunks = gather_chunk_ids(result)

    recall = chunk_recall(expected_chunks, predicted_chunks)
    # f1 = squad_f1(result.answer, sample["answer"]) # Legacy metric
    
    # LLM Evaluation
    eval_res = await llm_eval_score(
        question=sample["question"],
        expected=sample["answer"],
        predicted=result.answer,
        llm_client=judge_llm,
        prompts_path=None # Use default resolution
    )
    llm_score = float(eval_res.get("score", 0.0))
    reasoning = eval_res.get("reasoning", "")

    diagnostics = result.diagnostics.to_dict() if result.diagnostics else None

    return {
        "sample_id": sample.get("id"),
        "question": sample["question"],
        "expected_answer": sample["answer"],
        "predicted_answer": result.answer,
        "expected_chunks": expected_chunks,
        "retrieved_chunks": predicted_chunks,
        "recall": recall,
        "f1": llm_score, # Replacing F1 with LLM Score in the main output field
        "reasoning": reasoning,
        "diagnostics": diagnostics,
    }


def write_log(records: Iterable[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation for D&D RAG assistant.")
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/dataset.jsonl"), help="JSONL с эталонными вопросами")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество примеров")
    parser.add_argument("--k", type=int, default=5, help="Количество чанков в контексте")
    parser.add_argument("--collection", default="dnd_rule_assistant", help="Коллекция Qdrant")
    parser.add_argument("--host", default="localhost", help="Хост Qdrant")
    parser.add_argument("--port", type=int, default=6333, help="Порт Qdrant")
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"), help="Путь к ingest-конфигу")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Модель эмбеддингов OpenAI")
    parser.add_argument("--temperature", type=float, default=None, help="Температура LLM")
    parser.add_argument("--no-rerank", action="store_true", help="Отключить переранжирование")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/eval"), help="Куда писать JSONL-отчёт")
    return parser


async def main_async(args: argparse.Namespace) -> None:
    samples = load_dataset(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        record = await evaluate_sample(sample, args)
        records.append(record)
        print(
            f"[{idx}/{len(samples)}] recall={record['recall']:.2f} "
            f"f1={record['f1']:.2f} | {sample.get('id') or 'no-id'}"
        )

    if not records:
        print("Нет результатов для отчёта.")
        return

    avg_recall = sum(r["recall"] for r in records) / len(records)
    avg_f1 = sum(r["f1"] for r in records) / len(records)

    log_path = write_log(records, args.log_dir)

    print("-" * 60)
    print(f"Samples: {len(records)}")
    print(f"Average Recall@{args.k}: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Log written to: {log_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

