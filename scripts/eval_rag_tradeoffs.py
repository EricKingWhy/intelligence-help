"""Compare retrieval strategies under a shared answer and refusal policy."""
import argparse
import asyncio
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles
from time import perf_counter

from langchain_core.messages import AIMessage

from app.config import settings
from app.core import coref, llm, query_understanding, retrieval, selfcheck
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from scripts.eval_ch04 import (
    COVERAGE_PROMPT, STRATEGIES, _Cov, _Faith, _format_evidence, _load,
    _ranks_per_group, _recall_at, _rr,
)

FOLLOWUPS = [
    dict(id="M1", bucket="multiturn", history="用户：我想知道退货政策。\n客服：支持七天无理由。",
         query="那拆封了还能退吗？", expect_section=["退货政策"],
         expect_points=["不影响二次销售"], should_refuse=False),
    dict(id="M2", bucket="multiturn", history="用户：我买的商品要保修。\n客服：需要核对保修范围。",
         query="那人为损坏也算吗？", expect_section=["保修说明"],
         expect_points=["人为损坏不在保修"], should_refuse=False),
]


async def prepare_query(sample):
    start = perf_counter()
    query = sample["query"]
    if sample.get("history"):
        query = await coref.resolve(query, sample["history"])
    rewrite = await query_understanding.understand(query)
    return rewrite["standard"], " ".join([rewrite["standard"], *rewrite["expanded"]]), (
        perf_counter() - start
    ) * 1000


async def evaluate_one(strategy, sample, prepared, model, coverage, faith):
    query, bm25_text, preprocessing_ms = prepared
    start = perf_counter()
    clauses = len(retrieval.split_clauses(query)) if settings.subquery_split else 1
    hits = await retrieval.search_knowledge(query, strategy=strategy, bm25_text=bm25_text, top_k=10)
    ranks = _ranks_per_group(hits, sample) if not sample["should_refuse"] else []
    # A shared semantic gate compares evidence from strategies with different score scales.
    check = await selfcheck.check_sufficient(
        query, [f"{hit['question']} {hit['answer']}" for hit in hits],
    ) if hits else {"useful": False}
    if service_failure := check.get("service_failure"):
        raise RuntimeError(f"证据自评失败 ({service_failure['stage']}:{service_failure['kind']})")
    refused = not check["useful"]
    result = {
        "recall_at_5": _recall_at(ranks, 5) if ranks else None,
        "mrr": _rr(ranks) if ranks else None,
        "refused": refused,
        "should_refuse": sample["should_refuse"],
        "answer_calls": 0,
        "answer_tokens": 0,
        "preprocessing_ms": preprocessing_ms,
        "estimated_embedding_calls": clauses if strategy != "bm25" else 0,
        "estimated_rerank_calls_upper_bound": clauses if strategy == "hybrid_rerank" else 0,
    }
    if refused:
        result["elapsed_ms"] = result["first_output_ms"] = preprocessing_ms + (perf_counter() - start) * 1000
        result["coverage"] = 0 if not sample["should_refuse"] else None
        return result

    evidence = _format_evidence(hits)
    aggregate = None
    result["answer_calls"] = 1
    async for chunk in (RAG_ANSWER_PROMPT | model).astream({"query": query, "evidence": evidence}):
        if isinstance(chunk.content, str) and chunk.content and "first_output_ms" not in result:
            result["first_output_ms"] = preprocessing_ms + (perf_counter() - start) * 1000
        aggregate = chunk if aggregate is None else aggregate + chunk
    result["elapsed_ms"] = preprocessing_ms + (perf_counter() - start) * 1000
    if not isinstance(aggregate, AIMessage) or "first_output_ms" not in result:
        raise RuntimeError("生成没有返回文本")
    result["answer_tokens"] = (aggregate.usage_metadata or {}).get("total_tokens")
    answer = aggregate.content
    try:
        if sample["expect_points"]:
            points = sample["expect_points"]
            scored = await (COVERAGE_PROMPT | coverage).ainvoke(
                {"query": query, "n": len(points), "points": "\n".join(points), "answer": answer},
            )
            result["coverage"] = min(scored.covered_count, len(points)) / len(points)
        scored = await (FAITHFULNESS_PROMPT | faith).ainvoke({"evidence": evidence, "answer": answer})
    except Exception as exc:
        result["scoring_error"] = type(exc).__name__
        return result
    result["faithfulness"] = bool(scored.faithful)
    if not scored.faithful or sample["should_refuse"]:
        result["error_example"] = {"answer": answer, "reason": scored.reason}
    return result


def summarize(rows):
    ok = [row for row in rows if "error" not in row]
    counts = {"n": len(rows), "failed": len(rows) - len(ok),
              "scoring_failed": sum("scoring_error" in row for row in ok),
              "timeouts": sum(row.get("error") == "TimeoutError" for row in rows)}
    for key in ("recall_at_5", "mrr", "coverage", "faithfulness", "answer_calls", "answer_tokens",
                "preprocessing_ms", "estimated_embedding_calls", "estimated_rerank_calls_upper_bound"):
        values = [row[key] for row in ok if row.get(key) is not None]
        counts[key] = mean(values) if values else None
        counts[f"{key}_n"] = len(values)
    for key in ("elapsed_ms", "first_output_ms"):
        values = [row[key] for row in ok if key in row]
        counts[key] = ({f"p{p}": (quantiles(values, n=100, method="inclusive")[p - 1]
                                   if len(values) > 1 else values[0])
                        for p in (50, 95)} if values else None)
    absent = [row for row in ok if row["should_refuse"]]
    counts["refusal_rate"] = mean(row["refused"] for row in absent) if absent else None
    counts["refusal_n"] = len(absent)
    answerable = [row for row in ok if not row["should_refuse"]]
    counts["false_refusal_rate"] = mean(row["refused"] for row in answerable) if answerable else None
    return counts


async def run(args):
    samples = [*_load(), *FOLLOWUPS]
    if args.limit_per_bucket:
        buckets = defaultdict(list)
        for sample in samples:
            buckets[sample["bucket"]].append(sample)
        samples = [sample for bucket in buckets.values() for sample in bucket[:args.limit_per_bucket]]
    model = llm.get_chat_model(streaming=True, temperature=0)
    coverage = llm.structured(_Cov)
    faith = llm.structured(_Faith)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def compare(index, sample):
        async with semaphore:
            strategies = STRATEGIES[index % len(STRATEGIES):] + STRATEGIES[:index % len(STRATEGIES)]
            try:
                prepared = await asyncio.wait_for(prepare_query(sample), timeout=args.timeout)
            except Exception as exc:
                return [{"strategy": strategy, "id": sample["id"], "bucket": sample["bucket"],
                         "error": type(exc).__name__} for strategy in strategies]
            outcomes = []
            for strategy in strategies:
                try:
                    result = await asyncio.wait_for(
                        evaluate_one(strategy, sample, prepared, model, coverage, faith),
                        timeout=args.timeout,
                    )
                except Exception as exc:
                    result = {"error": type(exc).__name__}
                outcomes.append({"strategy": strategy, "id": sample["id"], "bucket": sample["bucket"],
                                 **result})
            return outcomes

    rows = [row for group in await asyncio.gather(*(compare(i, sample)
                                                  for i, sample in enumerate(samples)))
            for row in group]
    totals, by_bucket = {}, {}
    for strategy in STRATEGIES:
        items = [row for row in rows if row["strategy"] == strategy]
        totals[strategy] = summarize(items)
        by_bucket[strategy] = {bucket: summarize([row for row in items if row["bucket"] == bucket])
                               for bucket in sorted({row["bucket"] for row in items})}
    incomplete = any("error" in row or "scoring_error" in row for row in rows)
    recommended = None
    if not incomplete and args.limit_per_bucket is None:
        # Business quality first, with time as a tiebreaker. Report per-bucket tradeoffs alongside it.
        recommended = max(totals, key=lambda key: (
            totals[key]["refusal_rate"] or 0, totals[key]["coverage"] or 0,
            totals[key]["faithfulness"] or 0, -totals[key]["elapsed_ms"]["p95"],
        ))
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "samples_sha256": hashlib.sha256(json.dumps(samples, ensure_ascii=False).encode()).hexdigest(),
        "kb_version": args.kb_version, "chat_model": settings.chat_model,
        "embed_model": settings.embed_model, "rerank_model": settings.rerank_model,
        "recall_top_k": settings.recall_top_k, "temperature": 0, "answer_top_k": 10,
        "concurrency": args.concurrency, "timeout_seconds": args.timeout,
        "cost_scope": "Generation usage is measured. Retrieval calls are estimates per clause, excluding retries.",
        "complete": not incomplete, "recommended_default": recommended,
        "totals": totals, "buckets": by_bucket, "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return int(incomplete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-version", required=True, help="Identifier of the indexed knowledge snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--limit-per-bucket", type=int)
    args = parser.parse_args()
    if (args.concurrency < 1 or args.timeout <= 0
            or (args.limit_per_bucket is not None and args.limit_per_bucket < 1)):
        parser.error("concurrency, timeout and sample limit must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
