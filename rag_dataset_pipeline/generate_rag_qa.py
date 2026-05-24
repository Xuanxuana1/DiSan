#!/usr/bin/env python3
"""Generate grounded RAG QA pairs from anchor-annotated chunks."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from json_utils import append_jsonl, load_jsonl, parse_json_object
from llm_client import LLMConfig, chat_completion

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "rag_prompt.md"

SUSPICIOUS_TERMS = [
    "lack of diversity",
    "diversity in candidate pool",
    "diversity in the candidate pool",
    "candidate pool",
    "training needs analysis",
    "training needs",
]


def build_user_prompt(document_type: str, chunk_text: str, hooks_json: str) -> str:
    return f"Document Type: {document_type}\nChunk Text (Anonymized): {chunk_text}\nHOOKS (JSON): {hooks_json}"


def validate_qa_pair(qa_data: dict[str, Any], chunk_text: str) -> tuple[bool, str]:
    chunk_lower = chunk_text.lower()
    query = qa_data.get("query", "").lower()

    evidence_texts = []
    for evidence in qa_data.get("grounding", {}).get("evidence", []):
        evidence_text = evidence.get("text", "")
        evidence_texts.append(evidence_text.lower())
        evidence_words = evidence_text.lower().split()
        if len(evidence_words) >= 3:
            matching_words = sum(1 for word in evidence_words if word in chunk_lower)
            if matching_words / len(evidence_words) < 0.5:
                return False, f"Evidence text not found in chunk: {evidence_text[:100]}"

    answer = qa_data.get("answer_gt", {}).get("final_answer", "").lower()
    for term in SUSPICIOUS_TERMS:
        if term in query and term not in chunk_lower:
            return False, f"Query contains '{term}' but the chunk does not"
        if term in answer and term not in chunk_lower:
            return False, f"Answer contains '{term}' but the chunk does not"

    combined_evidence = " ".join(evidence_texts)
    for term in re.findall(r"['\"]([^'\"]+)['\"]", query):
        if len(term) > 3 and term.lower() not in combined_evidence:
            return False, f"Query asks about '{term}' but evidence does not"

    return True, "Valid"


def load_anchor_records(anchors_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(anchors_dir.glob("*_chunks_anchors.jsonl")):
        records.extend(
            record
            for record in load_jsonl(path)
            if record.get("chunk_text") and record.get("anchors")
        )
    return records


def normalize_qa_pairs(qa_data: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(qa_data, dict):
        return [qa_data]
    if isinstance(qa_data, list):
        return qa_data
    raise ValueError("Expected QA JSON object or list of objects")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate grounded RAG QA pairs")
    parser.add_argument("--anchors-dir", required=True, help="Directory containing *_chunks_anchors.jsonl files")
    parser.add_argument("--output-file", required=True, help="Output JSONL file for generated QA records")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH), help="RAG QA generation system prompt")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of chunks to sample")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--save-interval", type=int, default=10, help="Save every N generated records")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY")
    parser.add_argument("--model", default=None, help="Model name; defaults to OPENAI_MODEL")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum LLM request attempts")
    args = parser.parse_args()

    random.seed(args.random_seed)
    anchors_dir = Path(args.anchors_dir)
    output_file = Path(args.output_file)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    llm_config = LLMConfig.from_env(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    records = load_anchor_records(anchors_dir)
    if not records:
        print(f"No anchor records found in {anchors_dir}")
        return 1

    sample_count = min(args.num_samples, len(records))
    sampled_records = random.sample(records, sample_count)
    buffer: list[dict[str, Any]] = []
    processed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for index, record in enumerate(tqdm(sampled_records, desc="Generating QA")):
        chunk_text = record["chunk_text"]
        anchors = record["anchors"]
        document_type = record.get("document_type", "Unknown")
        hooks_json = json.dumps(anchors, ensure_ascii=False)
        user_prompt = build_user_prompt(document_type, chunk_text, hooks_json)

        try:
            raw_json, prompt_tokens, completion_tokens = chat_completion(
                llm_config,
                system_prompt=prompt,
                user_prompt=user_prompt,
            )
            qa_data = parse_json_object(raw_json)
            is_valid, validation_reason = validate_qa_pair(qa_data, chunk_text)
            if not is_valid:
                print(f"Skipping {record.get('chunk_id', index)}: {validation_reason}")
                continue
        except Exception as exc:
            print(f"Skipping {record.get('chunk_id', index)}: {exc}")
            continue

        qa_pairs = normalize_qa_pairs(qa_data)
        for qa_pair in qa_pairs:
            for evidence in qa_pair.get("grounding", {}).get("evidence", []):
                evidence["chunk_index"] = record.get("chunk_index", 0)

        processed += 1
        total_prompt_tokens += int(prompt_tokens or 0)
        total_completion_tokens += int(completion_tokens or 0)
        buffer.append(
            {
                "uid": record.get("uid"),
                "document_type": document_type,
                "domain": record.get("domain", ""),
                "source_file": record.get("source_file", ""),
                "num_chunks": 1,
                "chunk_ids": [record.get("chunk_id", f"chunk_{index}")],
                "merged_hooks": anchors,
                "qa_pairs": qa_pairs,
            }
        )

        if len(buffer) >= args.save_interval:
            append_jsonl(output_file, buffer)
            buffer.clear()

    if buffer:
        append_jsonl(output_file, buffer)

    print(f"Generated {processed} QA records")
    print(f"Prompt tokens: {total_prompt_tokens}")
    print(f"Completion tokens: {total_completion_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
