#!/usr/bin/env python3
"""Convert chunked pipeline outputs into per-party RAG context files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CLIENT_PATTERN = re.compile(r"(Client_\d+)", re.IGNORECASE)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def infer_party_id(chunk: dict[str, Any], fallback: str) -> str:
    metadata = chunk.get("metadata") or {}
    for value in (
        metadata.get("client_id"),
        metadata.get("party_id"),
        metadata.get("source_file"),
        chunk.get("chunk_id"),
        fallback,
    ):
        if not value:
            continue
        match = CLIENT_PATTERN.search(str(value))
        if match:
            return match.group(1)
    return fallback


def convert_chunk(chunk: dict[str, Any], global_index: int) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    text = chunk.get("text") or chunk.get("content") or ""
    chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id") or f"chunk_{global_index}"
    source_file = metadata.get("source_file", "unknown")
    chunk_index = metadata.get("chunk_index", 0)

    return {
        "content": text,
        "source_line": f"{source_file}:{chunk_index}",
        "length": len(text),
        "chunk_id": chunk_id,
        "uid": metadata.get("uid"),
        "domain": metadata.get("domain", ""),
        "document_type": metadata.get("document_type", "unknown"),
        "source_file": source_file,
        "sample_index": metadata.get("sample_index"),
        "chunk_index": chunk_index,
        "global_context_index": global_index,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare per-party RAG contexts from chunk JSONL files")
    parser.add_argument("--chunks-dir", required=True, help="Directory containing *_chunks.jsonl files")
    parser.add_argument("--output-dir", required=True, help="Directory for *_contexts.jsonl files")
    parser.add_argument(
        "--default-party",
        default="Client_0",
        help="Fallback party ID when a chunk has no Client_N marker",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)
    files = sorted(path for path in chunks_dir.rglob("*_chunks.jsonl") if path.is_file())
    if not files:
        print(f"No *_chunks.jsonl files found in {chunks_dir}")
        return 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_index = 0
    for path in files:
        for chunk in iter_jsonl(path):
            if not (chunk.get("text") or chunk.get("content")):
                continue
            party_id = infer_party_id(chunk, fallback=args.default_party)
            grouped[party_id].append(convert_chunk(chunk, global_index))
            global_index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for party_id, contexts in sorted(grouped.items()):
        out_path = output_dir / f"{party_id.lower()}_contexts.jsonl"
        with out_path.open("w", encoding="utf-8") as sink:
            for context in contexts:
                sink.write(json.dumps(context, ensure_ascii=False) + "\n")
        print(f"Wrote {len(contexts)} contexts to {out_path}")

    print(f"Prepared {global_index} contexts for {len(grouped)} parties")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
