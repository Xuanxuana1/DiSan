#!/usr/bin/env python3
"""Extract semantic anchors from chunk JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from json_utils import append_jsonl, load_jsonl, parse_json_object
from llm_client import LLMConfig, chat_completion

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "anchor_prompt.md"


def build_user_prompt(document_type: str, chunk_text: str) -> str:
    return f"Document Type: {document_type}\nRewritten Chunk: {chunk_text}"


def process_file(
    *,
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    llm_config: LLMConfig,
    save_interval: int,
) -> tuple[int, int, int]:
    records = [
        record
        for record in load_jsonl(input_path)
        if record.get("text") and isinstance(record.get("metadata"), dict)
    ]
    buffer: list[dict] = []
    processed = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for index, record in enumerate(tqdm(records, desc=f"Anchors: {input_path.name}")):
        metadata = record.get("metadata", {})
        user_prompt = build_user_prompt(metadata.get("document_type", "Unknown"), record["text"])
        try:
            raw_json, prompt_tokens, completion_tokens = chat_completion(
                llm_config,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            anchors = parse_json_object(raw_json)
        except Exception as exc:
            print(f"Skipping {record.get('chunk_id', index)}: {exc}")
            continue

        processed += 1
        prompt_tokens_total += int(prompt_tokens or 0)
        completion_tokens_total += int(completion_tokens or 0)
        buffer.append(
            {
                "chunk_id": record.get("chunk_id", f"chunk_{index}"),
                "uid": metadata.get("uid"),
                "domain": metadata.get("domain", ""),
                "document_type": metadata.get("document_type", "Unknown"),
                "source_file": metadata.get("source_file", ""),
                "sample_index": metadata.get("sample_index", 0),
                "chunk_index": metadata.get("chunk_index", 0),
                "chunk_text": record["text"],
                "anchors": anchors,
            }
        )

        if len(buffer) >= save_interval:
            append_jsonl(output_path, buffer)
            buffer.clear()

    if buffer:
        append_jsonl(output_path, buffer)

    return processed, prompt_tokens_total, completion_tokens_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract semantic anchors from chunked documents")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_chunks.jsonl files")
    parser.add_argument("--output-dir", required=True, help="Directory where *_chunks_anchors.jsonl files are written")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH), help="Anchor extraction system prompt")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of files to process")
    parser.add_argument("--save-interval", type=int, default=10, help="Save every N records")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY")
    parser.add_argument("--model", default=None, help="Model name; defaults to OPENAI_MODEL")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum LLM request attempts")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    llm_config = LLMConfig.from_env(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    files = sorted(path for path in input_dir.glob("*_chunks.jsonl") if path.is_file())
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print(f"No *_chunks.jsonl files found in {input_dir}")
        return 1

    total_processed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for input_path in files:
        output_path = output_dir / f"{input_path.stem}_anchors.jsonl"
        processed, prompt_tokens, completion_tokens = process_file(
            input_path=input_path,
            output_path=output_path,
            system_prompt=prompt,
            llm_config=llm_config,
            save_interval=args.save_interval,
        )
        total_processed += processed
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

    print(f"Processed {total_processed} chunks")
    print(f"Prompt tokens: {total_prompt_tokens}")
    print(f"Completion tokens: {total_completion_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
