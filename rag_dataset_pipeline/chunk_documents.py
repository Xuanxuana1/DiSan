#!/usr/bin/env python3
"""Chunk JSONL documents for downstream anchor and QA generation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from json_utils import write_jsonl

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "clients_original_data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "build" / "chunks"

try:
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    try:
        from llama_index.schema import Document
        from llama_index.text_splitter import SentenceSplitter

        LLAMA_INDEX_AVAILABLE = True
    except ImportError:
        LLAMA_INDEX_AVAILABLE = False


class DocumentChunker:
    """Sentence-aware chunker with a paragraph fallback."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_splitter = None
        if LLAMA_INDEX_AVAILABLE:
            self.sentence_splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    def chunk_document(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        if self.sentence_splitter is not None:
            document = Document(text=text, metadata=metadata)
            nodes = self.sentence_splitter.get_nodes_from_documents([document])
            return [
                {
                    "text": node.text,
                    "metadata": {
                        **metadata,
                        "chunk_index": index,
                        "chunk_type": "llamaindex_sentence",
                        "chunk_size": len(node.text.split()),
                        "has_overlap": index > 0,
                        "overlap_size": self.chunk_overlap if index > 0 else 0,
                    },
                    "chunk_id": f"{metadata.get('uid', 'unknown')}_chunk_{index}",
                }
                for index, node in enumerate(nodes)
            ]

        chunks = self._basic_sentence_chunking(text)
        return [
            {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": index,
                    "chunk_type": "basic_sentence",
                    "chunk_size": len(chunk.split()),
                    "has_overlap": index > 0,
                    "overlap_size": self.chunk_overlap if index > 0 else 0,
                },
                "chunk_id": f"{metadata.get('uid', 'unknown')}_chunk_{index}",
            }
            for index, chunk in enumerate(chunks)
        ]

    def _basic_sentence_chunking(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        chunks: list[str] = []
        current_chunk = ""
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = len(paragraph.split())
            if current_tokens + paragraph_tokens <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
                current_tokens += paragraph_tokens
                continue

            if current_chunk:
                chunks.append(current_chunk.strip())

            if paragraph_tokens > self.chunk_size:
                current_chunk, current_tokens = self._split_long_paragraph(paragraph, chunks)
            else:
                current_chunk = paragraph + "\n\n"
                current_tokens = paragraph_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]
        for index in range(1, len(chunks)):
            previous_words = chunks[index - 1].split()
            overlap_words = previous_words[-self.chunk_overlap :]
            overlapped_chunks.append(" ".join(overlap_words + chunks[index].split()))
        return overlapped_chunks

    def _split_long_paragraph(self, paragraph: str, chunks: list[str]) -> tuple[str, int]:
        sentence_parts = re.split(r"(?<=[.!?])\s+", paragraph)
        current_chunk = ""
        current_tokens = 0
        for sentence in sentence_parts:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= self.chunk_size:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
                continue
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_tokens = sentence_tokens
        return current_chunk, current_tokens


def iter_input_files(input_dir: Path, max_files: int | None) -> list[Path]:
    files = sorted(path for path in input_dir.rglob("*.jsonl") if path.is_file())
    if max_files is not None:
        return files[:max_files]
    return files


def build_metadata(
    sample: dict[str, Any],
    input_file: Path,
    input_dir: Path,
    sample_index: int,
) -> dict[str, Any]:
    try:
        source_path = input_file.relative_to(input_dir)
    except ValueError:
        source_path = Path(input_file.name)

    source_file = str(source_path)
    default_uid = f"{source_path.with_suffix('').as_posix()}:{sample_index}"

    return {
        "uid": sample.get("uid", default_uid),
        "domain": sample.get("domain", ""),
        "document_type": sample.get("document_type", "unknown"),
        "source_file": source_file,
        "sample_index": sample_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk JSONL documents for RAG dataset generation")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing input JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where chunk JSONL files are written",
    )
    parser.add_argument("--text-field", default="generated_text", help="Input record field containing document text")
    parser.add_argument("--chunk-size", type=int, default=256, help="Approximate chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Word overlap between chunks")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of files to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    files = iter_input_files(input_dir, args.max_files)
    if not files:
        print(f"No JSONL files found in {input_dir}")
        return 1

    total_chunks = 0
    grouped_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for input_file in tqdm(files, desc="Chunking files"):
        with input_file.open("r", encoding="utf-8") as source:
            for sample_index, line in enumerate(source):
                if not line.strip():
                    continue
                sample = json.loads(line)
                text = sample.get(args.text_field, "")
                if not text:
                    continue
                metadata = build_metadata(sample, input_file, input_dir, sample_index)
                chunks = chunker.chunk_document(text, metadata)
                document_type = str(metadata["document_type"]).replace(" ", "_")
                grouped_chunks[document_type].extend(chunks)
                total_chunks += len(chunks)

    for document_type, chunks in sorted(grouped_chunks.items()):
        write_jsonl(output_dir / f"{document_type}_chunks.jsonl", chunks)

    print(f"Wrote {total_chunks} chunks to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
