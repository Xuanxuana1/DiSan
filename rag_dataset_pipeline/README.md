# RAG Dataset Pipeline

This directory provides a pipeline for generating grounded RAG evaluation QA
pairs from JSONL document records.


## Pipeline

1. Chunk input documents.
2. Extract retrieval-oriented semantic anchors from each chunk.
3. Generate grounded QA pairs from anchor-annotated chunks.
4. Convert chunks into per-party RAG context files.
5. Run local or multi-party hybrid retrieval over the prepared contexts.

## Input Format

The chunking step expects JSONL input files. Each line should be a JSON object
with at least a text field. By default the text field is `generated_text`; it can
be changed with `--text-field`. The default input directory is
`../data/clients_original_data`, and nested client folders are scanned
recursively.

Optional metadata fields are preserved when present:

```json
{
  "uid": "record-001",
  "domain": "finance",
  "document_type": "Policy",
  "generated_text": "Document text to chunk..."
}
```

## Configuration

The LLM calls use an OpenAI-compatible `/chat/completions` endpoint. Configure it
with environment variables:

```bash
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="replace-with-api-key"
export OPENAI_MODEL="gpt-4o-mini"
```

The `--base-url`, `--api-key`, and `--model` arguments can also be passed directly to
`extract_anchors.py` and `generate_rag_qa.py`.

## Usage

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run the pipeline from this directory:

```bash
python chunk_documents.py \
  --input-dir ../data/clients_original_data \
  --output-dir ./build/chunks \
  --text-field generated_text \
  --chunk-size 256 \
  --chunk-overlap 50

python extract_anchors.py \
  --input-dir ./build/chunks \
  --output-dir ./build/anchors

python generate_rag_qa.py \
  --anchors-dir ./build/anchors \
  --output-file ./build/rag_qa_pairs.jsonl \
  --num-samples 100 \
  --random-seed 42

python prepare_rag_contexts.py \
  --chunks-dir ./build/chunks \
  --output-dir ./build/rag_contexts
```

The final step groups chunks by `Client_N` markers found in metadata such as
`source_file`, `client_id`, `party_id`, or `chunk_id`. If no marker is found, the
record is assigned to `Client_0`.

To start retrieval services for all prepared clients:

```bash
export BGE_M3_PATH=../bge-m3
export BGE_RERANKER_PATH=../bge-reranker-v2-m3
export CONTEXT_DIR=./build/rag_contexts
export CUDA_VISIBLE_DEVICES=0
./start_retrieval_services.sh
```

Each service exposes:

```text
GET  /health
POST /retrieve
GET  /info
POST /cleanup
```

For a single local context file, instantiate `AdvancedHybridRetrieval` directly:

```python
from advanced_hybrid_retrieval import AdvancedHybridRetrieval

retriever = AdvancedHybridRetrieval(
    contexts_path="./build/rag_contexts/client_1_contexts.jsonl",
    bge_m3_path="../bge-m3",
    reranker_path="../bge-reranker-v2-m3",
)
results = retriever.retrieve("What changed in Q3 risk exposure?")
```

## Outputs

Chunk files:

```text
build/chunks/<Document_Type>_chunks.jsonl
```

Anchor files:

```text
build/anchors/<Document_Type>_chunks_anchors.jsonl
```

QA file:

```text
build/rag_qa_pairs.jsonl
```

RAG context files:

```text
build/rag_contexts/client_<N>_contexts.jsonl
```

Each QA record includes source metadata, chunk IDs, merged hooks, the generated
query, a ground-truth answer, and evidence snippets used for grounding.

Each RAG context record contains `content`, `chunk_id`, `source_line`,
`document_type`, `uid`, `chunk_index`, and preserved metadata. This is the input
format consumed by `advanced_hybrid_retrieval.py` and
`retrieval_api_service.py`.
