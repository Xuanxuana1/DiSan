# DiSan Anonymous Code Release

This repository contains the code needed to reproduce the main DiSan training, retrieval, and privacy-attack experiments. 

## Directory Layout

- `fed_lightweight/`: lightweight federated DiSan training code.
- `attack_light/`: privacy and attribution experiments.
- `rag_dataset_pipeline/`: RAG dataset construction plus hybrid retrieval.
- `data/`: small example or placeholder data layout. Replace with the datasets
  used for a specific reproduction run.
- `long-t5-tglobal-base/`, `bge-m3/`, `bge-reranker-v2-m3/`,
  `gliner-pii-large-v1.0/`: placeholder directories documenting expected local model locations. Do not commit model weights unless their licenses permit it.


## Environment

Create an environment with the dependencies listed in the root
`requirements.txt`:

```bash
pip install -r requirements.txt
```

For LLM calls in the RAG dataset pipeline, use environment variables or a local
`.env` file based on `rag_dataset_pipeline/.env.example`:

```bash
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="replace-with-api-key"
export OPENAI_MODEL="gpt-4o"
```

Do not commit filled `.env` files, API keys, checkpoints, generated outputs, or
raw private data.

## Federated DiSan Training

From the repository root:

```bash
bash fed_lightweight/train_lora_v2.sh
```

The script uses relative paths by default:

- training data: `./data`
- outputs: `./checkpoints/fed_lora_v2`
- logs: `./logs`

Edit the client list only if the released or local dataset uses different
anonymous client identifiers.

## Attack Experiments

Representation probe:

```bash
python -m attack_light.exp1_representation_probe \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints/fed_lora_v2 \
  --output_dir ./outputs/exp1_representation_probe
```

Prototype attack:

```bash
python -m attack_light.exp2_prototype_attack \
  --data_dir ./data \
  --checkpoint_dir ./checkpoints/fed_lora_v2 \
  --output_dir ./outputs/exp2_prototype_attack
```

Text stylometry on the synthetic/sanitized JSONL records:

```bash
python -m attack_light.exp3a_text_stylometry \
  --data_dir ./data \
  --output_dir ./outputs/exp3a_text_stylometry
```

## EXP-3b Enron Placeholder-Masking Attack

EXP-3b has three stages so the same fixed sample set is reused across raw,
placeholder-masked, and DiSan-sanitized attribution probes.

1. Build the fixed Enron sample set and evaluate raw text:

```bash
python -m attack_light.exp3b_eval_enron_stylometry_bert \
  --enron_path ./data/enron_data \
  --samples_per_sender 500 \
  --save_samples ./outputs/enron_exp3b_samples.jsonl \
  --output_dir ./outputs/enron_exp3b
```

2. Generate placeholder-masked files with the three PII detectors:

```bash
export GLINER_MODEL=./gliner-pii-large-v1.0
export PIIRANHA_MODEL=/path/to/piiranha-v1-detect-personal-information
export DEBERTA_PII_MODEL=/path/to/deberta-pii-finetuned

python -m attack_light.exp3b_placeholder_masking \
  --input ./outputs/enron_exp3b_samples.jsonl \
  --output_dir ./outputs/enron_exp3b/placeholders \
  --detectors gliner piiranha deberta
```

The masking script writes one JSONL and one summary JSON per detector, including
span counts and token-mask rates.

3. Run the same TF-IDF/BERT attribution probe on all placeholder files:

```bash
python -m attack_light.exp3b_eval_enron_stylometry_bert \
  --samples_file ./outputs/enron_exp3b_samples.jsonl \
  --masked_jsonl \
    gliner=./outputs/enron_exp3b/placeholders/enron_placeholder_gliner.jsonl \
    piiranha=./outputs/enron_exp3b/placeholders/enron_placeholder_piiranha.jsonl \
    deberta=./outputs/enron_exp3b/placeholders/enron_placeholder_deberta.jsonl \
  --output_dir ./outputs/enron_exp3b
```

To include DiSan-sanitized Enron text in the same comparison, add
`--checkpoint ./checkpoints/fed_lora_v2/final_model.pt`. If the base model is not
in `./long-t5-tglobal-base`, set `DISAN_BASE_MODEL=/path/to/base-model`.

## RAG Pipeline

Use `rag_dataset_pipeline/` as the single RAG workflow:

```bash
cd rag_dataset_pipeline
python chunk_documents.py --input-dir ./input_jsonl --output-dir ./build/chunks
python extract_anchors.py --input-dir ./build/chunks --output-dir ./build/anchors
python generate_rag_qa.py --anchors-dir ./build/anchors --output-file ./build/rag_qa_pairs.jsonl
python prepare_rag_contexts.py --chunks-dir ./build/chunks --output-dir ./build/rag_contexts
./start_retrieval_services.sh
```

See `rag_dataset_pipeline/README.md` for detailed input and output formats.
