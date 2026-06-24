# Running Examples

These examples give DiSan a concrete story that can be reused in the README,
project page, talks, and demos. They are derived from the released sample
finance records under `data/clients_original_data/` and from the project-page
threat model.

The sanitized outputs below are illustrative DiSan-style outputs. Exact text
depends on the trained checkpoint, base model, decoding settings, and the data
used for a reproduction run. The public code release includes sample records and
pipeline code, but not model weights, private checkpoints, or full benchmark
datasets.

## Example 1: One Evidence Snippet

**Scenario.** A corporate-bank agent needs to answer an operational-risk query
for another agent:

```text
Query: What controls should be recommended when operational risk comes from
single-site data-center dependence and exposed customer credentials?
```

A retrieved record from
`data/clients_original_data/Client_1_CorporateBank/Financial_Risk_Assessment.jsonl`
contains useful business evidence, but it also contains explicit PII and
source-specific wording.

### Raw Evidence

```text
The company's heavy dependence on a single data center located at
73 rue David Menard, BoucherBourg, poses a risk to business continuity.
...
The company's financial transactions involve sensitive information, such as
Vicenta D. Martin's account pin (059332) and phone number (3505216216).
```

What should be preserved:

- business continuity risk from a single data center
- fraud risk from exposed customer credentials
- recommended controls such as backups, alternative work arrangements,
  multi-factor authentication, software updates, and security training

What should not be shared:

- street address
- person name
- account PIN
- phone number
- source-specific drafting habits that help identify the client

### Placeholder Baseline

```text
The company's heavy dependence on a single data center located at
[STREET_ADDRESS], poses a risk to business continuity.
...
The company's financial transactions involve sensitive information, such as
[NAME]'s account pin ([ACCOUNT_PIN]) and phone number ([PHONE_NUMBER]).
```

This removes obvious identifiers, but it keeps the source's phrasing,
document structure, and distributional signature. A stylometry attacker can
still exploit repeated templates, vocabulary, and section ordering.

### DiSan-Style Output

```text
The operational-risk report flags two actionable exposures: dependence on a
single physical data-center site, which weakens business continuity, and
customer-credential exposure, which increases fraud risk. Recommended controls
include disaster recovery planning, redundant access paths, alternative work
arrangements, multi-factor authentication, regular software updates, and staff
security training.
```

DiSan's intended behavior is to preserve the role semantics needed by the
downstream task while suppressing both explicit identifiers and source-style
signals.

## Example 2: Multi-Party RAG Answer

**Scenario.** A coordinator agent asks a question that requires evidence from
multiple parties:

```text
Query: How should a financial-services alliance prepare for cyber incidents
that could disrupt operations, supplier delivery, and compliance reporting?
```

The RAG pipeline can retrieve evidence from several client folders:

| Party | Example document | Useful signal | Sensitive/source signal |
| --- | --- | --- | --- |
| `Client_1_CorporateBank` | Financial Risk Assessment | cyber threats, business continuity, fraud controls | addresses, names, account data |
| `Client_7_SupplierCo` | Supply Chain Management Agreement | supplier-base diversity, contingency planning, availability and lead-time metrics | client names and contract wording |
| `Client_6_ComplianceConsult` | Regulatory Compliance Guide | vehicle or insurance compliance, record keeping, safety and privacy controls | names, network identifiers, addresses |

### DiSan-Sanitized Evidence View

```text
Evidence A - Operational risk:
The financial-risk record links cyber incidents to market, credit, and
operational disruption. It recommends continuity planning, security controls,
and personnel training.

Evidence B - Supplier resilience:
The supply-chain agreement requires a written disruption plan, alternative
sources for critical components, notification procedures, and resilience
metrics such as availability and lead time.

Evidence C - Compliance operations:
The compliance guide emphasizes documented inspections, training records,
privacy-safe handling of technical identifiers, and auditable records.
```

### Grounded Answer

```text
The alliance should prepare a three-part response plan. First, reduce cyber
operational risk with MFA, patching, employee security training, and disaster
recovery procedures. Second, protect supplier continuity by maintaining
alternative sources for critical inputs, setting availability and lead-time
metrics, and predefining disruption-notification workflows. Third, keep the
response auditable by recording inspections, training, incident actions, and
data-handling controls. The shared answer should expose these controls and
metrics, not the names, addresses, account credentials, internal templates, or
client-specific drafting patterns found in the source records.
```

This example shows the target behavior in distributed RAG: agents can share the
task-relevant evidence needed for a useful answer while reducing the attribution
surface seen by other parties.

## Example 3: Attribution Attack View

DiSan is motivated by a privacy failure that placeholder masking does not solve.
The project-page example uses a stylometric leak:

```text
Raw / masked-style evidence:
Per [ORG]'s Counterparty Risk Bulletin (Ref: [ID]), [ORG] carried $6.1M
exposure as of Q3 close, was downgraded to BB+, and was flagged for portfolio
review.
```

Even after entity masking, a source classifier can still learn signals such as:

- "Counterparty Risk Bulletin"
- "Ref: [ID]"
- downgrade and portfolio-review phrasing
- document-specific clause order

A DiSan-style output should keep the financial event but make the source less
identifiable:

```text
A Q3 risk review flags an industrial counterparty with $6.1M exposure, a BB+
downgrade, and a required portfolio review.
```

The privacy experiments under `attack_light/` test this threat model with:

- text stylometry attacks on raw and sanitized JSONL fields
- representation probes over role/style embeddings
- prototype attacks over client-level uploaded representations
- Enron placeholder-masking comparisons for fixed author samples

## Reproducing the Data Flow Locally

The public sample data is enough to exercise the RAG preparation path:

```bash
cd rag_dataset_pipeline

python chunk_documents.py \
  --input-dir ../data/clients_original_data \
  --output-dir ./build/chunks \
  --text-field generated_text \
  --chunk-size 256 \
  --chunk-overlap 50

python prepare_rag_contexts.py \
  --chunks-dir ./build/chunks \
  --output-dir ./build/rag_contexts
```

After this, inspect per-client context files such as:

```text
rag_dataset_pipeline/build/rag_contexts/client_1_contexts.jsonl
rag_dataset_pipeline/build/rag_contexts/client_7_contexts.jsonl
```

With the current public sample release, this path produces 6676 chunks and 7
per-client context files.

The full QA-generation and retrieval stack additionally requires configured LLM
credentials and local BGE model weights:

```bash
python extract_anchors.py \
  --input-dir ./build/chunks \
  --output-dir ./build/anchors

python generate_rag_qa.py \
  --anchors-dir ./build/anchors \
  --output-file ./build/rag_qa_pairs.jsonl \
  --num-samples 100 \
  --random-seed 42

export BGE_M3_PATH=../bge-m3
export BGE_RERANKER_PATH=../bge-reranker-v2-m3
export CONTEXT_DIR=./build/rag_contexts
./start_retrieval_services.sh
```

Federated DiSan training expects paired fields such as `original_text` and
`rewritten_text`. The small public `clients_original_data` release is intended
mainly for layout and RAG-pipeline examples; replace it with the paired training
data used for a reproduction run before launching:

```bash
bash fed_lightweight/train_lora_v2.sh
```

## What to Look For

A good running example should make these checks easy:

- **Utility:** the answer still contains the relevant risk, compliance, and
  mitigation facts.
- **Explicit privacy:** names, addresses, account credentials, emails, phone
  numbers, and technical identifiers are removed or generalized.
- **Attribution privacy:** client-specific templates and wording are rewritten
  into a more source-invariant style.
- **Grounding:** every answer claim can be traced back to a sanitized evidence
  snippet or source chunk.
