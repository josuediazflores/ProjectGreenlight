# Project Greenlight

An AI-powered O-1A/EB-1A visa intelligence system that combines a LoRA fine-tuned language model trained on immigration case law with a RAG pipeline grounded in current USCIS policy.

Built as a portfolio application to [LegalOS](https://www.legalos.ai/) (YC W26).

## What It Does

- **Document Upload** — Upload a resume/CV or draft petition; the system extracts qualifications automatically.
- **Criteria Mapping** — Maps evidence against the 8 O-1A evidentiary criteria and flags gaps.
- **Strength Assessment** — Per-criterion strength rating citing policy and real case patterns.
- **RFE Prediction** — Identifies likely Requests for Evidence based on AAO decision patterns.
- **Policy Citation** — Every recommendation links to a specific USCIS Policy Manual section via RAG.
- **Cross-Reference Engine** — Automatic conflict detection between case patterns and current policy.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend                         │
│              (React or Swift)                       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 FastAPI Backend                      │
│         Orchestrates RAG + Inference                │
└────────┬─────────────────────────────┬──────────────┘
         │                             │
┌────────▼────────┐          ┌─────────▼──────────┐
│  Fine-Tuned LLM │          │   RAG Pipeline     │
│  Gemma 4 26B    │          │   ChromaDB /       │
│  + LoRA Adapter │          │   Pinecone         │
│  (vLLM serving) │          │                    │
└─────────────────┘          └────────────────────┘
         │                             │
┌────────▼────────┐          ┌─────────▼──────────┐
│  AAO Case Law   │          │  USCIS Policy      │
│  4,643 decisions│          │  Manual + Forms    │
│  (EB-1A/O-1A)   │          │  + Policy Memos    │
└─────────────────┘          └────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Gemma 4 26B MoE (Apache 2.0, 256K context) |
| Fine-Tuning | LoRA via HuggingFace PEFT on TPU (Google Cloud grant) |
| Inference | vLLM on AMD MI300X (Hack for Humanity dev credits) |
| RAG | ChromaDB (dev) / Pinecone (prod) |
| Backend | Python, FastAPI |
| Embeddings | BGE-large / Nomic Embed / GTE-large |
| Document Parsing | PyMuPDF + Tesseract OCR |

## Data Pipeline

The training data comes from **4,643 AAO Non-Precedent Decisions** on EB-1A extraordinary ability petitions, scraped from the [USCIS AAO repository](https://www.uscis.gov/administrative-appeals/aao-decisions/aao-non-precedent-decisions). The pipeline runs in 5 stages:

### 1. Scrape (`scrape_aao.py`)
Downloads all EB-1A AAO decision PDFs from the USCIS website. Handles pagination, rate limiting, and generates a manifest CSV for tracking.

```bash
python scripts/scrape_aao.py --items-per-page 50 --delay 2.0
```

**Result:** 4,643 PDFs downloaded.

### 2. Parse (`parse_pdfs.py`)
Extracts text from each PDF using PyMuPDF with Tesseract OCR fallback for scanned documents.

```bash
python scripts/parse_pdfs.py
```

**Result:** 4,643 text files extracted (0 failures).

### 3. Score (`score_quality.py`)
Uses Claude to quality-score each case against a [detailed rubric](rubric/scoring_rubric.md) covering criteria coverage, analytical depth, evidence discussion, legal reasoning, and outcome clarity. Auto-rejects procedural dismissals.

```bash
python scripts/score_quality.py --batch-size 10
```

**Result:** 3,181 cases scored. Distribution:
- Score 8.0+: 388 cases (highest quality)
- Score 7.0–7.9: 1,097 cases
- Score 6.0–6.9: ~500 cases
- Auto-rejected: 180 (procedural dismissals, withdrawn appeals, etc.)

### 4. Deduplicate (`deduplicate.py`)
Removes near-duplicate decisions using TF-IDF cosine similarity at a 0.95 threshold. Keeps the higher-scored copy.

```bash
python scripts/deduplicate.py --threshold 0.95
```

**Result:** 18 duplicate pairs removed → **1,467 final cases** in `data/final/`.

### 5. Structured Extraction (`extract_structure.py`)
Uses Claude to decompose each raw AAO decision into structured components:
- Petitioner background/facts
- Evidence submitted per criterion
- AAO analysis per criterion
- Outcome reasoning
- Legal citations
- Fraud/procedural issues

Runs concurrent extraction agents for throughput.

```bash
python scripts/extract_structure.py --batch-size 5
```

### 6. Format Training Data (`format_training_data.py`)
Converts structured extractions into multi-task ChatML/ShareGPT training examples in `.jsonl` format. Generates 4 task types per case:

| Task | Input | Output |
|------|-------|--------|
| **Full criteria analysis** | Petitioner facts + evidence summary | Per-criterion analysis |
| **Single criterion deep-dive** | Evidence for one criterion | Detailed assessment |
| **Gap identification** | Evidence summary | Weaknesses + likely RFEs |
| **Outcome prediction** | Case facts + criteria results | Predicted outcome + reasoning |

```bash
python scripts/format_training_data.py --min-score 7.5 --tasks all
```

**Target:** ~11,000 training examples across all task types.

## Evaluation (`benchmark_eval.py`)

Compares model performance before and after LoRA fine-tuning on a held-out test set:

- **Criteria identification** — F1 score on which criteria the model identifies
- **Met/not-met accuracy** — Whether the model correctly predicts if each criterion was satisfied
- **Gap identification** — Recall on weaknesses and RFE predictions
- **Outcome prediction** — Accuracy on sustain/dismiss/remand
- **Optional LLM judge** — Claude rates reasoning quality on a 1–5 scale

```bash
# Run baseline eval against untuned Gemma 4
python scripts/benchmark_eval.py --run-name baseline --endpoint http://localhost:8000/v1

# Run post-LoRA eval
python scripts/benchmark_eval.py --run-name post-lora --endpoint http://localhost:8000/v1

# Compare runs
python scripts/benchmark_eval.py --compare baseline post-lora
```

## Setup

```bash
# Clone
git clone https://github.com/josuediazflores/ProjectGreenlight.git
cd ProjectGreenlight

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Project Status

- [x] Phase 1 — Data collection & curation (4,643 decisions scraped, parsed, scored, deduped)
- [ ] Phase 2 — Structured extraction + LoRA fine-tuning (in progress)
- [ ] Phase 3 — RAG pipeline over USCIS Policy Manual
- [ ] Phase 4 — Application layer & demo
- [ ] Phase 5 — Polish & LegalOS application

## Data Sources

| Source | Type | Access |
|--------|------|--------|
| [USCIS AAO Repository](https://www.uscis.gov/administrative-appeals/aao-decisions/aao-non-precedent-decisions) | EB-1A case decisions | Public domain |
| [USCIS Policy Manual](https://www.uscis.gov/policy-manual) | Immigration policy (Vol 2, Part M) | Public domain |
| I-129 / I-140 Form Instructions | Filing requirements | Public domain |

> **Note:** USCIS stopped publishing new AAO decisions in March 2025. All training data comes from pre-March 2025 decisions.

## License

MIT
