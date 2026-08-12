# LLM From Scratch

A compact, local-first LLM project built from first principles in PyTorch. This repository is designed to make the full language-model stack visible and understandable: tokenization, embeddings, causal attention, training, checkpointing, generation, and dialogue fine-tuning.

<p align="center">
  <a href="#overview"><img src="https://img.shields.io/badge/python-3.10%2B-3776AB" alt="Python 3.10+" /></a>
  <a href="#setup"><img src="https://img.shields.io/badge/torch-2.x-EE4C2C" alt="PyTorch 2.x" /></a>
  <a href="#project-structure"><img src="https://img.shields.io/badge/local-first-LLM-000000" alt="Local first" /></a>
  <a href="#status"><img src="https://img.shields.io/badge/status-learning%20project-6EE7B7" alt="Status" /></a>
</p>

> This repo is focused on learning, experimentation, and understanding how modern LLMs are assembled from source-level building blocks. It is not a production deployment stack and it intentionally avoids external APIs or hosted model services.

## Overview

This project examines the practical mechanics behind a small decoder-only Transformer:

- BPE tokenization built from scratch
- vocabulary learning from raw text
- next-token prediction training
- validation loss tracking
- text generation with temperature and top-k sampling
- checkpoint save/load workflows
- adaptation to a dialogue format for assistant-style interactions

The goal is not only to make the model work, but to make the logic visible enough to reason about how each piece contributes to the whole system.

## Why this repo exists

The strongest way to understand an LLM is to build one with the core ingredients in view.

This repo does that without hiding the system behind abstractions. It exposes the training loop, sampling logic, tokenizer details, checkpoint structure, and fine-tuning workflow in a way that is easy to inspect, modify, and learn from.

## What it can do

- Train a causal Transformer on plain-text corpora
- Learn BPE merges directly from a dataset
- Save and restore local checkpoints
- Generate text with controlled sampling behavior
- Adapt a pretrained checkpoint for assistant-style dialogue training
- Run a terminal-based chat experience using a fine-tuned model

## Project structure

```text
LLM/
├── char_transformer.py                    # Training, evaluation, generation, and terminal chat
├── tokenization.py                       # From-scratch BPE tokenizer
├── device.py                             # CUDA -> MPS -> CPU device selection
├── assistant_data.py                     # Dialogue formatting and assistant-only masking logic
├── prepare_assistant_checkpoint.py       # Expands checkpoint vocab with reserved dialogue tokens
├── finetune_assistant.py                 # Fine-tuning loop for dialogue-style training
├── assistant_chat.py                     # Interactive terminal chat with sliding context window
├── decoder.py                            # Decoder block experiment and standalone attention flow
├── mps_smoke_test.py                     # Device-level forward/backward smoke test
├── raj_bio.txt                           # Small example corpus for text generation
├── raj_chat.txt                          # Example Q/A text data for conversational training
├── data/
│   ├── assistant_examples.jsonl          # Example dialogue JSONL dataset
│   └── large_text.txt                    # Larger text source used for bigger runs
├── test_tiny.py                          # Quick validation checks
├── test_stage4.py                        # Stage 4 validation for token and checkpoint logic
├── requirements.txt                      # Core project dependency set
├── README.md                             # High-level project documentation
├── docs/
│   ├── OVERVIEW.md                       # Project overview and technical narrative
│   └── ROADMAP.md                        # Milestones and future direction
└── .gitignore                            # Standard local artifact exclusions
```

## Status

This project is currently a research and learning repository. It demonstrates a practical path from:

1. tokenization
2. autoregressive modeling
3. checkpoint reuse
4. dialogue adaptation
5. interactive generation

It is intentionally compact, readable, and local-first rather than optimized for production-scale deployment.

## Requirements

- Python 3.10+
- PyTorch 2.x
- A local machine with CPU, MPS, or CUDA capability

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Check versions:

```bash
python3 --version
python3 -c "import torch; print(torch.__version__)"
```

## Quick start

### 1. Train on a small text corpus

```bash
python3 char_transformer.py \
  --input raj_bio.txt \
  --steps 1500 \
  --save raj_model.pt
```

### 2. Generate text from a saved checkpoint

```bash
python3 char_transformer.py \
  --steps 0 \
  --load raj_model.pt \
  --prompt "Raj Pandit" \
  --sample-length 250
```

### 3. Run a chat session

```bash
python3 char_transformer.py \
  --input raj_chat.txt \
  --steps 3000 \
  --save raj_chat_model.pt \
  --chat
```

## Core training flow

### Stage 1: text and tokenizer

The project begins with raw text and builds a BPE tokenizer directly from the corpus. This gives the model a compact vocabulary and exposes the mechanics of tokenization rather than hiding them behind a packaged library.

### Stage 2: autoregressive learning

The model learns to predict the next token from previous context. The training loop records metrics such as training and validation loss so the learning process is transparent and measurable.

### Stage 3: model checkpointing and generation

Once trained, weights can be saved and reloaded without retraining. Generation is controlled using sampling parameters such as temperature and top-k to produce different output styles.

### Stage 4: assistant-style fine-tuning

The repo adds a dedicated dialogue workflow:

- expand the base checkpoint with reserved role tokens
- format user/assistant messages into a consistent structure
- train on assistant-only loss masking
- run an interactive terminal conversation with context-window management

## Stage 4 workflow

### 1. Validate the hardware path

```bash
python3 mps_smoke_test.py --require-mps --checkpoint small_gpt.pt
```

### 2. Expand the base checkpoint for dialogue tokens

```bash
python3 prepare_assistant_checkpoint.py \
  --base-checkpoint small_gpt.pt \
  --output checkpoints/small_gpt_assistant_base.pt
```

### 3. Fine-tune on dialogue data

```bash
python3 finetune_assistant.py \
  --base-checkpoint checkpoints/small_gpt_assistant_base.pt \
  --data data/assistant_examples.jsonl \
  --output checkpoints/small_gpt_assistant_finetuned.pt \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 1e-5
```

### 4. Talk to the fine-tuned assistant

```bash
python3 assistant_chat.py \
  --checkpoint checkpoints/small_gpt_assistant_finetuned.pt \
  --temperature 0.5 \
  --top-k 12 \
  --top-p 0.9
```

## Sampling controls

The generation loop exposes a few important knobs:

- `--temperature`: controls randomness in sampling
- `--top-k`: restricts generation to the most likely tokens
- `--top-p`: filters the candidate set by cumulative probability

Lower temperature values produce safer, more conservative outputs. Higher values yield more creative but less stable generation.

## Validation and verification

Quick checks are included for local validation:

```bash
python3 test_tiny.py
python3 test_stage4.py
```

These tests cover tokenizer behavior, checkpoint handling, loss sanity, and assistant-oriented formatting logic.

## What this repo is not

This project is intentionally not:

- a production inference service
- a hosted chatbot backend
- a major pretrained model deployment pipeline
- a general-purpose assistant stack with RAG, tools, or persistent memory

It is a transparent learning project that makes the engineering logic visible so it can be understood and expanded.

## Repository conventions

- code is intentionally explicit and readable
- checkpoints are kept local and reproducible
- the model is trained on compact, human-inspectable datasets
- experiments are designed to be understandable rather than hidden behind opaque abstractions

## Roadmap and direction

This repository is best understood as an educational progression in building a small LLM stack. The current work focuses on a clean, transparent path from raw text to working local generation and dialogue adaptation.

For the longer-term direction, see the docs in [docs/ROADMAP.md](docs/ROADMAP.md).

For the broader background and technical framing, see [docs/OVERVIEW.md](docs/OVERVIEW.md).

## License

This project is organized for education and experimentation. Please check the repository license and any dataset constraints before using materials outside the local learning workflow.

---

Built for understanding, iteration, and experimentation in modern LLM systems.
