# Roadmap

This project is best understood as a structured learning and experimentation roadmap for modern LLM development.

## Current focus

- tokenizer implementation from scratch
- decoder-only transformer understanding
- causal training behavior
- checkpoint management
- assistant-style fine-tuning
- local interactive generation

## Near-term milestones

### M1: core language model fundamentals

- BPE vocabulary training
- token embedding and positional encoding
- causal attention and decoder blocks
- training and validation loops
- text generation from checkpoints

### M2: assistant adaptation

- reserved role-token setup
- dialogue formatting and masking
- assistant-only loss labeling
- standalone fine-tuning workflow
- context-window-aware chat loops

### M3: practical experimentation

- larger corpora and stronger training runs
- deeper evaluation of generation quality
- prompt and sampling analysis
- tuning behavior across model settings

## Longer-term direction

- improved dataset quality and data hygiene
- more robust evaluation metrics
- better sampling strategies
- larger models for real-world experimentation
- clearer separation between pretraining and assistant specialization

## Engineering philosophy

This repository will continue to prioritize:

- clarity over cleverness
- transparency over abstraction
- reproducibility over hype
- local experimentation over cloud dependence

The goal is not to chase a perfect chatbot; it is to understand how an LLM can be built, trained, adapted, and reasoned about in a controlled local environment.
