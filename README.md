# LLM From Scratch

A compact, educational decoder-only Transformer built with PyTorch. The project
is designed to make the core parts of a language model visible: tokenization,
embeddings, causal self-attention, training, evaluation, sampling, and local
text generation.

It uses no APIs, no pretrained models, and no tokenizer libraries.

## What it can do

- Learn a BPE vocabulary directly from a plain-text file
- Train a causal Transformer to predict the next token
- Report training and validation loss during training
- Generate text using temperature and top-k sampling
- Save and reload complete local checkpoints
- Run a small terminal chat when trained on question-and-answer examples

This is a learning project, not a replacement for a production chat model. A
small model trained on a small file can imitate its source text, but it cannot
reliably reason about arbitrary questions.

## Project structure

```text
LLM/
├── char_transformer.py  # Training, evaluation, generation, and terminal chat
├── tokenization.py      # From-scratch BPE tokenizer
├── device.py            # CUDA → MPS → CPU device selection
├── assistant_data.py    # Stage 4 dialogue format and assistant-only labels
├── prepare_assistant_checkpoint.py  # Adds reserved dialogue-token rows
├── finetune_assistant.py # Fine-tunes a prepared checkpoint on dialogue JSONL
├── assistant_chat.py     # Stage 4 terminal chat with context-window sliding
├── mps_smoke_test.py    # CPU/MPS forward, backward, and optimizer-step check
├── decoder.py           # Standalone Transformer decoder-block experiment
├── raj_bio.txt          # Small sample biography for text generation
├── raj_chat.txt         # Sample question-and-answer training data
├── test_tiny.py         # Quick structural and tokenizer checks
└── requirements.txt     # Project dependency
```

## Requirements

- Python 3.10 or later (includes `argparse`, `math`, `pathlib`, and other
  standard-library modules used by the project)
- PyTorch 2.x

Install the dependency:

```bash
python3 -m pip install -r requirements.txt
```

Check the installed versions if needed:

```bash
python3 --version
python3 -c "import torch; print(torch.__version__)"
```

## Train on a text file

The included biography is deliberately small, so use it only to check that the
project runs. For better output, train on a clean plain-text file of a few MB.

```bash
python3 char_transformer.py \
  --input raj_bio.txt \
  --steps 1500 \
  --save raj_model.pt
```

During training, the script reports cross-entropy loss for both training and
held-out validation data. Lower validation loss generally means the model is
learning patterns that extend beyond its sampled training batches.

## Generate text from a checkpoint

Checkpoints store the model weights, optimizer state, model configuration, and
tokenizer vocabulary. You do not need to retrain before generating again.

```bash
python3 char_transformer.py \
  --steps 0 \
  --load raj_model.pt \
  --prompt "Raj Pandit" \
  --sample-length 250
```

### Sampling controls

```bash
python3 char_transformer.py \
  --steps 0 \
  --load raj_model.pt \
  --temperature 0.5 \
  --top-k 8
```

- Lower `--temperature` produces safer, more repetitive text.
- Higher `--temperature` produces more varied, less predictable text.
- `--top-k` limits sampling to the most likely tokens; set it to `0` to turn
  this filter off.

## Train and use the terminal chat

> This is the earlier Stage 2 learning demo. It is not instruction-tuned and
> should not be treated as the Stage 4 assistant workflow below.

For conversational responses, the model needs examples in the same format as
the conversation. `raj_chat.txt` teaches a few facts using `User:` and
`Assistant:` pairs.

```bash
python3 char_transformer.py \
  --input raj_chat.txt \
  --steps 3000 \
  --save raj_chat_model.pt \
  --chat
```

Then ask a question such as `What is my name?`. Type `quit` or `exit` to close
the chat.

To reopen the saved chat model later:

```bash
python3 char_transformer.py --steps 0 --load raj_chat_model.pt --chat
```

## Configuration notes

The default tokenizer is BPE with 40 merges and the default context window is
128 tokens. The low number of merges keeps the supplied short files usable.
For larger datasets, experiment with a larger vocabulary and context window:

```bash
python3 char_transformer.py \
  --input my_text.txt \
  --steps 5000 \
  --bpe-merges 500 \
  --context 256 \
  --save my_model.pt
```

To compare raw character tokenization with BPE:

```bash
python3 char_transformer.py --input my_text.txt --tokenizer character --steps 2000
```

## Stage 3: train a small GPT-style model

The `small-gpt` preset increases the context window, embedding size, attention
heads, and number of Transformer blocks. It is intended for a larger local
dataset, not the short example files in this repository.

```bash
python3 char_transformer.py \
  --preset small-gpt \
  --input data/large_text.txt \
  --steps 10000 \
  --save small_gpt.pt
```

The goal is to understand the training process, not to make a smart assistant.
Use a plain UTF-8 text file of at least a few MB. The model learns one task:
predicting the next token from the earlier tokens in its context window.

The preset is deliberately modest so it is still practical to experiment with:

| Setting | Learning preset | `small-gpt` preset |
| --- | ---: | ---: |
| BPE merges | 40 | 300 |
| Context window | 128 tokens | 256 tokens |
| Embedding size | 128 | 192 |
| Attention heads | 4 | 6 |
| Transformer blocks | 3 | 4 |

Every setting can be changed from the command line. For example, use a longer
context window and less frequent evaluation:

```bash
python3 char_transformer.py \
  --preset small-gpt \
  --input data/large_text.txt \
  --context 512 \
  --steps 20000 \
  --eval-interval 500 \
  --save small_gpt_context512.pt
```

If the machine runs out of memory, reduce `--context`, `--batch-size`, or
`--embed-size` before reducing the dataset size.

## Stage 4a: adapt the pretrained model to dialogue

Stage 3 pretraining teaches next-token prediction on general text. Stage 4a
keeps that checkpoint unchanged and creates a separate fine-tuned copy that
learns a small, consistent dialogue format.

### 1. Check MPS before a long run

The trainer now selects CUDA first, then Apple MPS, then CPU. On an Apple
Silicon Mac, verify that the real project model can complete a forward pass,
backward pass, and optimizer step on MPS:

```bash
python3 mps_smoke_test.py --require-mps --checkpoint small_gpt.pt
```

### 2. Create a dialogue-ready checkpoint

This preserves every existing Stage 3 vocabulary row and appends three reserved
tokens: `<|user|>`, `<|assistant|>`, and `<|end|>`. It also creates fresh
fine-tuning optimizer state later; the pretrained optimizer state is not reused
because its vocabulary-shaped tensors are too small.

```bash
python3 prepare_assistant_checkpoint.py \
  --base-checkpoint small_gpt.pt \
  --output checkpoints/small_gpt_assistant_base.pt
```

Never use the Stage 3 checkpoint as the output path. Keeping it untouched makes
it possible to compare pretraining and fine-tuning, or restart a failed run.

### 3. Write dialogue data

Use one JSON object per line. Each dialogue starts with a user turn, alternates
between `user` and `assistant`, and ends with an assistant response. The
included [assistant_examples.jsonl](data/assistant_examples.jsonl) shows both
single-exchange and short two-exchange conversations.

```json
{"messages": [
  {"role": "user", "content": "Explain self attention."},
  {"role": "assistant", "content": "Self attention lets each token use earlier tokens as context."}
]}
```

The trainer formats that example as:

```text
<|user|>Explain self attention.<|end|><|assistant|>Self attention lets each token use earlier tokens as context.<|end|>
```

Only assistant answer tokens and the assistant closing `<|end|>` receive loss.
The model can attend to user tokens, but the training objective does not ask it
to generate the user prompt or the assistant marker.

The Stage 3 BPE tokenizer only recognizes characters seen during pretraining.
The included sample deliberately uses its known character set. If the loader
reports an unknown character, curate or normalize that dialogue text rather
than silently dropping information.

### 4. Fine-tune a separate checkpoint

The defaults are deliberately conservative for an M2 MacBook Air with 8 GB of
unified memory: batch size 4, three epochs, and learning rate `1e-5`.

```bash
python3 finetune_assistant.py \
  --base-checkpoint checkpoints/small_gpt_assistant_base.pt \
  --data data/assistant_examples.jsonl \
  --output checkpoints/small_gpt_assistant_finetuned.pt \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 1e-5
```

The data split is by whole dialogue, never through the middle of a conversation.
Examples longer than the model context window fail loudly instead of having a
response silently removed.

### 5. Talk to the fine-tuned model

```bash
python3 assistant_chat.py \
  --checkpoint checkpoints/small_gpt_assistant_finetuned.pt \
  --temperature 0.5 \
  --top-k 12 \
  --top-p 0.9
```

The chat runner inserts the role tokens itself, keeps the newest complete
user/assistant exchanges that fit in the context window, and stops a response
when the model samples `<|end|>`. It rejects unknown input characters clearly
instead of silently changing the user message. Long-term memory, retrieval, and
tools are intentionally separate later Stage 4 steps.

## Verify the project

Run the quick checks after making changes:

```bash
python3 test_tiny.py
python3 test_stage4.py
```

The test verifies model output shapes, finite loss, generation length, and BPE
encode/decode round-tripping. The Stage 4 test verifies special-token handling,
checkpoint vocabulary expansion, preserved pretrained rows, and assistant-only
loss labels.
