# ResMapShelfLM: Two-Stage Uncompressed Shelf/Chunk Retrieval Language Model

`ResMapShelfLM` is a next-generation hybrid language model architecture designed to bridge the gap between local sliding-window transformers and infinite-context associative recall.

---

## Key Architectural Principles

1. **Two-Stage Uncompressed Retrieval**:
   - **Stage 1 (Coarse Routing)**: Past key-value tokens are organized into discrete "shelves" or chunks ($C = 64$ tokens). Each shelf is summarized into a dual-pooled ($\text{mean} + \text{max}$) landmark vector and projected via a learnable holographic semantic lens ($\text{map\_dim} = 32$). Top-$K$ relevant chunks ($K=4$) are selected via coarse dot-product routing with strict causal masking.
   - **Stage 2 (Fine Attention)**: Instead of computing attention against lossy, blurred chunk embeddings, the model gathers the **uncompressed, raw keys and values** of the selected chunks and computes exact dot-product attention.
2. **Local Sliding Window**:
   - Local context within $W = 256$ tokens is processed with standard full-resolution attention and 2D Rotary Position Embeddings (RoPE).
3. **Adaptive Gated Fusion**:
   - A per-head learnable sigmoid gate dynamically arbitrates between local syntactical dependencies and long-distance uncompressed shelf memory.
4. **Memory-Efficient Chunked Cross-Entropy**:
   - Loss is calculated in token slices over the 32,000 BPE vocabulary, avoiding large logit allocation in VRAM.

---

## Directory Structure

- `model.py`: Standalone PyTorch implementation containing `ResMapShelfConfig`, `ResMapShelfAttention`, `ResMapShelfBlock`, `ResMapShelfLM`, and autoregressive sampling (`generate`).
- `train_and_predict.py`: Unified command-line interface for training on `input.txt` and generating text predictions.

---

## Usage

### 1. Training over `input.txt`

Train on the corpus (auto-detects GPU and pre-tokenized binary `input_bpe_32k.bin`):

```bash
python train_and_predict.py --mode train --steps 1000 --batch_size 8 --seq_len 512 --lr 5e-4
```

### 2. Text Prediction / Generation from Checkpoint

Generate text completions given a custom prompt:

```bash
python train_and_predict.py --mode predict --checkpoint resmap_shelf_checkpoint.pt --prompt "In the beginning" --max_new_tokens 150 --temperature 0.8
```

### 3. End-to-End (Train & Predict in One Run)

```bash
python train_and_predict.py --mode both --steps 500 --prompt "And it came to pass"
```
