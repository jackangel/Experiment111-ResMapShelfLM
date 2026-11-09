"""
ResMapShelfLM: Training & Prediction Engine
===========================================
Architecture: Two-Stage Uncompressed Shelf/Chunk Retrieval Language Model
Trained and evaluated on input.txt using Byte-Level BPE (32K vocabulary).
"""

import argparse
import math
import os
import time
from typing import Tuple

import numpy as np
import torch
from tokenizers import Tokenizer

from model import ResMapShelfConfig, ResMapShelfLM


# -----------------------------------------------------------------------------
# Data Loading & Tokenization Utilities
# -----------------------------------------------------------------------------

def resolve_file_path(target_name: str, fallback_paths: list[str]) -> str:
    """Finds an existing file from a list of potential relative or absolute paths."""
    for p in fallback_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return target_name


def prepare_dataset(input_txt_path: str, tokenizer_path: str, bin_output_path: str = "dataset_bpe_32k.bin") -> np.memmap:
    """
    Loads or creates a memory-mapped binary dataset of token IDs (uint16).
    If the pre-tokenized binary exists, loads it instantly via zero-copy memmap.
    Otherwise tokenizes input.txt in streaming chunks and persists the binary.
    """
    # Check if a pre-existing bin is already present
    potential_bins = [
        bin_output_path,
        os.path.join(os.path.dirname(input_txt_path), "input_bpe_32k.bin"),
        "../input_bpe_32k.bin",
        "input_bpe_32k.bin",
    ]
    for b in potential_bins:
        if os.path.exists(b) and os.path.getsize(b) > 1024:
            print(f"[Dataset] Found existing pre-tokenized binary: {os.path.abspath(b)} ({os.path.getsize(b):,} bytes)")
            return np.memmap(b, dtype=np.uint16, mode="r")

    print(f"[Dataset] Binary dataset not found. Tokenizing '{input_txt_path}' using '{tokenizer_path}'...")
    tok = Tokenizer.from_file(tokenizer_path)

    all_ids = []
    chunk_lines = []
    chunk_size_bytes = 10 * 1024 * 1024  # 10 MB text chunks

    with open(input_txt_path, "r", encoding="utf-8", errors="replace") as f:
        accumulated = 0
        for line in f:
            chunk_lines.append(line)
            accumulated += len(line)
            if accumulated >= chunk_size_bytes:
                text_chunk = "".join(chunk_lines)
                enc = tok.encode(text_chunk)
                all_ids.extend(enc.ids)
                chunk_lines = []
                accumulated = 0
                print(f"[Dataset] Tokenized {len(all_ids):,} tokens so far...", end="\r")

        if chunk_lines:
            text_chunk = "".join(chunk_lines)
            enc = tok.encode(text_chunk)
            all_ids.extend(enc.ids)

    print(f"\n[Dataset] Total tokens extracted: {len(all_ids):,}. Writing to {bin_output_path}...")
    np_arr = np.array(all_ids, dtype=np.uint16)
    np_arr.tofile(bin_output_path)
    return np.memmap(bin_output_path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts a random batch of contiguous sequences (x, y) from the memory-mapped dataset."""
    max_idx = len(data) - seq_len - 1
    start_indices = np.random.randint(0, max_idx, size=batch_size)
    x_stack = [data[i : i + seq_len].astype(np.int64) for i in start_indices]
    y_stack = [data[i + 1 : i + seq_len + 1].astype(np.int64) for i in start_indices]

    x = torch.tensor(np.stack(x_stack), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_stack), dtype=torch.long, device=device)
    return x, y


# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# -----------------------------------------------------------------------------
# Training Pipeline
# -----------------------------------------------------------------------------

def train(args, model: ResMapShelfLM, dataset: np.memmap, device: torch.device):
    print("=" * 80)
    print(f"Starting Training: ResMapShelfLM on {args.input}")
    print(f"Device: {device} | Total Steps: {args.steps} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 80)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    # Autocast precision setup
    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    t0 = time.time()
    tokens_processed = 0

    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.steps, args.warmup_steps, args.lr, args.lr * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(dataset, args.batch_size, args.seq_len, device)

        optimizer.zero_grad(set_to_none=True)
        if use_cuda:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, targets=y)
        else:
            _, loss = model(x, targets=y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_processed += args.batch_size * args.seq_len

        # Periodic logging
        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / max(1e-5, elapsed)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if use_cuda else 0.0
            print(f"[Step {step:5d}/{args.steps:5d}] Loss: {loss.item():.4f} | LR: {lr:.2e} | "
                  f"Throughput: {tok_per_sec:,.0f} tok/s | VRAM: {vram_mb:.1f} MB")

        # Periodic checkpointing
        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(model, args.checkpoint, step)

    print(f"[Training Complete] Final Loss: {loss.item():.4f} | Saved to: {os.path.abspath(args.checkpoint)}")


def save_checkpoint(model: ResMapShelfLM, checkpoint_path: str, step: int):
    torch.save({
        "step": step,
        "config": model.config,
        "model_state_dict": model.state_dict(),
    }, checkpoint_path)
    print(f"--> Checkpoint saved to '{os.path.abspath(checkpoint_path)}' (step {step})")


def load_checkpoint(model: ResMapShelfLM, checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Checkpoint] Loaded weights from '{os.path.abspath(checkpoint_path)}' (step {ckpt.get('step', 'unknown')})")


# -----------------------------------------------------------------------------
# Text Prediction / Generation Pipeline
# -----------------------------------------------------------------------------

def predict(args, model: ResMapShelfLM, tokenizer_path: str, device: torch.device):
    print("=" * 80)
    print(f"Text Prediction / Generation with ResMapShelfLM")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Settings: max_new_tokens={args.max_new_tokens}, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 80)

    tok = Tokenizer.from_file(tokenizer_path)
    enc = tok.encode(args.prompt)
    prompt_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

    t0 = time.time()
    out_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    gen_time = time.time() - t0

    generated_text = tok.decode(out_ids[0].tolist())
    new_tokens_count = out_ids.size(1) - prompt_ids.size(1)
    gen_speed = new_tokens_count / max(1e-5, gen_time)

    print("\n[Generated Output]:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    print(f"[Stats]: {new_tokens_count} tokens generated in {gen_time:.2f}s ({gen_speed:.1f} tok/s)\n")


# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ResMapShelfLM: Train & Predict over input.txt")
    
    # Operation Mode
    parser.add_argument("--mode", type=str, default="both", choices=["train", "predict", "both"],
                        help="Mode: 'train', 'predict', or 'both'")
    
    # Paths
    parser.add_argument("--input", type=str, default="../input.txt", help="Path to input.txt corpus")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer_32k.json", help="Path to tokenizer_32k.json")
    parser.add_argument("--checkpoint", type=str, default="resmap_shelf_checkpoint.pt", help="Path to checkpoint file")
    
    # Model Hyperparameters
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding & hidden dimension")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--window", type=int, default=256, help="Local sliding window size W")
    parser.add_argument("--chunk_size", type=int, default=64, help="Shelf/chunk size C")
    parser.add_argument("--top_k_chunks", type=int, default=4, help="Top-K past chunks to retrieve")
    parser.add_argument("--map_dim", type=int, default=32, help="Holographic routing projection dimension")
    
    # Training Hyperparameters
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (multiple of chunk_size)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation and logging interval")
    parser.add_argument("--save_interval", type=int, default=250, help="Checkpoint saving interval")
    
    # Prediction Hyperparameters
    parser.add_argument("--prompt", type=str, default="In the beginning God created the heaven and the earth.",
                        help="Input text prompt for prediction")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve file paths
    input_path = resolve_file_path(args.input, [args.input, "../input.txt", "input.txt"])
    tok_path = resolve_file_path(args.tokenizer, [args.tokenizer, "../tokenizer_32k.json", "tokenizer_32k.json"])

    # Instantiate model
    config = ResMapShelfConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        window=args.window,
        chunk_size=args.chunk_size,
        top_k_chunks=args.top_k_chunks,
        map_dim=args.map_dim,
    )
    model = ResMapShelfLM(config).to(device)

    # Execute selected mode
    if args.mode in ("train", "both"):
        dataset = prepare_dataset(input_path, tok_path)
        train(args, model, dataset, device)

    if args.mode in ("predict", "both"):
        if args.mode == "predict":
            load_checkpoint(model, args.checkpoint, device)
        predict(args, model, tok_path, device)


if __name__ == "__main__":
    main()
