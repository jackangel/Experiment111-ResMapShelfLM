import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResMapShelfConfig:
    """
    Configuration for ResMapShelfLM (Two-Stage Uncompressed Shelf/Chunk Retrieval).
    """
    vocab_size: int = 32000
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 8
    window: int = 256           # Local sliding window size W
    chunk_size: int = 64        # Shelf/chunk size C
    top_k_chunks: int = 4       # Number of past chunks K to retrieve
    map_dim: int = 32           # Dimension of holographic semantic landmark projection
    max_len: int = 4096         # Maximum sequence length supported by RoPE cache
    dropout: float = 0.0


# -----------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# -----------------------------------------------------------------------------

def build_rope_cache(max_len: int, dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary cosine and sine tables."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies 2D rotary position embeddings to query or key tensor [B, H, T, hd]."""
    T = x.shape[2]
    c = cos[:T].view(1, 1, T, -1)
    s = sin[:T].view(1, 1, T, -1)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * c + rot * s


# -----------------------------------------------------------------------------
# Two-Stage Uncompressed Shelf Attention
# -----------------------------------------------------------------------------

class ResMapShelfAttention(nn.Module):
    """
    Two-Stage Shelf/Chunk Retrieval Attention:
      1. Local Attention: Strict sliding window (W=256) with RoPE.
      2. Coarse Routing: Dual-pooled (mean + max) chunk landmarks projected into
         semantic space (map_dim=32). Strictly causal: chunk i cannot retrieve chunk >= i.
      3. Fine Attention: Dot-product attention over uncompressed, raw keys and values
         of the selected top-K chunks without any lossy averaging.
      4. Gated Fusion: Per-head sigmoid gate dynamically balancing local sliding window
         and global uncompressed shelf retrieval.
    """
    def __init__(self, config: ResMapShelfConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.hd = config.d_model // config.n_head
        self.window = config.window
        self.chunk_size = config.chunk_size
        self.top_k = config.top_k_chunks
        self.map_dim = config.map_dim

        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

        # Per-head adaptive gate balancing local vs global context
        self.gate = nn.Linear(config.d_model, config.n_head, bias=True)
        nn.init.constant_(self.gate.bias, 0.0)

        # Learnable semantic lenses for coarse chunk routing
        self.lens_q = nn.Linear(2 * self.hd, config.map_dim, bias=False)
        self.lens_k = nn.Linear(2 * self.hd, config.map_dim, bias=False)
        nn.init.normal_(self.lens_q.weight, std=0.02)
        nn.init.normal_(self.lens_k.weight, std=0.02)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H, hd = self.n_head, self.hd
        C = self.chunk_size
        NC = T // C

        # Linear projections
        q_raw = self.q(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
        k_raw = self.k(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]
        v = self.v(x).view(B, T, H, hd).transpose(1, 2)      # [B, H, T, hd]

        # 1. Local Sliding-Window Attention with RoPE
        q_rope = apply_rope(q_raw, cos, sin)
        k_rope = apply_rope(k_raw, cos, sin)

        t_idx = torch.arange(T, device=x.device)
        win_mask = (t_idx[:, None] >= t_idx[None, :]) & ((t_idx[:, None] - t_idx[None, :]) < self.window)
        out_local = F.scaled_dot_product_attention(q_rope, k_rope, v, attn_mask=win_mask[None, None])

        # If sequence is shorter than 2 chunks, global retrieval has no past full chunks
        if NC <= 1:
            return self.out(out_local.transpose(1, 2).reshape(B, T, d))

        # 2. Coarse Chunk Routing
        # Reshape into chunks of size C
        k_c = k_raw[:, :, :NC * C].view(B, H, NC, C, hd)
        v_c = v[:, :, :NC * C].view(B, H, NC, C, hd)
        q_c = q_raw[:, :, :NC * C].view(B, H, NC, C, hd)

        # Landmark features: dual pooling (mean + max) preserves both background context and sharp salient tokens
        k_summary = torch.cat([k_c.mean(dim=3), k_c.max(dim=3)[0]], dim=-1)  # [B, H, NC, 2*hd]
        q_summary = torch.cat([q_c.mean(dim=3), q_c.max(dim=3)[0]], dim=-1)

        k_lm = self.lens_k(k_summary)  # [B, H, NC, map_dim]
        q_lm = self.lens_q(q_summary)  # [B, H, NC, map_dim]

        coarse_logits = torch.matmul(q_lm, k_lm.transpose(-1, -2)) / math.sqrt(self.map_dim)  # [B, H, NC, NC]

        # Causal mask: chunk i can only retrieve chunks < i
        causal_chunk_mask = torch.tril(torch.ones(NC, NC, device=x.device, dtype=torch.bool), diagonal=-1)
        coarse_logits = coarse_logits.masked_fill(~causal_chunk_mask.view(1, 1, NC, NC), -1e9)

        n_sel = min(self.top_k, NC)
        top_scores, top_chunks = coarse_logits.topk(n_sel, dim=-1)  # [B, H, NC, n_sel]

        # 3. Fine Attention Over Uncompressed Raw Keys and Values of Selected Chunks
        c_offsets = torch.arange(C, device=x.device).view(1, 1, 1, 1, C)
        tok_idx = (top_chunks.unsqueeze(-1) * C + c_offsets).view(B, H, NC, n_sel * C)

        k_flat = k_raw[:, :, :NC * C]
        v_flat = v[:, :, :NC * C]
        idx_exp = tok_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hd)

        k_sel = torch.gather(k_flat.unsqueeze(2).expand(-1, -1, NC, -1, -1), 3, idx_exp)  # [B, H, NC, n_sel*C, hd]
        v_sel = torch.gather(v_flat.unsqueeze(2).expand(-1, -1, NC, -1, -1), 3, idx_exp)  # [B, H, NC, n_sel*C, hd]

        # Exact uncompressed dot-product
        fine_logits = torch.matmul(q_c, k_sel.transpose(-1, -2)) / math.sqrt(hd)  # [B, H, NC, C, n_sel*C]

        # Differentiable boost linking coarse routing scores into fine attention
        coarse_boost = top_scores.unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape(B, H, NC, 1, n_sel * C)
        fine_logits = fine_logits + 0.2 * coarse_boost

        # Mask invalid chunks (chunks with top_score <= -1e5, such as chunk 0 which has no past chunks)
        valid_mask = (top_scores > -1e5).unsqueeze(-1).expand(-1, -1, -1, -1, C).reshape(B, H, NC, 1, n_sel * C)
        fine_logits = fine_logits.masked_fill(~valid_mask, float("-inf"))

        fine_attn = F.softmax(fine_logits, dim=-1)
        fine_attn = torch.nan_to_num(fine_attn, 0.0)
        out_global_c = torch.matmul(fine_attn, v_sel)  # [B, H, NC, C, hd]
        out_global = out_global_c.view(B, H, NC * C, hd)

        if T > NC * C:
            out_global = F.pad(out_global, (0, 0, 0, T - NC * C))

        # 4. Gated Fusion
        g = torch.sigmoid(self.gate(x)).transpose(1, 2).unsqueeze(-1)  # [B, H, T, 1]
        out_combined = (1.0 - g) * out_local + g * out_global
        return self.out(out_combined.transpose(1, 2).reshape(B, T, d))


# -----------------------------------------------------------------------------
# Transformer Block & Full Model
# -----------------------------------------------------------------------------

class ResMapShelfBlock(nn.Module):
    def __init__(self, config: ResMapShelfConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = ResMapShelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        mlp_dim = 4 * config.d_model
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, config.d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


def chunked_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor,
                           targets: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """
    Computes cross-entropy loss across token slices without ever allocating the full
    [N, vocab_size] logits matrix in VRAM.
    """
    N = hidden.size(0)
    total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
    for i in range(0, N, chunk_size):
        h_c = hidden[i : i + chunk_size]
        t_c = targets[i : i + chunk_size]
        logits_c = F.linear(h_c, weight)
        total_loss = total_loss + F.cross_entropy(logits_c, t_c, reduction="sum")
    return total_loss / N


class ResMapShelfLM(nn.Module):
    """
    ResMapShelfLM: Next-generation ultra-efficient hybrid language model.
    Combines local sliding-window attention with two-stage uncompressed shelf retrieval.
    """
    def __init__(self, config: Optional[ResMapShelfConfig] = None):
        super().__init__()
        self.config = config or ResMapShelfConfig()
        
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList([
            ResMapShelfBlock(self.config) for _ in range(self.config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(self.config.d_model)
        
        # Weight tying between token embeddings and LM head
        self.head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # Precompute RoPE tables
        cos, sin = build_rope_cache(self.config.max_len, self.config.d_model // self.config.n_head, torch.device("cpu"))
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                chunk_size_ce: int = 2048) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass. If targets are supplied, returns (logits, loss) using memory-efficient
        chunked cross-entropy. Otherwise returns (logits, None).
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.ln_f(x)

        if targets is not None:
            # Memory-efficient chunked cross entropy
            flat_x = x.view(-1, self.config.d_model)
            flat_targets = targets.view(-1)
            loss = chunked_cross_entropy(flat_x, self.head.weight, flat_targets, chunk_size=chunk_size_ce)
            return None, loss
        else:
            logits = self.head(x)
            return logits, None

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        """
        Autoregressively generates new tokens given a prompt tensor [1, T].
        """
        self.eval()
        curr_tokens = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            # Crop to max_len if needed
            cond = curr_tokens if curr_tokens.size(1) <= self.config.max_len else curr_tokens[:, -self.config.max_len:]
            logits, _ = self(cond)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Top-P (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        return curr_tokens
