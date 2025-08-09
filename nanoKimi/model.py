# Transformer + MoE + Latent Attention
# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x): return self.net(x)

class MoE(nn.Module):
    """Simple top-k MoE for small-scale experiments.
       - experts: list of FeedForward modules
       - k: top-k routing (k small like 1 or 2)
    """
    def __init__(self, d_model, d_ff, n_experts=4, k=1, capacity_factor=1.5):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        self.capacity_factor = capacity_factor

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        logits = self.gate(x)  # [B,T,E]
        scores = F.softmax(logits, dim=-1)  # soft allocation
        # top-k
        topk_vals, topk_idx = torch.topk(scores, k=self.k, dim=-1)  # [B,T,k]
        # Simple weighted combination of expert outputs (dense eval of chosen experts)
        out = torch.zeros_like(x)
        for ki in range(self.k):
            idx = topk_idx[..., ki]  # [B,T]
            weight = topk_vals[..., ki].unsqueeze(-1)  # [B,T,1]
            # gather expert outputs for each token by looping (simple but fine for small scale)
            expert_out = torch.zeros_like(x)
            for e in range(self.n_experts):
                mask = (idx == e).float().unsqueeze(-1)  # [B,T,1]
                if mask.sum() == 0:
                    continue
                # compute expert(e) on tokens that belong to it
                # for simplicity, compute expert for all tokens and multiply by mask
                apply = self.experts[e](x) * mask
                expert_out = expert_out + apply
            out = out + expert_out * weight
        return out

class LatentAttention(nn.Module):
    """Simple latent-slot attention:
       - compress tokens into L latent vectors via learned assignments
       - compute cross-attention from tokens to latents and back
    """
    def __init__(self, d_model, latent_slots=8):
        super().__init__()
        self.latent_slots = latent_slots
        self.to_latent_logits = nn.Linear(d_model, latent_slots)
        self.latent_proj = nn.Parameter(torch.randn(latent_slots, d_model))
        # attention components
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        logits = self.to_latent_logits(x)  # [B,T,L]
        weights = F.softmax(logits, dim=-1)  # assign tokens to latents
        # build latent vectors by weighted sum
        latents = torch.einsum('btd, btl -> bld', x, weights)  # [B, L, D]
        # cross attention: tokens attend to latents (token queries, latent keys/values)
        q_t = self.q(x).reshape(B, T, -1)
        k_l = self.k(latents).reshape(B, self.latent_slots, -1)
        v_l = self.v(latents).reshape(B, self.latent_slots, -1)
        attn_logits = torch.matmul(q_t, k_l.transpose(-2, -1)) / self.scale  # [B, T, L]
        attn = torch.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn, v_l)  # [B,T,D]
        return self.out(attended)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, use_moe=False, moe_params=None, use_latent=False, latent_slots=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.use_latent = use_latent
        if use_latent:
            self.latent = LatentAttention(d_model, latent_slots)
        self.use_moe = use_moe
        if use_moe:
            self.moe = MoE(d_model, d_ff, **(moe_params or {}))
        else:
            self.ff = FeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # Self-attention
        res = x
        x, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        x = self.ln1(x + res)
        # FF or MoE and latent
        res = x
        if self.use_latent:
            x = self.latent(x) + x
        if self.use_moe:
            x = self.moe(x)
        else:
            x = self.ff(x)
        x = self.ln2(x + res)
        return x

class NanoKimi(nn.Module):
    def __init__(self, vocab_size=20000, seq_len=128, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, use_moe=False, moe_params=None, use_latent=False, latent_slots=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, use_moe=use_moe, moe_params=moe_params, use_latent=use_latent, latent_slots=latent_slots)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = None  # causal mask omitted for simplicity in training on synthetic data; add for real LM
        for l in self.layers:
            x = l(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
