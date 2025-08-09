# Training loop
# train.py
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import NanoKimi
from optimizer import Muon
from utils import get_synthetic_batch, evaluate_perplexity, peak_memory_mb
import math

MODEL_PRESETS = {
    "tiny": dict(d_model=128, n_heads=4, n_layers=4, d_ff=512),
    "small": dict(d_model=256, n_heads=8, n_layers=8, d_ff=1024)
}

def build_model_from_cfg(cfg):
    preset = MODEL_PRESETS.get(cfg.get("model_size","tiny"), MODEL_PRESETS["tiny"])
    moe_params = dict(n_experts=cfg.get("moe_experts",4), k=cfg.get("moe_k",1)) if cfg.get("use_moe") else None
    model = NanoKimi(
        vocab_size=cfg.get("vocab_size",20000),
        seq_len=cfg.get("seq_len",128),
        d_model=preset["d_model"],
        n_heads=preset["n_heads"],
        n_layers=preset["n_layers"],
        d_ff=preset["d_ff"],
        use_moe=cfg.get("use_moe", False),
        moe_params=moe_params,
        use_latent=cfg.get("use_latent_attn", False),
        latent_slots=cfg.get("latent_slots", 8)
    )
    return model

def train_model(cfg, device):
    device = torch.device(cfg["device"])
    model = build_model_from_cfg(cfg).to(device)
    optimizer = Muon(model.parameters(), lr=cfg.get("lr", 2e-4))
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = cfg["batch_size"]
    seq_len = cfg["seq_len"]
    vocab_size = cfg["vocab_size"]
    max_steps = cfg["max_steps"]
    tokens_processed = 0
    start = None
    peak_mem = 0.0
    model.train()
    for step in tqdm(range(max_steps), desc=f"train-{device}"):
        x, y = get_synthetic_batch(batch_size, seq_len, vocab_size, device)
        if start is None:
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tokens_processed += batch_size * seq_len
        if device.type == "cuda":
            cur = torch.cuda.max_memory_allocated(device) / (1024*1024)
            peak_mem = max(peak_mem, cur)
            torch.cuda.reset_peak_memory_stats(device)
    total_time = 1.0  # fallback
    # compute tokens/sec by measuring a small loop
    import time
    t0 = time.time()
    for _ in range(5):
        x, y = get_synthetic_batch(batch_size, seq_len, vocab_size, device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
    t1 = time.time()
    # approximate single-batch forward+loss time
    single_batch_time = (t1 - t0) / 5.0
    tokens_per_sec = (batch_size * seq_len) / single_batch_time
    val_ppl = evaluate_perplexity(model, device, vocab_size=vocab_size, steps=5, batch_size=min(4,batch_size), seq_len=min(128, seq_len))
    metrics = {
        "val_perplexity": float(val_ppl),
        "tokens_per_sec_est": float(tokens_per_sec),
        "peak_mem_MB": float(peak_mem) if device.type == "cuda" else None,
        "params_count": sum(p.numel() for p in model.parameters())
    }
    return metrics
