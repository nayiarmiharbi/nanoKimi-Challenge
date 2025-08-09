# utils.py
import torch
import numpy as np
import time
import psutil
import math

def get_synthetic_batch(batch_size, seq_len, vocab_size, device):
    data = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    target = data.clone()
    return data, target

def evaluate_perplexity(model, device, vocab_size=20000, steps=10, batch_size=4, seq_len=128):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_synthetic_batch(batch_size, seq_len, vocab_size, device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    ppl = math.exp(total_loss / steps)
    model.train()
    return ppl

def now():
    return time.time()

def peak_memory_mb(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024*1024)
    else:
        return psutil.Process().memory_info().rss / (1024*1024)
