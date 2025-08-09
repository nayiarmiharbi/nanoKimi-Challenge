# Runs CPU + GPU benchmarks, saves reports
# benchmark.py
import argparse
import json
import os
from train import train_model

def run_benchmark(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"Running benchmark with config: {cfg_path} on device {cfg.get('device')}")
    metrics = train_model(cfg, cfg.get('device'))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fw:
        json.dump(metrics, fw, indent=2)
    print(f"Saved report to {out_path}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_benchmark(args.config, args.out)
