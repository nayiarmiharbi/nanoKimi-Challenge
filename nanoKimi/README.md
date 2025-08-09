# nanoKimi

A small transformer + MoE model with benchmarking tools.

Intall dependencies
```
pip install -r requirements.txt
```
To run locally with ***no GPU**
```
python benchmark.py --config configs/cpu_small.json --out results/cpu_report.json
```
To run in Google Colab with ***T4 GPU**
```
python benchmark.py --config configs/t4_gpu.json --out results/t4_report.json
```
