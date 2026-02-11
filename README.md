<div align="center">
<h1>SmallLM</h1>
<p>A small, GPT-like Language Model with KV Cache and Batched Attention</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

This project provides a minimal, modular, and extensible framework for training and generating text with a transformer-based GPT-like language model. It features a small language model optimized with **KV Caching** for fast inference.

<div align="center">
<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>BPE Tokenization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/271fe3e1-07cb-4cbd-a3ea-1f3d32d7d2d3" width="400" /></td>
      <td><img src="https://github.com/user-attachments/assets/dbcad90f-0d78-4407-a48d-4973027fb9b2" width="550" /></td>
    </tr>
  </tbody>
</table>
</div>

## Key Features

- **Batched Attention:** Optimized multi-head attention that projects and computes all heads in parallel, doubling raw throughput.
- **KV Cache:** Accelerates autoregressive generation by caching and reusing Keys and Values from previous tokens, providing up to **4.5x speedup** on larger model configurations.
- **Detailed Documentation:** Comprehensive guides on [architecture](docs/architecture.md) and [KV cache](docs/kv_cache.md).

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/smallLM.git
   cd smallLM
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Training

```sh
python main.py train
```
The best model checkpoint will be saved to `checkpoints/best_model.pt`.

### 2. Fast Generation (with KV Cache)

```sh
python main.py generate --query "Once upon a time" --max_new_tokens 100 --use_kv_cache
```

- `--use_kv_cache`: Enables the Key-Value cache for faster inference.
- `--max_new_tokens`: Number of tokens to generate.
- `--temperature`: Sampling temperature.

### 3. Benchmarking

Compare performance with and without KV cache:

```sh
# Benchmark default model
python benchmark_kv_cache.py --max_new_tokens 200

# Benchmark a larger model to see scaling benefits
python benchmark_kv_cache.py --max_new_tokens 500 --n_embd 768 --n_layer 12 --n_head 12
```
```
┌────────────────────────────────────────────────────────┐
│                      RESULTS                           │
├──────────────────┬──────────────────┬──────────────────┤
│                  │  Without Cache   │   With Cache     │
├──────────────────┼──────────────────┼──────────────────┤
│ Tokens generated │              500 │              500 │
│ Time (seconds)   │          28.2134 │           6.1797 │
│ Tokens/sec       │            17.72 │            80.91 │
├──────────────────┴──────────────────┴──────────────────┤
│ Speedup: 4.57x                                         │
| GPU: Nvidia GTX 1650                                   |
└────────────────────────────────────────────────────────┘
```
## License

MIT License

## Acknowledgements

Inspired by GPT and nanoGPT projects.
