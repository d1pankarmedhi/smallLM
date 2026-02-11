import argparse
import time

import torch

from smalllm.config import Config
from smalllm.model import LanguageModel
from smalllm.tokenizer import Tokenizer


def benchmark_generate(
    model: LanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    use_kv_cache: bool,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, float]:
    """Run generation and return (output_ids, elapsed_seconds)."""
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.synchronize()

    start = time.perf_counter()
    output = model.generate(
        idx=input_ids.clone(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        use_kv_cache=use_kv_cache,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache generation")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Tokens to generate"
    )
    parser.add_argument(
        "--prompt", type=str, default="Once upon a time", help="Input prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_embd", type=int, default=128, help="Model embedding dim")
    parser.add_argument("--n_layer", type=int, default=8, help="Model layers")
    parser.add_argument("--n_head", type=int, default=8, help="Model attention heads")
    args = parser.parse_args()

    config = Config()
    config.n_embd = args.n_embd
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    
    tokenizer = Tokenizer()
    device = config.device

    print("=" * 60)
    print("  KV Cache Benchmark — smallLM")
    print("=" * 60)
    print(f"  Device          : {device}")
    print(f"  Model params    : n_embd={config.n_embd}, n_head={config.n_head}, "
          f"n_layer={config.n_layer}")
    print(f"  Context size    : {config.context_size}")
    print(f"  Prompt          : {args.prompt!r}")
    print(f"  Max new tokens  : {args.max_new_tokens}")
    print(f"  Temperature     : {args.temperature}")
    print(f"  Seed            : {args.seed}")
    print("=" * 60)

    # Initialise model (random weights — no checkpoint needed)
    model = LanguageModel(config).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Encode prompt
    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded], device=device)
    prompt_len = input_ids.shape[1]

    # --- Run WITHOUT KV cache ---
    print("▶ Generating WITHOUT KV cache ...")
    out_no_cache, time_no_cache = benchmark_generate(
        model, input_ids, args.max_new_tokens, args.temperature,
        use_kv_cache=False, seed=args.seed, device=device,
    )

    # --- Run WITH KV cache ---
    print("▶ Generating WITH KV cache ...")
    out_with_cache, time_with_cache = benchmark_generate(
        model, input_ids, args.max_new_tokens, args.temperature,
        use_kv_cache=True, seed=args.seed, device=device,
    )

    # --- Correctness check ---
    tokens_match = torch.equal(out_no_cache, out_with_cache)

    # --- Results ---
    tps_no_cache = args.max_new_tokens / time_no_cache
    tps_with_cache = args.max_new_tokens / time_with_cache
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float("inf")

    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                      RESULTS                           │")
    print("├──────────────────┬──────────────────┬──────────────────┤")
    print("│                  │  Without Cache   │   With Cache     │")
    print("├──────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Tokens generated │ {args.max_new_tokens:>16} │ {args.max_new_tokens:>16} │")
    print(f"│ Time (seconds)   │ {time_no_cache:>16.4f} │ {time_with_cache:>16.4f} │")
    print(f"│ Tokens/sec       │ {tps_no_cache:>16.2f} │ {tps_with_cache:>16.2f} │")
    print("├──────────────────┴──────────────────┴──────────────────┤")
    print(f"│ Speedup: {speedup:.2f}x                                      │")
    print(f"│ Outputs match: {'✓ YES' if tokens_match else '✗ NO':>8}                                │")
    print("└─────────────────────────────────────────────────────────┘")

    if not tokens_match:
        print("\n⚠ WARNING: Outputs do NOT match! There may be a bug in the KV cache.")
        print(f"  Without cache shape: {out_no_cache.shape}")
        print(f"  With cache shape   : {out_with_cache.shape}")
        # Find first divergence
        min_len = min(out_no_cache.shape[1], out_with_cache.shape[1])
        for i in range(min_len):
            if out_no_cache[0, i] != out_with_cache[0, i]:
                print(f"  First divergence at position {i}: "
                      f"no_cache={out_no_cache[0, i].item()} vs "
                      f"cache={out_with_cache[0, i].item()}")
                break
    else:
        print("\n✓ Correctness verified: both paths produce identical token sequences.")

    # Show a sample of generated text
    print(f"\n--- Sample output (with cache) ---")
    decoded = tokenizer.decode(out_with_cache[0].tolist())
    print(decoded[:500])
    print("---")


if __name__ == "__main__":
    main()
