# Environment
- Always activate the micromamba environment before running any commands: `micromamba activate scale`

# SATA Attention - Symmetry-Aware Taylor Approximated Attention

## Overview
A novel attention mechanism that computes scaled dot-product attention at **constant cost per token** via Taylor expansion of the exponential function combined with symmetric tensor decomposition. Achieves orders-of-magnitude reductions in both memory and computation vs. conventional attention. Proof-of-concept implementation. Paper by Heinsen & Kozachkov, 2026.

## Tech Stack
- **Framework**: PyTorch (torch.nn.Module)
- **Benchmarking**: torch.utils.benchmark, torch.cuda memory APIs
- **Compilation**: torch.compile (dynamic=True)
- **Precision**: Float16/BFloat16/Float32/Float64 support
- **Visualization**: Matplotlib (with LaTeX rendering)
- **License**: MIT (GlassRoom, 2026)

## Project Structure
```
sata_attention.py          # Core implementation (147 lines, 3 classes, 1 helper)
replicate_results.py       # Comprehensive benchmarking suite (562 lines)
replication_deps.txt       # Benchmark dependencies (tqdm, numpy, pandas, matplotlib)
README.md                  # Usage guide with mathematical derivations
images/
  symmetric_tensors.png    # Visualization of symmetric tensor concepts
```

## Core Algorithm

### Mathematical Foundation
```
exp((q*k)/c) = Sum_p alpha_p * (q*k)^p           [Taylor series]
             = Sum_p alpha_p * Sum Phi(q) * Phi(k)  [Monomial decomposition]
```
Where alpha_p = 1/(p! * c^p), c = sqrt(d_key), and Phi() extracts the upper hyper-triangular region of the symmetric tensor product.

### Key Classes (`sata_attention.py`)

**`_calculate_n_idx_permutations(M)`**
- Calculates multiset permutation counts for index matrix M
- Uses lgamma for numerical stability
- Supports vmap for batch processing

**`TightlyPackedTaylorTerm(nn.Module)`**
- Implements a single p-th order Taylor term
- `Phi(x)`: Feature map that tightly packs monomials using `combinations_with_replacement` indices
- `forward(Q, K, V)`: Computes attention contribution for one term
- Handles p=0 (constant) specially
- Supports causal masking via cumulative sums
- Maintains hidden state (prev_H_S, prev_H_Z) for streaming/continuous generation

**`SymmetryAwareTaylorApproximatedAttention(nn.Module)`**
- Main user-facing class, composes multiple TightlyPackedTaylorTerm modules
- `forward(Q, K, V, continue_prev=False)`: Full attention computation
- Sums all Taylor term outputs and normalizes
- `get_forward_FLOPs_per_query_head()`: Analytical FLOP calculation
- `get_hidden_state_sizes()`: Memory footprint analysis

## Usage

### Basic
```python
from sata_attention import SymmetryAwareTaylorApproximatedAttention

attn = SymmetryAwareTaylorApproximatedAttention(d_key=64, d_val=64, is_causal=True, n_taylor=4)
Y = attn(Q, K, V)  # Q,K: [n_tok, d_key], V: [n_tok, d_val]
```

### Streaming (constant cost per token)
```python
Y = attn(new_Q, new_K, new_V, continue_prev=True)
```

### Billion-token scale
```python
attn = SymmetryAwareTaylorApproximatedAttention(d_key=8, d_val=8, is_causal=True, n_taylor=8)
for tok_num in range(1_000_000_000):
    q, k, v = torch.randn(3, 1024, 1, 8)
    y = attn(q, k, v, continue_prev=tok_num > 0)
```

## Key Parameters
- `d_key`: Query/key dimension
- `d_val`: Value dimension
- `is_causal`: Enable causal (autoregressive) masking
- `n_taylor`: Number of Taylor terms (default 4, up to 8 tested)

## Benchmarking (`replicate_results.py`)
Benchmarks compare SATA vs. conventional attention across:
- Hidden state memory requirements
- FLOPs per token (analytical)
- Actual GPU memory allocation
- Execution time per token
- Numerical reconstruction error

Run: `python replicate_results.py`
Outputs: `benchmark_data.pt` + `fig_*.png` figures (300 DPI)

## Known Limitations (proof-of-concept)
1. `x[..., M]` returns copies, not views (memory bandwidth bottleneck)
2. No hierarchical optimization of index matrix structure
3. Taylor terms evaluated sequentially on GPU (could be parallelized)
4. No fused CUDA kernels
5. No basis compression or random projection approximations

## Dependencies
- **Core**: torch (only hard requirement)
- **Benchmarking**: tqdm, numpy, pandas, matplotlib
