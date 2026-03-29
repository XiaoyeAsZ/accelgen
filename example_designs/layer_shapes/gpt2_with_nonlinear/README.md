# GPT-2 with Nonlinear Proxy Layers

This directory extends the original GPT-2 layer shapes with **proxy workloads
for nonlinear operations** (GELU activation and Softmax).

## GPT-2 Transformer Block Structure

Each of the 24 Transformer blocks contains these operations in order:

| Step | Operation            | Type       | Layer file pattern     |
|------|----------------------|------------|------------------------|
| 0    | QKV projection       | Linear     | `C=1024,M=3072,P=256`  |
| 1    | Attn output project  | Linear     | `C=1024,M=1024,P=256`  |
| 2    | Attn QK^T (per head) | Linear     | `M=64,C=256,P=16`      |
| 2.5  | **Softmax**          | Nonlinear  | proxy layer            |
| 3    | Attn score × V       | Linear     | `M=256,C=256,P=16`     |
| 4    | FFN expand           | Linear     | `C=1024,M=4096,P=256`  |
| 4.5  | **GELU**             | Nonlinear  | proxy layer            |
| 5    | FFN contract         | Linear     | `C=4096,M=1024,P=256`  |

## Nonlinear Proxy Modeling

Since Timeloop's problem template only supports Conv/GEMM-style workloads,
nonlinear ops are approximated as **pointwise 1×1 convolutions**:

### GELU (polynomial proxy)
- GELU ≈ cubic polynomial: `0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x³)))`
- Approximated as **4-term polynomial** per element
- Shape: `C=4, M=1, P=1048576` (=4096×256, the FFN hidden dim × seq_len)
- This means: 4 multiply-accumulate ops per activation element

### Softmax (multi-step proxy)
- Softmax = exp(x) / Σexp(x), decomposed into:
  1. **exp proxy**: element-wise polynomial approx → `C=4, M=1, P=4096` (=256×16 per head)
  2. **reduce-sum proxy**: reduction → `C=4096, M=1, P=1`  (sum over seq dim)
  3. **div/normalize proxy**: element-wise → `C=1, M=1, P=4096`
- Per-head softmax applied to the 256×16 attention score matrix (16 heads)

## How to run

```bash
cd workspace/example_designs
python3 run_example_designs.py --architecture simba_like --problem gpt2_with_nonlinear
```
