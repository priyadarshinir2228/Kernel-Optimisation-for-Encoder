# Likely judge questions and answers

For: the presenting team. Numbers marked PENDING are filled in from `results_gpu/SUMMARY.md` after the GPU runs.

**Did you change the model?**
No. `team.py` loads the same BAAI/bge-base-en-v1.5 checkpoint and applies the same 12 BERT layers with the same
weights. We rewrote the forward pass by hand only so we could choose kernels (fused QKV, SDPA), skip unused last-layer
rows, and control precision. FP16 is explicitly allowed; INT8 and weight changes are not used.

**Is the CLS-only last layer really exact?**
Yes. In the last layer the CLS output depends on the keys and values of every token, which we still compute for all
tokens. The query, attention output, projection, FFN and LayerNorms are row-wise, so only the CLS row is needed. The
FP32 tests show cosine 1.0000000 with the reference.

**How do you know you never fall below 0.999?**
Four layers of evidence: edge-case tests against `reference.py` (lengths at every bucket boundary, empty, Unicode,
>512 tokens); the official 2,000 passage + 1,500 query references with an internal gate of 0.9995; a full check of
all 100k passages against our FP32 path (which matches the official reference); and a precision study that found the
worst-case inputs and the fix (FP32 residual stream). Hidden data can still differ, which is why the margin matters.

**Why is the worst case a short passage?**
With few tokens the CLS vector is built from little context, so the same absolute rounding error in the residual
stream is a larger relative change in direction. Keeping the residual stream in FP32 removes most of it.

**Why not BF16?**
BF16 has an 8-bit mantissa (FP16 has 11). In our study its worst passage dropped to 0.9918 and 8 of 2,000 passages
were under 0.9995. It would risk a zero score.

**Why not INT8 or a smaller model?** Not allowed by the rules, and not needed.

**Where does the speed-up come from?**
Mostly from not computing padding (5.6x less work, exact count) and from FP16 Tensor Cores; the async pipeline keeps
the GPU busy; the CLS-only last layer removes ~7% more work. Ablation table: PENDING.

**Why not 40x?**
The model's FLOPs are fixed. After removing padding, the remaining work is compute-bound GEMMs on a laptop GPU. The
ceiling is roughly (padding gain) × (FP16 vs FP32 GEMM throughput on this GPU). Measured: PENDING.

**Why is p99 higher than p50?**
`bench.py` sends queries one at a time in arrival order. In a burst of 7 queries within 10 ms the last one waits for
the six before it, so p99 ≈ burst size × service time. We cannot batch across calls, so we minimised service time
with CUDA Graphs. Our latency script reports queueing and service time separately.

**What happens on a GPU you have never seen?**
FP16 needs only CUDA. The token budget halves automatically if warm-up exceeds 75% of VRAM, and any batch that runs
out of memory is split and retried. CUDA Graph capture or `torch.compile` failures fall back to the eager path. No GPU
model, path or environment variable is required.

**Is the 28.4 passages/s baseline fair?**
It is `reference.py` itself on the same GPU over 5,000 passages. Because the reference pads every input to 512 tokens,
its time per passage does not depend on the text, so the sample is representative of the full set.

**Is anything tuned to the practice set?**
No. The scheduler uses only token lengths; the budget is chosen by a sweep with headroom rules; no embeddings or
inputs are cached; the hidden set goes through exactly the same code.

**What would you do with more time?**
Fused Triton kernels for bias+GELU and bias+residual+LayerNorm if profiles show element-wise work is still large;
multi-stream execution; a varlen attention path for flatter length distributions.
