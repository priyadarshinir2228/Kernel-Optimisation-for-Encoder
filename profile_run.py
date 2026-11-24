"""Profiler evidence for the report (25% of the score). Profiles reference.py or team.py on a sample of passages
and writes: a Chrome trace (open in chrome://tracing or https://ui.perfetto.dev), the top-kernel table, and a
GPU-time breakdown by category (GEMM / attention / normalisation+element-wise / memcpy / other).

Usage:
  python profile_run.py --which reference --data DATA_DIR --n 2000 --out results/prof_reference
  PS09_STAGE=4 python profile_run.py --which team --data DATA_DIR --n 2000 --out results/prof_stage4
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CATS = [("gemm", ("gemm", "cutlass", "sm80_xmma", "sm90_xmma", "ampere_", "turing_", "volta_", "cublas", "matmul")),
        ("attention", ("flash", "fmha", "efficient_attention", "attention", "softmax", "bmm")),
        ("norm+elementwise", ("layer_norm", "layernorm", "gelu", "elementwise", "vectorized", "add", "mul", "copy_kernel",
                              "fill", "index", "normalize", "triton")),
        ("memcpy", ("memcpy", "memset"))]


def category(name):
    n = name.lower()
    for cat, keys in CATS:
        if any(k in n for k in keys):
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["reference", "team"], required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", default=os.environ.get("PS09_MODEL", "BAAI/bge-base-en-v1.5"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    texts = [json.loads(l)["text"] for l in open(Path(a.data) / "practice_passages.jsonl", encoding="utf-8")][: a.n]

    if a.which == "reference":
        sys.path.insert(0, str(Path(a.data)))
        import reference
        tok, m = reference.load(a.model)
        run = lambda xs: reference.encode(tok, m, xs)
        tokenise = lambda xs: [tok(xs[i:i + 32], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                               for i in range(0, len(xs), 32)]
    else:
        import team
        team.setup()
        run = team.encode
        tokenise = team._tokenise
    run(texts[:64])  # warm-up outside the trace
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); tokenise(texts); tok_s = time.perf_counter() - t   # CPU tokenisation alone

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    t = time.perf_counter()
    with profile(activities=acts, record_shapes=True) as prof:
        with record_function(f"encode_{a.which}"):
            run(texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    wall = time.perf_counter() - t

    prof.export_chrome_trace(os.path.join(a.out, "trace.json"))
    import gzip, shutil  # compressed copy small enough to keep as evidence (open it in https://ui.perfetto.dev)
    with open(os.path.join(a.out, "trace.json"), "rb") as fi, gzip.open(os.path.join(a.out, "trace.json.gz"), "wb") as fo:
        shutil.copyfileobj(fi, fo)
    try:
        table = prof.key_averages().table(sort_by="device_time_total" if torch.cuda.is_available() else "cpu_time_total", row_limit=30)
    except Exception:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    open(os.path.join(a.out, "top_ops.txt"), "w").write(table)

    # GPU kernel time by category (device events only) + timeline of device activity
    cats, total, spans, n_kernels = {}, 0.0, [], 0
    for e in prof.events():
        if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA"):
            dt = getattr(e, "device_time_total", None)
            dt = e.cuda_time_total if dt is None else dt
            c = category(e.name)
            cats[c] = cats.get(c, 0.0) + dt
            total += dt
            n_kernels += 1
            tr = getattr(e, "time_range", None)
            if tr is not None and tr.end > tr.start:
                spans.append((tr.start, tr.end))
    timeline = {}
    if spans:  # union of busy intervals -> busy time and idle gaps between device activity (microseconds)
        spans.sort()
        busy, gaps, (s0, e0) = 0.0, [], spans[0]
        for s1, e1 in spans[1:]:
            if s1 > e0:
                busy += e0 - s0; gaps.append(s1 - e0); s0, e0 = s1, e1
            else:
                e0 = max(e0, e1)
        busy += e0 - s0
        g = np.array(gaps) if gaps else np.zeros(1)
        span = spans[-1][1] - spans[0][0]
        timeline = {"device_span_ms": span / 1000, "device_busy_ms": busy / 1000, "device_busy_fraction_of_span": busy / span,
                    "idle_gaps": int(len(gaps)), "idle_total_ms": float(g.sum() / 1000), "idle_gaps_over_100us": int((g > 100).sum()),
                    "largest_gaps_ms": [round(float(x) / 1000, 3) for x in np.sort(g)[::-1][:5]],
                    "kernels_per_passage": n_kernels / len(texts)}
    summary = {"which": a.which, "n": len(texts), "wall_s": wall, "passages_per_s": len(texts) / wall,
               "gpu_kernel_ms": total / 1000, "gpu_busy_fraction": (total / 1e6) / wall if wall else None,
               "gpu_time_share": {k: v / total for k, v in sorted(cats.items(), key=lambda x: -x[1])} if total else {},
               "gpu_ms_by_category": {k: v / 1000 for k, v in sorted(cats.items(), key=lambda x: -x[1])},
               "timeline": timeline, "tokenise_only_s": tok_s, "tokenise_share_of_wall": tok_s / wall,
               "note": "wall_s is measured under the profiler (adds overhead); use bench.py for throughput claims"}
    if torch.cuda.is_available():
        summary["peak_gpu_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        summary["gpu"] = torch.cuda.get_device_name(0)
    if a.which == "team":
        import team
        summary.update(stage=team.STAGE, cfg=team.CFG, token_budget=team.TOKEN_BUDGET, runtime=team.runtime_info())
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(table[:3000])


if __name__ == "__main__":
    main()
