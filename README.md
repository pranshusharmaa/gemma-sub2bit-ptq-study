# Gemma Sub-2-bit PTQ Study

In-progress research prototype studying sub-2-bit post-training quantization on Gemma-3-270M.

This project tests how far a small language model can be pushed below 4-bit weight quantization before perplexity collapses. The notebook compares simple RTN baselines, simplified paper-inspired PTQ methods, and an experimental mixed-precision ternary stack with saliency rescue.

## Status

This is a research-progress notebook, not a finished compressor.

I first developed and checked the evaluation pipeline locally, including baseline perplexity/accuracy evaluation. Local runs worked, but repeated sweeps and ablations were slow, so I used remaining Google Colab credits from a course to fast-track the heavier experiment runs.

Current status:
- evaluation pipeline working
- bf16, int8, int4, ternary, and binary baselines tested
- component sensitivity sweep added
- simplified paper-inspired PTQ methods tested
- mixed-precision saliency-rescue stack tested
- simplified GPTQ / Hadamard sanity baselines added
- real low-bit kernels, full GPTQ, and proper rotation folding are future work

## Main finding so far

Naive sub-2-bit PTQ is too fragile on Gemma-3-270M.

Plain ternary and binary quantization collapse badly under post-training fake quantization. Int8 remains close to the bf16 baseline, and int4 is degraded but still much stronger than the sub-2-bit runs.

The most promising direction so far is mixed precision:

> ternary for most weights, plus saliency-based INT4 rescue for a small fraction of important weights.

The best current sub-2-bit experiment uses saliency rescue, but it is still far worse than int4. So this project should be read as a failure-mode study and early research prototype, not as a solved compression method.

## What the notebook does

The notebook runs:

- bf16 baseline
- int8 RTN
- int4 RTN
- 1.58-bit ternary RTN
- 1-bit binary RTN
- component-level sensitivity tests
- simplified paper-inspired PTQ baselines
- experimental mixed-precision ternary stack
- ablations for saliency rescue, per-row fitting, alternation, and bias correction
- simplified GPTQ / Hadamard sanity checks

## Important clarification

The paper-inspired methods in this notebook are not full official reproductions.

They are compact implementations of the core ideas so I can test whether those ideas survive on Gemma-3-270M under one controlled setup. Full reproductions would require more exact calibration, grouping, Hessian handling, rotation folding, layerwise tuning, and optimized kernels.

So the right framing is:

> simplified paper-inspired baselines, not full SOTA replications.

## Why this project matters

Sub-2-bit quantization is attractive because it could reduce model memory and inference cost, but this notebook shows that naive post-training ternary quantization can destroy model quality on small LLMs.

The interesting result is not that sub-2-bit PTQ works immediately. It is that the failures are diagnosable:

- local reconstruction improvement does not always improve perplexity
- embeddings and output-writing layers are highly sensitive
- pure ternary quantization is too aggressive
- saliency rescue is currently the strongest stabilizer
- mixed precision looks more realistic than pure 1.58-bit PTQ

## Limitations

- This is fake quantization, not a custom low-bit inference kernel.
- Perplexity is the main evaluation metric so far.
- The paper-inspired methods are simplified.
- The GPTQ and Hadamard baselines are sanity checks, not full implementations.
- The notebook currently focuses on Gemma-3-270M only.
- The current best sub-2-bit result is still much worse than int4.

## Next steps

- Move the quantization code into clean `src/quant/` modules.
- Add a proper GPTQ implementation with damping and Cholesky-style Hessian handling.
- Implement real Hadamard / rotation folding instead of a simplified check.
- Sweep saliency rescue percentage per layer.
- Add layer skipping for highly fragile modules.
- Compare against a real int4 GPTQ baseline.
- Test actual low-bit storage and inference once the accuracy side is stronger.

## Repository structure

```text
gemma-sub2bit-ptq-study/
├── README.md
└── notebooks/
    └── gemma_sub2bit_ptq_study.ipynb
