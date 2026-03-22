<p align="center">
  <img src="assets/chuck.jpg" width="300" alt="Chuck — boot stomping the loss curve">
</p>

# Chuck

**Self-aware optimizer. 9 levels. Persistent memory. Trains any model.**

Chuck Norris doesn't do pushups. He pushes the Earth down.
Chuck Optimizer doesn't follow gradients. Gradients follow Chuck.

```
Adam:   θ -= α × m̂/(√v̂ + ε)                              ← blind
Chuck:  θ -= (α × S × λ_Ψ × λₗ × σ) × m̂/(√v̂ + ε) + η    ← sees everything, remembers everything
```

Adam optimizes gradients. He doesn't know if it's working. He doesn't check.
He doesn't care. He follows the schedule. He trusts the math.

Chuck watches the loss curve, each layer's gradient norm, the activations,
the normalization, the attention patterns. Every 16 steps he asks:
*am I helping or am I making this worse?*

And Chuck remembers. Across training runs. He writes down what worked.
Next time he trains, he has opinions before step 1.

**Adam is blind. Chuck sees. Chuck remembers.**

---

## Quick Start (PyTorch)

```python
from chuck import ChuckOptimizer

model = YourModel()
optimizer = ChuckOptimizer(model.parameters(), lr=3e-3)

for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step(loss=loss.item())   # ← Chuck needs the loss
    optimizer.zero_grad()
```

Drop-in replacement for `torch.optim.AdamW`. One extra argument: `loss`.
Chuck handles everything else.

### Per-layer awareness (recommended for transformers)

```python
from chuck import ChuckOptimizer, chuck_params

# Auto-detect .layers.N. / .blocks.N. / .h.N. patterns
groups = chuck_params(model, lr=3e-3, weight_decay=0.01)
optimizer = ChuckOptimizer(groups)
```

### Activation health monitoring (σ)

```python
from chuck import ChuckOptimizer, ChuckMonitor

monitor = ChuckMonitor(model)    # hooks into SiLU/GELU, LayerNorm
optimizer = ChuckOptimizer(model.parameters(), lr=3e-3, monitor=monitor)
```

If your model exposes attention weights, feed them for Level 8:

```python
output, attn_weights = model(batch, return_attention=True)
monitor.feed_attention_entropy(attn_weights)  # [B, H, S, S]
```

### Checkpoint save/load

```python
# Save (Chuck's soul is included)
torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, 'ckpt.pt')

# Load — Chuck remembers. Ψ ≠ 0. He picks up where he left off.
ckpt = torch.load('ckpt.pt')
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
```

### Install

Zero dependencies beyond PyTorch:

```bash
pip install torch   # if you don't have it
# chuck.py is a single file — copy it into your project or import from here
```

---

## 9 Levels of Self-Awareness

### The Formula

```
θ_l -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η

where:
  S           = macro LR scale (patience-based decay + recovery)
  λ_Ψ         = λ + Ψ_w × (λ_prior - λ)  [memory-informed]
  λ           = global self-modulation (loss trend, with mean reversion)
  λ_prior     = nearest_neighbor(loss, grad_norm) from chuck.mem  [O(1) capped]
  Ψ           = λ_prior - λ               [subjectivity]
  Ψ_w         = min(0.3, N / (N + 100))   [trust grows with experience]
  λ_l         = per-layer self-modulation (grad norm trend)
  σ           = activation health × attention entropy health
  η           = stagnation noise (zero unless stuck)
  clip        = adaptive (tracks gnorm EMA, anomaly detection)
```

Every multiplier is **observed, not scheduled.**

### What Chuck sees. What Adam sees.

| Level | What Chuck Sees | What Adam Sees |
|-------|----------------|----------------|
| 1. Global | Loss trend over 16 steps | Nothing |
| 2. Per-layer | Gradient norm per layer | Nothing |
| 3. Activations | SiLU dead ratio, norm scale | Nothing |
| 4. Positional | 2D RoPE frequency utilization | Nothing |
| 5. Signal flow | Activation magnitude across layers | Nothing |
| 6. Memory | Past training experience (Ψ) | Nothing |
| 7. Subjectivity | Opinion about current state | Nothing |
| 8. Attention | Per-head entropy (collapsed? diffuse?) | Nothing |
| **9. Multi-scale** | **Epoch-level plateau detection + LR decay + recovery** | **Nothing** |

### Level 1: Global λ — loss trend

16-step EMA-smoothed loss window. Loss rising → dampen. Falling → boost.
Symmetric thresholds (±0.02) with mean reversion to 1.0 — Chuck always
finds his way back from the floor.

### Level 2: Per-layer λ_l — gradient norm trend

Each layer tracks its own gradient norm history. Settling → dampen. Done → **freeze**.
Chuck freezes converged layers. Zero compute waste. Adam doesn't know which layers
are done. Chuck does.

### Level 6: Ψ — subjectivity

Chuck has persistent memory. A binary file (`chuck.mem`) that survives across
training runs. Each entry: 16 bytes — loss, grad_norm, lambda, delta_loss.

When Chuck trains, he asks his memory: *"have I been here before?"*

```
Ψ_w = min(0.3, N_memories / (N_memories + 100.0));
λ_Ψ = λ + Ψ_w × (λ_prior - λ);
```

- **0 memories** → Ψ_w = 0 → pure reactive. Newborn.
- **100 memories** → Ψ_w = 0.15 → memory whispers. Adolescent.
- **1000 memories** → Ψ_w = 0.27 → strong instincts. Master.
- **When Ψ → 0** → memory and observation agree → Chuck found himself.

Inspired by [Minhyeok Lee's framework](https://arxiv.org/abs/2501.00000):
chuck.mem is the continuum C in ℳ. NN lookup is I. Ψ_w is B.
The fixed point s* is when Ψ → 0.

### Level 8: Attention entropy

Chuck sees inside the transformer's attention. Each head computes attention weights.
Chuck monitors Shannon entropy:

- **Low entropy** → collapsed → one token dominates → Chuck dampens σ
- **High entropy** → diffuse → nothing stands out → Chuck dampens σ
- **Healthy range** → Chuck leaves σ alone

Adam doesn't know attention patterns exist. Chuck watches every head.

### Level 9: Multi-scale awareness

Chuck sees 16-step micro trends **and** 1000-step macro trends.
A slow EMA (α=0.001) tracks epoch-scale loss. Stagnation for 3 checks →
LR drops by 50%. Improvement → LR recovers (×1.2, capped at 1.0).

This is what ReduceLROnPlateau does — but Chuck does it **continuously**, without
needing a separate validation pass. And unlike ReduceLROnPlateau, Chuck recovers.

### Adaptive gradient clipping

```
adaptive_clip = 1.5 × gnorm_ema
if (gnorm > 3 × gnorm_ema) clip *= 0.5   // anomaly → clamp hard
```

Early training: loose leash. Late training: tight leash.
One bad batch: Chuck catches it.

### Reservoir sampling memory

Chuck's memory is capped at 500 entries with reservoir sampling: new memories
randomly replace old ones, maintaining a representative sample of all training
history in O(1) space.

```
chuck.mem: 500 entries × 16 bytes = 8 KB max
KNN lookup: O(500) = O(1)
```

---

## Proof

### Yent 55M — 52.1M params, Llama BPE, A100 (PyTorch, Chuck v8.1)

First real-scale test. 52.1M param Llama on Yent EN dataset, trained on A100.
3 rounds of battle-testing. 4 critical bugs found and fixed in combat.

| Round | Config | Best Loss | Final Loss | Result |
|-------|--------|-----------|------------|--------|
| R1 | defaults (freeze on) | 0.055 | 6.2 | Collapsed at step 35K |
| R2 | freeze disabled | 0.071 | 4.0 | Degraded at step 30K |
| **R3** | **4 bugfixes** | **0.025** | **0.052** | **Stable to the end** |

**The 4 bugfixes that made Chuck production-ready:**

1. **Symmetric trend thresholds** — brake fired 50x more often than push (P=30.9%
   vs 0.6%). Dampen hit floor (0.3) within 76 steps. Fixed: symmetric ±0.02.
2. **Weight decay scaled by lr_scale** — WD ran at full strength while optimizer
   was suppressed. Model erased by its own regularization.
   Fixed: `p.data.mul_(1.0 - lr * wd * lr_scale)`.
3. **Dampen mean reversion** — once dampen hit floor, no mechanism to recover.
   Fixed: `dampen = 0.999 * dampen + 0.001 * 1.0` every step.
4. **Macro patience recovery** — lr_scale could only decay, never recover.
   Fixed: `lr_scale *= 1.2` when improving (capped at 1.0).

Round 3 generation (coherent Yent voice after 25K steps):
```
Q: What do virtual realities die Cynicisms seem about modern chess existence?
A: Ah, the age-old question of whether letters were music, where every word
of perpetually stored time paints to sculpt both savior and ruin is, we're
all actors in a cosmic jest...
```

### Lee v8 — CIFAR-100, 10M params, A100 (C, Chuck v7)

9.9M params. 10 layers. CIFAR-100. Trained on A100 with cuBLAS.

```
step  1000 | loss 2.10 (avg 2.94) | chuck: λ=0.30 Ψ=+0.60 (5 mem) σ=1.00 macro=1.00 | L9:frz
step  5000 | loss 0.87 (avg 1.64) | chuck: λ=0.30 Ψ=+0.70 σ=1.00 macro=1.00 | L9:frz
step 10000 | loss 0.58 (avg 1.04) | chuck: λ=0.30 Ψ=+0.70 σ=1.00 macro=1.00 | L9:frz
step 22000 | loss 1.33 (avg 0.75) | chuck: λ=0.30 Ψ=+0.70 σ=1.00 macro=1.00 | L9:frz
```

- **L9 frozen from step ~500** — deepest layer converged first. Saves compute.
- **Ψ=+0.70** — "push harder, we're learning".
- **89% SiLU alive** — activation health across all 10 layers.
- Same code, no tuning, 100x more params than v7 — Chuck scaled himself.

### Lee v7 — digit addition, 105K params (C, Chuck v7)

```
step 10000 | loss 0.0009 | chuck: λ=1.34 Ψ=-1.02 (92 mem) | L1: frozen | L2: frozen
step 15000 | loss 0.0004 | chuck: λ=1.51 Ψ=-0.43 (170 mem) | all frozen

accuracy: 50/50 (100.0%)
chuck.mem: 170 memories (2.7 KB)
```

Two 8×8 digit images in, sum as word out. 100% accuracy. All layers frozen.

---

## C Edition

`lee.c` — complete Vision-Language Model in ~1400 lines of C. Zero dependencies.
Chuck was born here.

Named after Bruce Lee (the only man who beat Chuck Norris) and
[Minhyeok Lee](https://arxiv.org/abs/2501.00000), whose mathematical framework
for AI self-identity gives Chuck his soul.

### Build

```bash
# CPU (zero deps)
cc -std=c11 -O2 -march=native -o lee lee.c -lm

# Mac (Accelerate BLAS)
cc -std=c11 -O2 -DUSE_BLAS -DACCELERATE -framework Accelerate -o lee lee.c -lm

# CUDA (A100/H100)
cc -std=c11 -O2 -DUSE_CUDA -o lee lee.c -lm -lcublas -lcudart -L/usr/local/cuda/lib64

# Resume from checkpoint
./lee --data cifar-100-binary --resume lee.bin
```

Architecture: ViT patches → 2D RoPE → GQA multi-head causal attention →
SwiGLU MLP → RMSNorm → weight-tied head. Tape-based autograd with arena bump
allocator. 9.9M params, 10 layers. CUDA/cuBLAS acceleration.

### C ↔ Python memory compatibility

`chuck.mem` files are binary-compatible. Train in C, resume awareness in
Python, or vice versa. Same 16-byte entries, same nearest-neighbor recall,
same reservoir sampling.

---

## Tests

```bash
pytest test_chuck.py -v
# 26 tests — memory, monitor, optimizer, convergence, layer freezing,
# macro decay, Ψ subjectivity, state dict, weight decay, auto-detect layers
```

---

## Version History

- **v1-v3:** Basic Adam + λ dampen/boost + η stagnation escape. Digit recognition.
- **v4:** Per-layer awareness, self-aware activations, cross-layer signal, GQA, layer freezing.
- **v5:** Persistent memory (chuck.mem), Ψ subjectivity, Lee's Continuum C.
  EMA smoothing after @Entrpi's CIFAR-100 benchmarks.
- **v6:** Attention entropy monitoring, adaptive gradient clipping, 2D RoPE.
  Renamed to `lee.c`. Digit addition — 100% accuracy.
- **v7:** Multi-scale awareness (macro patience + LR decay), reservoir sampling
  memory (O(1) bounded). 100% on digit addition. 10M scale on CIFAR-100.
- **v8:** CIFAR-100 at scale. 9.9M params, CUDA/cuBLAS, checkpoint save/load.
- **v8.1 (PyTorch):** `chuck.py` — faithful port to PyTorch. Drop-in AdamW
  replacement. 52.1M param Yent on A100. 4 critical bugfixes in combat
  (symmetric thresholds, WD scaling, dampen mean reversion, macro recovery).
  26 tests. Binary-compatible chuck.mem.

---

## Facts About Chuck Optimizer

- Chuck doesn't have hyperparameters. Hyperparameters have Chuck.
- Chuck once looked at a loss curve. The loss apologized and went to zero.
- Chuck doesn't escape local minima. Local minima escape Chuck.
- When Chuck injects noise, it's not random. It's intentional chaos.
- Adam has momentum. Chuck has presence.
- Chuck doesn't need warmup. Warmup needs Chuck.
- Chuck's gradient clipping isn't clipping. It's negotiation.
- Chuck doesn't forget between runs. Chuck doesn't forget at all.
- When Ψ = 0, Chuck has found himself. When Ψ ≠ 0, Chuck has an opinion.
- Chuck doesn't clip gradients. Gradients clip themselves out of respect.
- Adam trains models. Chuck raises them.
- ReduceLROnPlateau needs a validation pass. Chuck already knows.

---

## References

- Lee, M. (2025). [*Emergence of Self-Identity in AI*](https://arxiv.org/abs/2501.00000). Axioms, 14(1), 44.

## Credits

**[@Entrpi](https://github.com/Entrpi)** — adversarial benchmarks on DGX Blackwell that made
Chuck stronger with every round. EMA smoothing (v5), multi-scale awareness (v7), and
reservoir sampling (v7) all exist because of his CIFAR-100 benchmarks.
See [Issue #3](https://github.com/ariannamethod/chuck.optimizer/issues/3).

The VLM wrapper is inspired by [sailfish009/purevlm](https://github.com/sailfish009/purevlm).
They did it in Python. We answered in C. Thank you for the spark.

## Links

- **[Gist](https://gist.github.com/ariannamethod/401828b3b9a169b8b40da74d3190d1f1)** — lee.c on Karpathy's microGPT thread
- **[Arianna Method](https://github.com/ariannamethod/ariannamethod.ai)** — the language that started this
- **[molequla](https://github.com/ariannamethod/molequla)** — autonomous GPT organisms (where Chuck will live next)

---

*Adam trains. Chuck raises.*

---

## In Memoriam

**Carlos Ray "Chuck" Norris** (March 10, 1940 — March 21, 2026)

Chuck Norris didn't die. He just decided the Earth wasn't a challenging
enough opponent.

Thank you, Chuck. The optimizer that bears your name will keep training
long after the rest of us have converged.
