# Model architectures — how `Ids` is computed

This explains, in plain terms, every way this codebase turns two numbers —
`Vgs` (gate voltage) and `Vds` (drain voltage) — into a predicted drain
current `Ids`. No prior ML or device-physics background assumed; jargon is
defined the first time it's used.

Which method a training run uses is controlled by one config string,
`equation_type`. It comes in exactly two shapes:

```
equation_type = "pure"                → the neural network IS the model
equation_type = "pure:<wrapper>"       → the network's output gets reshaped by <wrapper>
equation_type = "noNN_knee:<eq>"       → an empirical formula <eq> is the model, and the
                                          network only adds a small correction on top
```

(Confirmed by checking every `equation_type` value actually used in
`physics_nn_configs/*.json` — nothing else appears in this project.)

```mermaid
flowchart TD
    IN["Vgs, Vds"] --> SPLIT{"equation_type"}
    SPLIT -->|"pure[:wrapper]"| PUREPATH["Neural network → optional wrapper"]
    SPLIT -->|"noNN_knee:eq"| HYBRIDPATH["Empirical formula + a gated correction from the network"]
    PUREPATH --> OUT["Ids"]
    HYBRIDPATH --> OUT
```

---

## 1. The base network

Every path starts with the same kind of network: a small stack of layers
where — unusually — **each neuron in a layer can use a different activation
function**. Normal networks use one activation for a whole layer; here, one
layer might mix `tanh` and `sin` neurons side by side. An architecture string
like

```
[["tanh", "sin", "swish"], ["tanh", "tanh"]]
```

means: layer 1 has 3 neurons (tanh, sin, swish), layer 2 has 2 neurons (both
tanh).

```mermaid
flowchart LR
    subgraph L1["layer 1 · 3 neurons"]
        A1["tanh"]
        A2["sin"]
        A3["swish"]
    end
    subgraph L2["layer 2 · 2 neurons"]
        B1["tanh"]
        B2["tanh"]
    end
    Vgs(("Vgs")) --> A1 & A2 & A3
    Vds(("Vds")) --> A1 & A2 & A3
    A1 & A2 & A3 --> B1 & B2
    B1 & B2 --> OUT["output_activation"]
```

The network's **very last step** squashes its output through one more
activation, `output_activation` — usually `linear` (no squashing at all) or
`softplus` (forces the output to be ≥ 0, a soft ramp instead of a hard
cutoff). See §2b for what happens if you pick a *bounded* one instead
(`sigmoid`, `tanh`).

---

## 2. Family A — Pure NN (`equation_type: "pure"` or `"pure:<wrapper>"`)

### 2a. No wrapper: the network's raw prediction

$$I_{ds} = \mathrm{NN}(V_{gs}, V_{ds})$$

The network is asked to learn the whole curve by itself, with no known
transistor behavior built in. Simple, but nothing stops it from predicting
something physically silly (e.g. a nonzero current at `Vds = 0`, where a
real transistor always reads exactly zero).

### 2b. Bounded output activations need a "margin"

If `output_activation` is `sigmoid` (squashes to the range `(0, 1)`) or
`tanh` (squashes to `(-1, 1)`), there's a problem: real measured currents can
be several amps, but the network's raw output can never exceed 1. A sigmoid
output alone can never predict a 2.7 A current — it's mathematically capped.

The fix, `ids_out_margin`, rescales the squashed output up to the real range:

$$I_{ds} = \text{scale} \cdot \text{activation}\big(\mathrm{NN}(V_{gs}, V_{ds})\big)$$
$$\text{scale} = (1 + \text{margin}) \cdot \max(I_{ds}^{\text{measured}})$$

`scale` is computed once, from the training data, before training starts.
With `margin = 0` (the default), `scale = 1` and bounded activations are
left uselessly capped at ±1 — so any project using `sigmoid`/`tanh` as
`output_activation` sets a margin (`0.1` is what this repo actually uses:
`scale = 1.1 × max(measured Ids)`, giving 10% headroom above the largest
value ever seen). Unbounded activations (`linear`, `softplus`) don't need
this at all — `margin` is simply ignored for them.

| `output_activation` | bounded? | needs `ids_out_margin`? |
|---|---|---|
| `linear` | no | no |
| `softplus` | no (≥0, but unbounded above) | no |
| `sigmoid` | yes, `(0,1)` | yes |
| `tanh` | yes, `(-1,1)` | yes |

(Verified directly in the repo's own sweeps: the `sigmoid_margin10` and
`tanh_margin10` config sweeps set `output_activation: "sigmoid"`/`"tanh"`
together with `ids_out_margin: 0.1`; the `softplus` sweep sets neither —
`per_neuron_simple_angelov_nn_test.py:778-790`.)

### 2c. Wrapper: reshaping the output to look like a transistor curve

The **wrapper** is a second, independent squashing step, applied *after* the
network's own `output_activation`. It's how a plain "pure" network is made
to automatically obey basic transistor physics — exactly zero current at
`Vds = 0`, current flowing the correct direction — instead of hoping the
network learns that from data alone. Every `vdsgate*` config in this repo
uses one.

**The key idea: it borrows its shape directly from the Angelov empirical
equation described in §3.** Look at the *tail* of that equation:

$$\underbrace{I_{pk}\cdot(1+\tanh\psi)}_{\text{depends on }V_{gs}\text{ only}} \cdot\ \underbrace{\tanh(\alpha \cdot V_{ds})\cdot(1+\lambda \cdot V_{ds})}_{\text{depends on }V_{ds}\text{ only — the "envelope"}}$$

The `vdsgate` wrapper reuses that exact `Vds`-envelope — literally the same
`tanh(α·Vds)·(1+λ·Vds)` formula — but **replaces the empirically-derived
`Vgs`-dependent part with the neural network**:

$$I_{ds} = \underbrace{g(\mathrm{NN})}_{\substack{\text{network stands in}\\\text{for the empirical term}}} \cdot\ \underbrace{\tanh(\alpha \cdot V_{ds})\cdot(1+\lambda \cdot V_{ds})}_{\text{same envelope as Angelov}}$$

- $g(\cdot)$ is a small "gate" function on the network's raw output — by
  default $g = \mathrm{softplus}(\mathrm{NN})$, which keeps the sign of
  `Ids` locked to the sign of `Vds` (physically required).
- $\alpha$ (steepness) and $\lambda$ (slope) are just two extra learned
  numbers, same role as in the Angelov equation.

```mermaid
flowchart LR
    NN["neural network(Vgs, Vds)"] --> GATE["gate g( )<br/>stands in for the empirical equation"]
    GATE --> M1["× tanh(α·Vds)"]
    M1 --> M2["× (1 + λ·Vds)"]
    M2 --> OUT["Ids"]
```

**`vdsgate_aeff*` — the version actually used most in this repo — takes this
one step further**: instead of fixed numbers, $\alpha$ and $\lambda$ become
small polynomials *in* `Vgs` (so the knee shape can change across the gate
voltage range, not stay identical everywhere):

$$I_{ds} = g(\mathrm{NN}) \cdot \tanh\big(a_{eff}(V_{gs}) \cdot V_{ds}\big) \cdot \big(1 + l_{eff}(V_{gs}) \cdot V_{ds}\big)$$

The suffix on the wrapper name picks how complex that polynomial is allowed
to be:

| suffix | polynomial order | meaning |
|---|---|---|
| `_lin` | 1 | straight line in Vgs |
| `_quad` | 2 | curve — **the one used most in this repo** |
| `_cub` (also the bare `vdsgate_aeff`) | 3 | more flexible curve |
| `_quart`/`_quint`/`_sext`/`_sept` | 4–7 | higher-order fits, rarely needed |
| `_sig` | — | bounds $a_{eff}$ with sigmoid instead of softplus |
| `_clam` | — | forces $\lambda$ to a single constant instead of its own polynomial |
| `_freelam` | — | lets $\lambda$ go negative (unconstrained) |

A close cousin, **`vdsgate_vdsk*`**, models the same idea but stores *where
the knee sits, in volts* directly, instead of a steepness number — more
numerically stable for curves very close to pinch-off:

$$I_{ds} = g(\mathrm{NN}) \cdot \tanh\!\left(\frac{V_{ds}}{V_{ds,knee}(V_{gs})}\right) \cdot \big(1 + l_{eff}(V_{gs}) \cdot V_{ds}\big)$$

---

## 3. Family B — Empirical + NN hybrid (`equation_type: "noNN_knee:<eq>"`)

Here the roles flip: the empirical equation is the model, and the network only
supplies a small, gated *correction* on top of it.

### 3a. The empirical equation itself (the Angelov family)

`classic_angelov`, `angelov_6_term`, and `angelov_9_term` are the same
formula with a longer or shorter polynomial in `Vgs`:

$$\psi = \sum_{n=1}^{N} P_n \cdot (V_{gs}-V_{pk})^n \qquad (N = 3,\ 6,\ \text{or } 9)$$

$$I_{emp} = \underbrace{I_{pk}\cdot(1+\tanh\psi)}_{\text{"how much current, as a function of } V_{gs}\text{"}} \cdot\ \underbrace{\tanh(\alpha\cdot V_{ds})\cdot(1+\lambda\cdot V_{ds})}_{\text{"the } V_{ds}\text{ envelope"}}$$

More polynomial terms ($N=9$) trace the curve's shape more precisely, at the
cost of needing smaller learning rates on the higher-order terms to avoid
blowing up.

`mod1_angelov` is a more advanced variant that lets the peak voltage and
steepness *themselves* shift with `Vds` — useful for capturing current
dispersion effects the simpler forms can't. See
`optim_utils/per_neuron_noNN.py:mod1_angelov` for its full form.

### 3b. The gate: when does the network get to speak?

A **gate** decides how much the network's correction is allowed to
contribute, and it fades toward zero as `Vds` grows (deep in the region
where the empirical equation is already reliable on its own):

$$\text{gate} = 1 - \tanh\!\left(\frac{|\alpha|}{k} \cdot V_{ds}\right)$$

Near `Vds = 0` the gate is close to 1 (network fully active); at large `Vds`
it decays toward 0 (the empirical equation takes over). It reuses that equation's
own $\alpha$ — it isn't learned separately — so the gate automatically
matches how steep that particular equation's knee is. $k$
(`knee_alpha_scale`) just widens or narrows that transition.

```mermaid
flowchart TD
    EMP["Empirical equation: I_emp(Vgs, Vds)"] --> COMBINE
    Vds --> GATEFN["gate = 1 − tanh(|α|/k · Vds)"]
    NN["Neural network(Vgs, Vds)"] --> COMBINE
    GATEFN --> COMBINE{"knee_combiner"}
    COMBINE -->|sum| C1["I_emp + gate·NN"]
    COMBINE -->|product| C2["I_emp · (1 + gate·NN)"]
    COMBINE -->|sum_gated_vgs| C3["I_emp + gate·h(Vgs)·NN"]
    COMBINE -->|residual| C4["I_emp · (1 + α_max·gate·tanh(NN))"]
    C1 & C2 & C3 & C4 --> OUT["Ids"]
```

### 3c. `knee_combiner` — how the correction is folded in

| `knee_combiner` | formula | plain-language meaning |
|---|---|---|
| `sum` (default) | $I_{emp} + \text{gate}\cdot\mathrm{NN}$ | network adds/subtracts current directly |
| `product` | $I_{emp}\cdot(1+\text{gate}\cdot\mathrm{NN})$ | network's output is a *percentage* nudge on the empirical-equation prediction |
| `sum_gated_vgs` | $I_{emp} + \text{gate}\cdot h(V_{gs})\cdot\mathrm{NN}$ | adds a *second* gate that also fades the network out near pinch-off (fixes a common gm "bump" artifact there) |
| `residual` | $I_{emp}\cdot(1+\alpha_{max}\cdot\text{gate}\cdot\tanh(\mathrm{NN}))$ | caps the correction to at most $\pm\alpha_{max}$ of the empirical-equation prediction, so it can never run away |

> **One correction to a related doc:** `docs/README.md` used to also list
> `max`/`min` as valid `knee_combiners` values. They don't exist in the code
> — only the four above are implemented (anything else silently behaves like
> `sum`). Already fixed there.

---

## 4. Tight-prior seeding (`use_opt_params` + `freeze_physics`)

A cross-cutting trick used with the `noNN_knee` family: rather than training
an empirical equation's parameters from a random start, seed them from an
empirical-only fit (done separately via SLSQP, a classic optimization
algorithm — see `docs/README.md` §4), then keep each parameter **within
±10% of that seeded value** during training:

$$\delta = |v_{seed}| \cdot 0.10$$
$$v = (v_{seed}-\delta) + \big[(v_{seed}+\delta)-(v_{seed}-\delta)\big]\cdot\mathrm{sigmoid}(\theta)$$

$\theta$ is the actual trainable number, but no matter what value it takes,
`sigmoid(θ)` always stays between 0 and 1 — so `v` can *never* leave its
±10% box. This keeps the equation's parameters physically sensible even while
letting them adjust slightly to fit the data better.

- **`freeze_physics: false`** — `θ` trains; `v` refines anywhere inside the box.
- **`freeze_physics: true`** — `θ` never updates; `v` stays exactly at its seed.

```mermaid
flowchart LR
    SEED["SLSQP-fitted seed value"] --> BOX["allowed range: seed ± 10%"]
    THETA["trainable number θ"] --> SIG["sigmoid(θ), always between 0 and 1"]
    SIG --> BOX
    BOX --> V["the value actually used in the equation"]
```
