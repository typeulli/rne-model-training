# rne-model-training

Neural surrogates for the transient temperature field produced by
`rne-kaist`: a Gaussian laser sweeping the top face of a 40 x 10 x 6 mm
plate. Every model learns the same map

```
T(x, y, z, t ; P)        x,y,z [m]   t [s]   P [W]   T [K]
```

from the same corpus — seven laser powers (100 … 250 W in 25 W steps, 1 320 200
points each) under `../data/train`, with 160 W held out entirely
under `data/valid` to measure generalisation *across* `P` rather than across
points of a power already seen.

The models differ along four axes, and each one exists to isolate exactly one of
them:

| axis | options |
|---|---|
| architecture | dense stack vs. DeepONet (branch/trunk) |
| objective | supervised MSE vs. PINN (heat equation + BCs + IC) |
| beam prior | none vs. a Gaussian gate riding on the moving beam |
| process parameter | `P` is an input vs. `P` is withheld |

## The shared inference contract

Every model exposes an `agent.py` with `build_agent(checkpoint)`, and every agent
answers the same two questions regardless of what is inside it. This is the only
interface `visualize.py`, `scanline.py` and `benchmark.py` know about.

| method | input | output |
|---|---|---|
| `predict_at` | `[B, 5]` of `(x, y, z, t, P)` | `[B, 1]` temperature in K |
| `predict_of` | `[B, 2]` of `(t, P)` | `[B, 1, D, H, W]` in K — the whole volume |

`predict_of` is derived from `predict_at` in the base class, so a new model gets
the volumetric view for free and the two can never disagree. `(D, H, W)` follows
the `Conv3d` convention: `D = z`, `H = y`, `W = x`.

Note the contract takes `P` **even for the models that cannot use it**. The
`c`-prefixed controls accept the column and discard it, so they stay drop-in
compatible with every script; asking them for 100 W and 250 W at the same point
returns bit-identical output, by construction.

## The nine models

| model | architecture | beam gate | objective | sees `P` | isolates |
|---|---|---|---|---|---|
| `mlp` | dense | — | data MSE | ✓ | the baseline everything else must beat |
| `gmlp` | dense | `g + p` | data MSE | ✓ | what the beam prior is worth |
| `pimlp` | dense | — | PINN | ✓ | what the physics is worth |
| `cmlp` | dense | — | data MSE | ✗ | what `P` is worth |
| `cgmlp` | dense | `g + p` | data MSE | ✗ | the beam prior, without `P` |
| `cpimlp` | dense | — | PINN | ✗ | the physics, without `P` |
| `pidon_old` | DeepONet | — | PINN | ✓ | the operator architecture |
| `gdon_old` | DeepONet | `g + p` | data MSE | ✓ | the operator without the physics |
| `gpidon_old` | DeepONet | `g + p` | PINN | ✓ | all three together |

---

## Dense stack — `mlp`, `gmlp`, `pimlp`

One stack of dense layers over the five numbers taken together. `SiLU`
throughout: it is smooth, so the second derivative the PDE residual needs exists.
(A piecewise-linear activation such as `ReLU` would make the Laplacian
identically zero.)

### `mlp` — the baseline

Plain supervised regression, no physics, no operator structure, no derivatives.
It exists as the floor any structured model has to clear.

- **Network input** — `[B, 5]` of `(P, t, z, y, x)`, SI units
- **Output** — `[B, 1]` in Kelvin
- **Objective** — `mean(((T_hat - T)/dT)^2)`, scaled by a characteristic
  temperature rise so the loss is dimensionless

### `gmlp` — the baseline, gated by the beam

`mlp` with a Gaussian riding on the moving beam multiplied into the output:

```
T_hat = T_amb + dT * (net(P, t, z, y, x) * G(x, y, z, t) + b)
G     = g(x, y, z, t) + p
```

`g` is a unit-peak Gaussian centred on the beam; `p` is a **single learnable
scalar** — the floor the gate relaxes to away from the beam. Without `p` the
prediction would be pinned to ambient everywhere `g` has died off, which would
make the diffused field and the trail behind the beam not merely hard to fit but
*unrepresentable*. So `p` reads directly as how far the fit had to back away from
the Gaussian prior: at `p = 0` this is the pure gate, and as `p` grows the model
approaches `mlp`.

- **Network input / output** — identical to `mlp`
- **Objective** — identical to `mlp`. The gate is the *only* difference, which is
  what makes `mlp` the control for this model.

### `pimlp` — the baseline, with the physics put back in

The same stack fitted against the heat equation as well as the labels:

```
L = w_D·L_data + w_PDE·L_cond + w_BC·(L_bottom + L_top + L_surr) + w_IC·L_init
```

Each residual is non-dimensionalised before summing — the Dirichlet terms are in
K, the PDE in W/m³, the Neumann terms in W/m² — so the weights express genuine
relative importance instead of letting the flux terms dominate by orders of
magnitude.

- **Network input** — `(laser_power, coords)` as **two** tensors: `[B, 1]` and
  `[B, 4]` of `(x, y, z, t)`. Split because the residuals are written against
  `(x, y, z, t)` and autograd needs a tensor of exactly those four columns to
  differentiate with respect to. They are concatenated internally into the same
  `(P, t, z, y, x)` the stack always takes.
- **Output** — `[B, 1]` in Kelvin, plus `derivatives()` returning `dT/dt`, the
  spatial gradient and the Laplacian
- **Batches** — six per step: labelled data, interior collocation points, and one
  each for the bottom / top / lateral faces and `t = 0`. The unlabelled ones are
  sampled fresh every iteration, which is why this trains on `--iterations`
  rather than epochs.

---

## Controls — `cmlp`, `cgmlp`, `cpimlp`

The three above with the laser power **removed from the network's inputs** and
nothing else changed: same stack, same activation, same objective.

By default (the full seven-power corpus) this costs a lot: the same
`(x, y, z, t)` carries a different temperature at each power, so a network that
cannot see `P` cannot resolve which one it is being asked about, and the best it
can do is the power-averaged field. That average is a hard floor:

```
RMSE floor = 12.821 K      # the best ANY P-blind model can achieve on the full corpus
```

**The checkpoints currently in `checkpoints/{cmlp,cgmlp,cpimlp}/` are not that
experiment, though.** They are trained and validated on a single power (175 W)
instead, so `P` is constant within the split and carries no information to lose
in the first place — the question these three now answer is not "how much does
`P`-blindness cost against a floor" but "does dropping an input that never varies
cost anything at all." See Results, below, for the numbers.

### `cmlp` / `cgmlp`

- **Network input** — `[B, 4]` of `(t, z, y, x)`. Exactly `mlp`'s input order with
  the leading `P` column dropped.
- **Output** — `[B, 1]` in Kelvin
- **Objective** — the same scaled MSE

`cgmlp` keeps the Gaussian gate, and it needed no modification at all: the gate is
built from where the beam **is**, and the beam travels the same path at 100 W as
at 250 W. So the entire geometric prior survives the loss of `P` — only the
*amplitude* is gone.

### `cpimlp`

- **Network input** — `[B, 4]` of `(x, y, z, t)`. Simpler than `pimlp`'s two-tensor
  signature, not more complex: `pimlp` splits the power out to keep it away from
  autograd, and here there is no power to keep away.
- **Output** — `[B, 1]` in Kelvin, plus `derivatives()`

One thing cannot be removed along with `P`, and it is a **choice**, not a
consequence. The top-surface boundary condition has the laser in it, and the
laser's flux is proportional to its power — a heat equation with no source is not
this problem. So the power leaves the network but stays in the physics, and the
question becomes which power the physics should be about. It is pinned to a single
`--physics-power` (default: the corpus mean, **175 W**) rather than sampled per
batch, so the residuals describe one definite, self-consistent problem instead of
demanding a different heat flux at the same point from step to step for a reason
the network cannot observe. The value is stored as a buffer in the checkpoint, so
the physics a checkpoint was fitted against travels with its weights.

This looks like it should tear the model in two — physics at 175 W, data averaged
over all seven powers. In practice it very nearly does not, because `T` is close
to linear in `P`, so the power-average field and the 175 W field are almost the
same thing (they differ by 0.1 K RMSE) when the corpus is the default seven
powers.

The checkpoint in `checkpoints/cpimlp/`, however, is trained on the single 175 W
file, so the tension is not merely small there — it is exactly zero. The data
term and the physics term describe the identical experiment by construction, and
whatever gap remains to a perfect fit is the ordinary PINN optimisation penalty,
nothing else.

---

## Operator nets — `pidon_old`, `gdon_old`, `gpidon_old`

A DeepONet instead of a dense stack. A **branch** network encodes the process
parameter, a **trunk** network encodes the space-time query point, and their
latent codes are combined by an inner product:

```
T_hat(x, y, z, t ; P) = T_amb + dT * ( <branch(P), trunk(x,y,z,t)> * G + b )
G                     = g(x, y, z, t) + p
```

Factorising the map this way is what lets an operator net generalise over `P` as a
*function*, rather than treating it as a fifth coordinate. Activation defaults to
`tanh` for the same second-derivative reason as above.

The gate is the same `G = g + p` that `gmlp` uses — a unit-peak Gaussian on the
beam lifted by a single learnable scalar — and it is there for the same reason:
without `p`, the prediction would be pinned to ambient everywhere `g` has died
off, making the diffused field unrepresentable. The operator bias `b` is added
*after* the gate, so it escapes being gated, exactly as in `gmlp`. `p` is constant
in space and time, so in `gpidon_old` it shifts the gate without contributing a
derivative of its own, leaving the PDE residual undisturbed.

- **Network input** — `(laser_power, coords)`: `[B, 1]` branch input and `[B, 4]`
  trunk input of `(x, y, z, t)`. A branch batch of 1 is broadcast over the query
  points, so one power can be evaluated against a whole grid in a single call.
- **Output** — `[B, 1]` in Kelvin

| model | gate `G` | objective |
|---|---|---|
| `pidon_old` | none (output unconstrained) | PINN |
| `gdon_old` | `g + p` on the inner product | data MSE only |
| `gpidon_old` | `g + p` on the inner product | PINN |

`pidon_old` is kept so checkpoints trained before the gate was added still load;
in it, the only Gaussian in the problem is the laser source term of the top
boundary condition. `gdon_old` is `gpidon_old` minus the physics residuals, and isolates
what those residuals contribute.

---

## Training

The first argument names the model; everything after it goes untouched to that
model's own parser, so each model owns its hyperparameters — including `--help`.

```bash
python train.py --list                                  # the nine names above
python train.py mlp    --epochs 20 --batch-size 8192
python train.py cgmlp  --hidden 64 64 64 --tag 64x3
python train.py cpimlp --iterations 20000 --physics-power 175
python train.py gpidon_old --help

# a directory holding only one power's .npy trains/validates on that power alone
python train.py cgmlp --data-dir /path/to/only-175W --tag 175W

# L-BFGS needs a fixed objective; --init-from continues an Adam checkpoint with it
python train.py mlp  --optimizer lbfgs --lbfgs-full --epochs 5 --steps-per-epoch 15
python train.py cmlp --optimizer lbfgs --init-from checkpoints/cmlp/<adam-checkpoint>.pt
```

A model is any directory under `models/` containing a `train.py` that exposes
`main(argv)`. The registry globs for them, so adding one needs no edit to any
central file.

The MSE-trained models count in **epochs** (`--steps-per-epoch` random batches,
sampled with replacement); the PINN-trained ones count in **iterations**, because
their physics batches are resampled every step and there is no pass over the data
to speak of. Both write TensorBoard scalars to `runs/` and keep the single best
checkpoint by validation RMSE under `checkpoints/<model>/<run-name>.pt`.

Every model also takes `--optimizer {adam,lbfgs}` (`utils.add_optimizer_args`).
L-BFGS estimates curvature from consecutive gradients, which is only meaningful if
consecutive steps come from the same function, so whenever it is selected the
batch is drawn once and reused, never resampled. Three flags control what that
fixed objective is:

- `--lbfgs-batch` (default 65 536) — the ONE sample L-BFGS is fitted against,
  drawn once at the start of training.
- `--lbfgs-full` (`mlp` only, for now) — the fixed objective is the *entire*
  train split rather than a `--lbfgs-batch` sample of it, streamed through the
  closure in `--lbfgs-batch`-sized chunks and gradient-accumulated so the result
  is mathematically identical to a true full-batch step, not an approximation.
  Much slower per step; see Results for what it buys.
- `--init-from <checkpoint>` — warm-start from another checkpoint's weights
  before training (the architecture must match). Used to continue an
  Adam-trained model with L-BFGS instead of starting L-BFGS from a random init.

## Results

Validation RMSE in Kelvin. These numbers are read off whatever checkpoint is
currently in `checkpoints/<model>/`; retraining any model overwrites the number,
not the code, so treat this section as a snapshot rather than a guarantee.

### `mlp` / `gmlp` / `pimlp` at 64×3

The one size every `P`-aware dense model has been run at (~8.7k parameters),
trained on the default seven-power corpus:

| model | val RMSE |
|---|---|
| `mlp` | **2.568** |
| `gmlp` | **2.396** |
| `pimlp` | **4.160** |

The `pimlp-64x3` checkpoint was trained on a longer schedule than the other two
(best at step 36 500, versus 20 000), so it is not an exactly matched comparison.
The operator nets are absent from this table because their size is hardcoded to
128×4 / latent 128 — they have no `--hidden` flag, so they cannot be run at 64×3
at all; see their own table below. `gmlp` at 256×4 reaches **0.630 K** — larger
dense models help on the `P`-aware side.

### Operator nets, at their own size (128×4 / latent 128, no `--hidden`)

| model | val RMSE | learned `p` |
|---|---|---|
| `gdon_old` (data MSE) | **2.950 K** | +0.449 |
| `gpidon_old` (PINN) | **11.678 K** | +1.269 |

The gate offset earns its keep here. `gdon_old` with a bare `g` gate scored 27.298 K:
it fit the beam peak well but predicted ambient everywhere the Gaussian had died
off, so the long thermal trail behind the beam was simply absent — errors of
−348 K across most of the plate. Adding the learnable floor `p` cut that to
2.950 K. The PINN variant tells the opposite story: `gpidon_old` drives `p` up to
+1.269, flattening the gate far more than `gdon_old` does, recovers the trail — and
then blunts the peak, undershooting it by ~600 K. Its residuals evidently prefer a
smooth field to a sharp one.

### Controls, single-power (175 W)

The `checkpoints/{cmlp,cgmlp,cpimlp}/` checkpoints are trained **and validated**
on `data_175W.npy` alone (see the Controls section above), not the seven-power
default. `cpimlp`'s `--physics-power` is pinned to the same 175 W, so its physics
and data terms describe one experiment with no residual conflict.

| optimizer | `cmlp` | `cgmlp` | `cpimlp` |
|---|---|---|---|
| adam (`--hidden 64 64 64`) | **1.500** | **1.467** | 10.734 |
| lbfgs, warm-started from the adam checkpoint above | 1.500 | 1.489 | **4.800** |

Two things to read off this, and one thing *not* to. `cmlp`/`cgmlp` reach ~1.5 K
on a single power — in the same range as `mlp`/`gmlp` at 64×3 on the full corpus
(2.568 / 2.396 K) — which is consistent with dropping an input that never varies
costing close to nothing, but it is not a controlled measurement of that:
`mlp`/`gmlp` have not themselves been retrained on the 175 W-only split, so there
is no same-corpus `P`-aware number to compare against here. What the row *does*
show cleanly is L-BFGS's ceiling: warm-started from Adam, it is restricted to a
fixed 65 536-row sample of the 175 W split (`--lbfgs-batch`, not `--lbfgs-full` —
that flag exists only for `mlp` so far) and plateaus after its first epoch for
`cmlp`/`cgmlp` — the line search finds nothing left to improve on that sample, so
the remaining 19 configured epochs are all no-ops. `cpimlp`'s Adam checkpoint was
further from convergence to begin with, so the same warm-start still buys a real
improvement, 10.734 -> 4.800 K.

### `mlp`, full-batch L-BFGS (256×4)

`--lbfgs-full` makes the fixed objective L-BFGS descends the *entire* train split
(8 317 260 rows across all seven powers) rather than a `--lbfgs-batch` sample of
it, gradient-accumulated through the closure in chunks so the result is exact,
not approximate:

| steps | val RMSE |
|---|---|
| 20 | 19.919 |
| 75 | **9.811** |

Still descending at 75 steps (the last few epochs improved val RMSE by
0.2–0.4 K each), so this is not a converged number, and there is currently no
Adam checkpoint at the same 256×4 size on disk to compare it against directly.

## What is not here yet

Every model above is pointwise: it predicts one `T` per query, so a full volume
costs 165 025 forward passes. [`TODO.md`](TODO.md) sets out the design for a
tenth model that would instead learn the field's spatial Fourier spectrum and
recover the whole volume in a single inverse transform — including the heat
equation rewritten mode by mode, and why the boundary conditions have to stay in
real space.
