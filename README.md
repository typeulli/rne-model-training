# rne-model-training

Neural surrogates for the transient temperature field produced by
`rne-am-simulation`: a Gaussian laser sweeping the top face of a 40 x 10 x 6 mm
plate. Every model learns the same map

```
T(x, y, z, t ; P)        x,y,z [m]   t [s]   P [W]   T [K]
```

from the same corpus — seven laser powers (100 … 250 W in 25 W steps, 1 320 200
points each) under `rne-am-simulation/data/train`, with 160 W held out entirely
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
| `pidon` | DeepONet | — | PINN | ✓ | the operator architecture |
| `gdon` | DeepONet | `g + p` | data MSE | ✓ | the operator without the physics |
| `gpidon` | DeepONet | `g + p` | PINN | ✓ | all three together |

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
nothing else changed: same stack, same activation, same objective, same
hyperparameters, same rows drawn from the same corpus.

They exist to measure what `P` is worth. The corpus holds seven powers, and the
same `(x, y, z, t)` carries a different temperature in each of them, so a network
that cannot see `P` cannot resolve which one it is being asked about — the best it
can do is the power-averaged field. That average is a hard floor:

```
RMSE floor = 12.821 K      # the best ANY P-blind model can achieve on this corpus
```

`cmlp` and `cgmlp` land within a few tenths of a Kelvin of it, which is the point:
they have learned everything learnable from the coordinates alone, and they fail
*only* because `P` is missing.

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
same thing (they differ by 0.1 K RMSE). The choice is therefore close to optimal,
and `cpimlp`'s remaining gap to the floor is the ordinary PINN optimisation
penalty rather than that conflict.

---

## Operator nets — `pidon`, `gdon`, `gpidon`

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
in space and time, so in `gpidon` it shifts the gate without contributing a
derivative of its own, leaving the PDE residual undisturbed.

- **Network input** — `(laser_power, coords)`: `[B, 1]` branch input and `[B, 4]`
  trunk input of `(x, y, z, t)`. A branch batch of 1 is broadcast over the query
  points, so one power can be evaluated against a whole grid in a single call.
- **Output** — `[B, 1]` in Kelvin

| model | gate `G` | objective |
|---|---|---|
| `pidon` | none (output unconstrained) | PINN |
| `gdon` | `g + p` on the inner product | data MSE only |
| `gpidon` | `g + p` on the inner product | PINN |

`pidon` is kept so checkpoints trained before the gate was added still load;
in it, the only Gaussian in the problem is the laser source term of the top
boundary condition. `gdon` is `gpidon` minus the physics residuals, and isolates
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
python train.py gpidon --help
```

A model is any directory under `models/` containing a `train.py` that exposes
`main(argv)`. The registry globs for them, so adding one needs no edit to any
central file.

The MSE-trained models count in **epochs** (`--steps-per-epoch` random batches,
sampled with replacement); the PINN-trained ones count in **iterations**, because
their physics batches are resampled every step and there is no pass over the data
to speak of. Both write TensorBoard scalars to `runs/` and keep the single best
checkpoint by validation RMSE under `checkpoints/<model>/<run-name>.pt`.

## Results

Validation RMSE in Kelvin, on a random 10% row split. Measured at the **64×3**
architecture (~8.7k parameters), the one size at which every dense model has been
run and the numbers are therefore comparable:

| | with `P` | without `P` |
|---|---|---|
| plain | `mlp` **2.568** | `cmlp` **13.388** |
| gated | `gmlp` **2.396** | `cgmlp` **13.187** |
| physics | `pimlp` **4.160** | `cpimlp` **16.414** |

against a P-blind floor of **12.821 K**. Withholding the process parameter costs
roughly 5× the error and puts the controls flat against the information-theoretic
limit — they are not undertrained, they are starved.

Two caveats on this table. The `pimlp-64x3` checkpoint was trained on a longer
schedule than the others (best at step 36 500, versus 20 000 for the rest), so its
comparison with `cpimlp` is the one pair that is not exactly matched. And the
operator-net checkpoints on disk are at a different size (128×4, latent 128) and
so do not belong in it; `gpidon`'s current checkpoint is from an aborted run
(best at step 1) and is effectively untrained.

Larger dense models do better on the `P`-aware side — `gmlp` at 256×4 reaches
0.630 K — while the controls barely move (`cmlp` at 256×4: 12.916 K), which is
what a floor looks like.

## What is not here yet

Every model above is pointwise: it predicts one `T` per query, so a full volume
costs 165 025 forward passes. [`TODO.md`](TODO.md) sets out the design for a
tenth model that would instead learn the field's spatial Fourier spectrum and
recover the whole volume in a single inverse transform — including the heat
equation rewritten mode by mode, and why the boundary conditions have to stay in
real space.
