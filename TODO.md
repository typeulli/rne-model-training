# TODO — a Fourier-space model (`fmlp`)

Not implemented. This document is the design, written down before any code exists
so the reasoning can be argued with first. See `README.md` for the nine models
that do exist.

Every one of those models is **pointwise**: it maps one `(x, y, z, t; P)` to one
`T`, and a full volume costs `25 x 41 x 161 = 165 025` forward passes. The proposal
is to learn the field's **spatial Fourier spectrum** instead, and recover the
volume by one inverse transform.

## Why the spectrum is the better learning target

The real-space field is a bad regression target, and it is bad in a specific way.
The temperature is near-ambient almost everywhere and dramatically peaked in a
region a beam-radius wide. Most sample points therefore carry almost no signal,
the MSE is dominated by the vast cold region, and the peak — the only part anyone
cares about — is a rare event the loss barely notices.

The Fourier transform inverts exactly this pathology. A peak that is **sharp and
isolated rather than periodic** does not concentrate its energy in a few
wavenumbers; it *spreads* it. A narrow Gaussian of radius `r_b` in space
transforms to a broad, smooth Gaussian of width `~1/r_b` in `k`, so the spectrum
has many coefficients of comparable magnitude instead of one huge value and a sea
of near-zeros. Sharper in space means flatter in `k`. An evenly-distributed target
is a well-conditioned target, and the network learns it far more readily than it
learns a spike sitting in a desert.

The second gain is structural. A spectral model maps

```
(t, P)  ->  { T_hat(kappa) }  --iFFT-->  the entire [D, H, W] volume
```

so **one forward pass predicts the whole field at once**, where a pointwise model
needs 165 025. Note this inverts the current class hierarchy: `predict_of` becomes
the native operation and `predict_at` the derived one (by interpolating the
reconstructed grid), the exact reverse of `agent.BaseAgent` today.

## The PDE, transformed

Writing `alpha = k / (rho * c_p)` for the diffusivity (2.3935e-6 m²/s, already
calibrated) and `kappa = (kx, ky, kz)` for the wavevector — `kappa`, not `k`,
because `k` is the conductivity everywhere else in this repo — the spatial Fourier
transform sends

```
laplacian(T)  ->  -|kappa|^2 * T_hat(kappa, t)
```

so the conduction equation

```
rho*c_p * dT/dt = k * laplacian(T)
```

becomes, mode by mode,

```
dT_hat(kappa, t)/dt = -alpha * |kappa|^2 * T_hat(kappa, t)
```

**Every wavenumber decouples into an independent linear ODE.** The PDE residual is
then

```
R_hat(kappa, t) = dT_hat/dt + alpha * |kappa|^2 * T_hat
L_pde           = mean over kappa of |R_hat|^2
```

Two things follow, and they are the whole point of doing this:

- The Laplacian — which today costs three second-order autograd passes per
  collocation batch — collapses to a **multiplication by `-|kappa|^2`**. No second
  derivatives are taken anywhere.
- The only derivative left is `dT_hat/dt`, a first derivative with respect to a
  single scalar input.

Spatial derivatives, where they are still needed, are also exact and free:
`dT/dx = iFFT(i*kx * T_hat)`. No finite differences, no autograd through space.

Note the laser does **not** appear here. In this problem it is a surface flux, not
a volumetric source, so it lives entirely in the boundary condition — which is
where the trouble starts.

## Why the BC loss cannot be transformed

The boundary conditions **must not** be pushed into Fourier space, for three
independent reasons:

1. **A face is not a mode.** A boundary condition constrains a
   measure-zero surface (`z = 0`, `z = z_max`, the four sides). In spectral space
   such a constraint couples *every* wavenumber at once — it cannot be written per
   `kappa`. The very decoupling that makes the PDE cheap is what makes the BC
   inexpressible.
2. **The transform assumes periodicity; the plate is not periodic.** An FFT wraps
   the domain, silently gluing `z = 0` (held at ambient) to `z = z_max` (under the
   laser) — precisely the two faces with the most contradictory conditions.
3. **Radiation is nonlinear.** The `sigma*eps*(T^4 - T_amb^4)` term is a product in
   real space, and a product is a *convolution* in spectral space. Transforming
   `T^4` means a triple convolution over the whole spectrum: expensive and
   badly conditioned.

**So the BCs are computed after the inverse transform, in real space.** The
reconstruction is differentiable (`torch.fft.irfftn`), so gradients flow from the
real-space boundary residuals straight back into the spectral coefficients the
network actually predicts:

```
T_hat  --(spectral)-->  L_pde                    enforced on the modes
   |
 iFFT (differentiable)
   v
   T     --(real space)-->  L_bc, L_ic, L_data    enforced on the grid

L = w_pde*L_pde + w_bc*L_bc + w_ic*L_ic + w_data*L_data
```

The normal derivatives the Neumann faces need are obtained spectrally
(`dT/dn = iFFT(i*kappa_n * T_hat)` evaluated on the face nodes) and combined with
the real-space `T` for the nonlinear flux terms. The physics is thus split by
which space each term is natural in — conduction on the modes, boundaries on the
grid — rather than forced into one representation.

## Open questions, to settle before writing code

- **Validate the premise first.** Take an FFT of one snapshot and compare the
  distribution of `|T_hat(kappa)|` against the distribution of `T(x)`. The claim
  that the spectrum is the more even target is measurable, cheap to check, and
  worth checking before anything is built on it.
- **Periodicity.** The box is not periodic. A DCT-II (which encodes a Neumann
  boundary) or a DST (Dirichlet) may be the right transform rather than a plain
  FFT; failing that, zero-padding.
- **Truncation.** How many modes are kept? The grid spacing is 0.25 mm and the beam
  radius is 1.697 mm, so the beam is well resolved — but the truncation sets both
  the output dimension and the Gibbs ringing around the peak.
- **Relation to an FNO.** A Fourier Neural Operator learns *in* the spectral domain
  but predicts in real space; this predicts the spectrum itself. Worth stating the
  difference explicitly so the two are not conflated.
