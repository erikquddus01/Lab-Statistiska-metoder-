# Diffusion KDE Project

This project implements the main one-dimensional ideas from Botev, Grotowski
and Kroese (2010), *Kernel Density Estimation via Diffusion*.

The implementation does not call prebuilt density estimators such as
`scipy.stats.gaussian_kde`, `sklearn.neighbors.KernelDensity`, or an existing
diffusion-KDE/ISJ package. The Gaussian kernel, Gaussian KDE, Improved
Sheather-Jones recursion, finite-volume diffusion operator, roughness
estimation, and ISE calculations are implemented directly in the project.

## Requirements

Use Python 3.11 or newer with:

- `numpy`
- `scipy`
- `matplotlib`

## Run

Regenerate all results with:

```bash
python main.py
```

The script uses the fixed random seed `2935`.

## Outputs

Running `python main.py` creates:

- `results/figure1_reproduction.pdf`
- `results/sample_size_experiment.pdf`
- `results/results.csv`
- `results/individual_ise.csv`

`results/results.csv` contains the summary mean, standard deviation, minimum,
and maximum ISE for each sample size. `results/individual_ise.csv` contains
the individual repetition-level ISE and bandwidth diagnostics.

## Numerical Choices

- Figure 1 uses the paper's fixed bandwidth `sqrt(t)=0.05248` and does not run
  Algorithm 1.
- Algorithm 1 uses `l=5` stages. Derivative functionals are evaluated with the
  DCT acceleration recommended by the paper, implemented here from the
  recursive equations.
- The diffusion PDE is discretized on an equally spaced finite-volume grid with
  no-flux boundary conditions.
- For the main diffusion experiment, `a(x)=p(x)`, so `sigma(x)=1`.
- The empirical initial condition is represented by histogram cell densities
  whose total mass is one.
- The pilot density floor is named explicitly as `pilot_floor=1e-12` to avoid
  floating-point underflow in divisions by `p(x)`.
- Roughness is estimated using the plug-in idea from the paper:
  `||Lf||^2` is approximated by a finite difference of the numerical diffusion
  solution near `t2_hat`.

## Experiments

The sample-size variation uses the target density

```text
f(x) = 4(1-x)^3, 0 <= x <= 1
```

which is `Beta(1,4)`. It runs sample sizes
`[50, 100, 250, 500, 1000, 5000]` with `R=20` independent repetitions by
default.
