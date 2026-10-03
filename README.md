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

Figure 1 uses the fixed random seed `2935`. The sample-size experiment uses
`2936`, so its samples do not depend on whether Figure 1 was run first.

## Reading the Code

- `main.py` calls the two experiments.
- `methods.py` follows the presentation order: Gaussian kernel, Gaussian KDE,
  ISJ, pilot, diffusion PDE, roughness, final bandwidth, final estimate, ISE,
  and the reflected estimator.
- `experiments.py` generates samples, calculates errors, and saves CSVs and plots.
- `methodology.tex` describes the mathematical method.

Start with `algorithm2_diffusion_kde` to see the complete sequence:

```text
data -> Algorithm 1 -> t_pilot and t2
t_pilot -> Gaussian KDE -> pilot p(x)
empirical density -> diffusion at t2 -> roughness
roughness -> t_star
empirical density -> diffusion at t_star -> final density
```

`t_pilot` is the report's `t_hat`, and `t2` is its `t2_hat`. These are
squared bandwidths (diffusion times); the ordinary bandwidth is `sqrt(t)`.
Algorithm 1 returns `(t_pilot, t2, iterations)`. Algorithm 2 returns a normal
dictionary with `x_grid`, `density`, `pilot`, `initial_density`, `t_pilot`,
`t2`, `t_star`, `roughness`, `integral`, and `isj_iterations`.

## Outputs

Running `python main.py` creates:

- `results/figure1_reproduction.pdf`
- `results/sample_size_experiment.pdf`
- `results/results.csv`
- `results/individual_ise.csv`

`results/results.csv` contains the summary mean, standard deviation, minimum,
and maximum ISE for each sample size. `results/individual_ise.csv` contains
the individual repetition-level ISE and bandwidth diagnostics. Its columns
are `sample_size`, `repetition`, `ise`, `t_pilot`, `t2`, `t_star`, `roughness`,
`density_integral`, and `isj_iterations`. The old `t_hat`, `t2_hat`, and
`estimated_roughness` columns have these clearer names. Nonconvergence now
stops the experiment with an error instead of being recorded and ignored.

## Numerical Choices

- Figure 1 uses the paper's fixed bandwidth `sqrt(t)=0.05248` and does not run
  Algorithm 1.
- Algorithm 1 uses `l=5` stages. Derivative functionals are evaluated with the
  DCT acceleration recommended by the paper, implemented here from the
  recursive equations.
- The diffusion PDE is discretized on an equally spaced finite-volume grid with
  no-flux boundary conditions. `diffusion_rhs` shows the ratio `g/p`, face
  derivatives, probability fluxes, and the resulting density changes.
  `solve_diffusion` builds the three-diagonal matrix from these changes and
  retains SciPy's `expm_multiply` solver for the linear system.
- For the main diffusion experiment, `a(x)=p(x)`, so `sigma(x)=1`.
- The empirical initial condition is represented by histogram cell densities
  whose total mass is one.
- The pilot density floor is `1e-12`, as in the original implementation, to avoid
  floating-point underflow in divisions by `p(x)`.
- Roughness is estimated using the plug-in idea from the paper:
  `||Lf||^2` is approximated by a finite difference of the numerical diffusion
  solution near `t2`. The time step remains `max(t2 * 1e-3, 1e-7)`.
- The solver checks mass, finiteness, and negative values before correcting
  tiny roundoff errors. Invalid roughness and ISJ convergence failures stop
  the calculation. Each repetition prints the bandwidths, roughness, and
  final density integral.
- ISE retains the original trapezoidal quadrature; density mass uses
  `sum(density) * dx` because the PDE grid contains cell centres.

## Experiments

The sample-size variation uses the target density

```text
f(x) = 4(1-x)^3, 0 <= x <= 1
```

which is `Beta(1,4)`. It runs sample sizes
`[50, 100, 250, 500, 1000, 5000]` with `R=20` independent repetitions by
default.
