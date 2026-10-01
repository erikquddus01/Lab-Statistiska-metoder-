"""Core numerical methods for Botev, Grotowski and Kroese (2010).

The code in this module implements the ingredients used by the experiments:
manual Gaussian KDE evaluation, the paper's Improved Sheather-Jones bandwidth
recursion, and a finite-volume solver for the diffusion estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi, sqrt
from typing import Callable

import numpy as np
from scipy.fft import dct
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.special import eval_hermitenorm


Array = np.ndarray


@dataclass(frozen=True)
class ISJResult:
    """Result of Algorithm 1."""

    t_hat: float
    t2_hat: float
    converged: bool
    iterations: int
    scale_min: float
    scale_max: float


@dataclass(frozen=True)
class DiffusionKDEResult:
    """High-level result of the diffusion KDE pipeline."""

    x_grid: Array
    final_density: Array
    pilot_density: Array
    initial_density: Array
    t_hat: float
    t2_hat: float
    t_star: float
    estimated_roughness: float
    isj_converged: bool
    isj_iterations: int


def validate_sample(data: Array) -> Array:
    """Return a clean one-dimensional sample as float64."""

    sample = np.asarray(data, dtype=float).ravel()
    if sample.size < 2:
        raise ValueError("At least two observations are required.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("The sample contains NaN or infinite values.")
    return sample


def gaussian_kernel(x: Array, y: Array, t: float) -> Array:
    """Evaluate phi(x, y; t) manually for a Gaussian kernel."""

    if t <= 0:
        raise ValueError("The bandwidth parameter t must be positive.")
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z = (x_arr - y_arr) / sqrt(t)
    return np.exp(-0.5 * z * z) / sqrt(2.0 * pi * t)


def gaussian_kde(x_grid: Array, data: Array, t: float, chunk_size: int = 2048) -> Array:
    """Manual Gaussian KDE: mean_i phi(x, X_i; t)."""

    sample = validate_sample(data)
    x = np.asarray(x_grid, dtype=float).ravel()
    density = np.zeros_like(x)
    for start in range(0, sample.size, chunk_size):
        chunk = sample[start : start + chunk_size]
        density += gaussian_kernel(x[:, None], chunk[None, :], t).sum(axis=1)
    return density / sample.size


def gaussian_kernel_derivative(x: Array, y: Array, t: float, order: int) -> Array:
    """Derivative of phi(x, y; t) with respect to x.

    Uses probabilists' Hermite polynomials:
        d^m phi / dx^m = (-1)^m t^(-m/2) He_m((x-y)/sqrt(t)) phi.
    """

    if order < 0:
        raise ValueError("Derivative order must be nonnegative.")
    z = (np.asarray(x, dtype=float) - np.asarray(y, dtype=float)) / sqrt(t)
    return ((-1.0) ** order) * t ** (-0.5 * order) * eval_hermitenorm(order, z) * gaussian_kernel(x, y, t)


def _double_factorial_odd(j: int) -> int:
    """Return 1 x 3 x ... x (2j - 1)."""

    out = 1
    for value in range(1, 2 * j, 2):
        out *= value
    return out


def estimate_derivative_functional_pairwise(
    data: Array,
    derivative_order: int,
    t: float,
    chunk_size: int = 512,
) -> float:
    """Estimate ||f^(derivative_order)||^2 using equation (26).

    This direct O(N^2) implementation is useful for small diagnostic checks.
    The production ISJ implementation below uses the paper's DCT acceleration.
    """

    sample = validate_sample(data)
    if derivative_order < 1:
        raise ValueError("derivative_order must be at least 1.")
    total = 0.0
    kernel_t = 2.0 * t
    derivative_order_twice = 2 * derivative_order
    sign = (-1.0) ** derivative_order
    for start in range(0, sample.size, chunk_size):
        chunk = sample[start : start + chunk_size]
        values = gaussian_kernel_derivative(
            chunk[:, None],
            sample[None, :],
            kernel_t,
            derivative_order_twice,
        )
        total += values.sum()
    estimate = sign * total / (sample.size * sample.size)
    return float(max(estimate, np.finfo(float).tiny))


def _scale_to_unit_interval(data: Array, domain: tuple[float, float] | None) -> tuple[Array, float, float]:
    sample = validate_sample(data)
    if domain is None:
        lo = float(np.min(sample))
        hi = float(np.max(sample))
        width = hi - lo
        if width <= 0:
            raise ValueError("Cannot scale a constant sample.")
        pad = 0.05 * width
        lo -= pad
        hi += pad
    else:
        lo, hi = map(float, domain)
        if hi <= lo:
            raise ValueError("domain must satisfy max > min.")
    scaled = (sample - lo) / (hi - lo)
    eps = np.finfo(float).eps
    scaled = np.clip(scaled, eps, 1.0 - eps)
    return scaled, lo, hi


def _histogram_unit_masses(unit_data: Array, grid_size: int) -> Array:
    """Bin observations on [0, 1] as probability masses."""

    counts, _ = np.histogram(unit_data, bins=grid_size, range=(0.0, 1.0))
    masses = counts.astype(float)
    masses /= masses.sum()
    return masses


def _dct_derivative_functional(coefficients_squared: Array, derivative_order: int, t_scaled: float) -> float:
    """DCT approximation of ||f^(derivative_order)||^2 on [0, 1]."""

    if t_scaled <= 0:
        t_scaled = np.finfo(float).eps
    k_squared = np.arange(1, coefficients_squared.size + 1, dtype=float) ** 2
    weights = np.exp(-k_squared * pi * pi * t_scaled)
    functional = 2.0 * pi ** (2 * derivative_order) * np.sum(
        (k_squared**derivative_order) * coefficients_squared * weights
    )
    return float(max(functional, np.finfo(float).tiny))


def gamma_j_dct(
    j: int,
    t_next_scaled: float,
    sample_size: int,
    coefficients_squared: Array,
) -> float:
    """The recursive gamma_j map from equation (29), computed by DCT."""

    if j < 1:
        raise ValueError("j must be positive.")
    roughness_next = _dct_derivative_functional(coefficients_squared, j + 1, t_next_scaled)
    const = (1.0 + 2.0 ** (-(j + 0.5))) / 3.0
    k0 = _double_factorial_odd(j) / sqrt(2.0 * pi)
    return float((2.0 * const * k0 / (sample_size * roughness_next)) ** (2.0 / (3.0 + 2.0 * j)))


def _gamma_composition_dct(
    t_scaled: float,
    highest_j: int,
    sample_size: int,
    coefficients_squared: Array,
    lowest_j: int = 1,
) -> float:
    """Apply gamma_highest_j, ..., gamma_lowest_j."""

    out = float(t_scaled)
    for j in range(highest_j, lowest_j - 1, -1):
        out = gamma_j_dct(j, out, sample_size, coefficients_squared)
    return out


def algorithm1_isj(
    data: Array,
    stages: int = 5,
    grid_size: int = 4096,
    domain: tuple[float, float] | None = None,
    max_iter: int = 200,
    tol: float | None = None,
) -> ISJResult:
    """Algorithm 1: Improved Sheather-Jones bandwidth selector.

    The recursive derivative-functional estimates follow equations (26) and
    (29). They are evaluated with the DCT acceleration recommended in the
    paper, after linearly scaling the data to [0, 1]. Returned t values are
    converted back to the original data units.
    """

    if stages <= 2:
        raise ValueError("Algorithm 1 requires stages > 2.")
    sample = validate_sample(data)
    unit_data, lo, hi = _scale_to_unit_interval(sample, domain)
    if grid_size < 32:
        raise ValueError("grid_size must be at least 32.")
    masses = _histogram_unit_masses(unit_data, grid_size)
    coefficients = dct(masses, type=2, norm=None)
    coefficients_squared = (coefficients[1:] / 2.0) ** 2

    xi = ((6.0 * sqrt(2.0) - 3.0) / 7.0) ** (2.0 / 5.0)
    tolerance = np.finfo(float).eps if tol is None else float(tol)
    z = np.finfo(float).eps
    converged = False

    for iteration in range(1, max_iter + 1):
        gamma_value = _gamma_composition_dct(z, stages, sample.size, coefficients_squared)
        z_next = xi * gamma_value
        if abs(z_next - z) < tolerance:
            z = z_next
            converged = True
            break
        z = z_next
    else:
        iteration = max_iter

    # t2 maps from t_{l+1} to t_2, so the composition starts at gamma_l
    # and stops at gamma_2. This is the auxiliary bandwidth for ||f''||^2.
    t2_scaled = _gamma_composition_dct(z, stages, sample.size, coefficients_squared, lowest_j=2)
    scale = (hi - lo) ** 2
    return ISJResult(
        t_hat=float(z * scale),
        t2_hat=float(t2_scaled * scale),
        converged=converged,
        iterations=iteration,
        scale_min=lo,
        scale_max=hi,
    )


def make_cell_center_grid(domain: tuple[float, float], n_grid: int) -> tuple[Array, float]:
    """Create equally spaced finite-volume cell centers."""

    lo, hi = map(float, domain)
    if hi <= lo:
        raise ValueError("domain must satisfy max > min.")
    if n_grid < 8:
        raise ValueError("n_grid must be at least 8.")
    dx = (hi - lo) / n_grid
    grid = lo + (np.arange(n_grid, dtype=float) + 0.5) * dx
    return grid, dx


def empirical_initial_density(data: Array, domain: tuple[float, float], n_grid: int) -> tuple[Array, Array, float]:
    """Approximate the empirical measure with cell histogram densities."""

    sample = validate_sample(data)
    lo, hi = map(float, domain)
    edges = np.linspace(lo, hi, n_grid + 1)
    counts, _ = np.histogram(sample, bins=edges)
    dx = (hi - lo) / n_grid
    density = counts.astype(float) / (sample.size * dx)
    grid = lo + (np.arange(n_grid, dtype=float) + 0.5) * dx
    return grid, density, dx


def normalize_density(density: Array, dx: float) -> Array:
    """Clip tiny negative roundoff and renormalize a cell-centered density."""

    out = np.asarray(density, dtype=float).copy()
    out[out < 0.0] = 0.0
    total = float(out.sum() * dx)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Density cannot be normalized.")
    return out / total


def build_diffusion_operator(p: Array, dx: float, floor: float = 1e-12):
    """Build the finite-volume matrix for Lg = 1/2 d/dx[p d(g/p)/dx]."""

    pilot = np.asarray(p, dtype=float).copy()
    if not np.all(np.isfinite(pilot)):
        raise ValueError("Pilot density contains non-finite values.")
    pilot = np.maximum(pilot, floor)
    a = pilot
    n = pilot.size
    a_half = 0.5 * (a[:-1] + a[1:])
    factor = 0.5 / (dx * dx)

    main = np.zeros(n)
    upper = factor * a_half / pilot[1:]
    lower = factor * a_half / pilot[:-1]
    main[:-1] -= factor * a_half / pilot[:-1]
    main[1:] -= factor * a_half / pilot[1:]

    return diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")


def diffusion_rhs(_time: float, g: Array, operator) -> Array:
    """Right-hand side for the discretized diffusion equation."""

    return operator @ g


def solve_diffusion(initial_density: Array, pilot_density: Array, dx: float, t: float, floor: float = 1e-12) -> Array:
    """Solve the finite-volume diffusion system from 0 to t."""

    if t < 0:
        raise ValueError("Diffusion time cannot be negative.")
    g0 = normalize_density(initial_density, dx)
    if t == 0:
        return g0
    operator = build_diffusion_operator(pilot_density, dx, floor=floor)
    solution = expm_multiply(operator * t, g0)
    return normalize_density(solution, dx)


def estimate_roughness(
    initial_density: Array,
    pilot_density: Array,
    dx: float,
    t2_hat: float,
    epsilon_fraction: float = 1e-3,
    floor: float = 1e-12,
) -> tuple[float, float]:
    """Estimate ||Lf||^2 using a finite difference near t2_hat."""

    eps_t = max(float(t2_hat) * epsilon_fraction, 1e-7)
    g_t = solve_diffusion(initial_density, pilot_density, dx, t2_hat, floor=floor)
    g_t_eps = solve_diffusion(initial_density, pilot_density, dx, t2_hat + eps_t, floor=floor)
    time_derivative = (g_t_eps - g_t) / eps_t
    roughness = float(np.sum(time_derivative * time_derivative) * dx)
    return max(roughness, np.finfo(float).tiny), eps_t


def calculate_t_star(sample_size: int, roughness: float) -> float:
    """Compute the final diffusion time for a(x)=p(x), where sigma(x)=1."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if roughness <= 0 or not np.isfinite(roughness):
        raise ValueError("roughness must be positive and finite.")
    return float((1.0 / (2.0 * sample_size * sqrt(pi) * roughness)) ** (2.0 / 5.0))


def algorithm2_diffusion_kde(
    data: Array,
    domain: tuple[float, float] = (0.0, 1.0),
    n_grid: int = 301,
    isj_grid_size: int = 4096,
    pilot_floor: float = 1e-12,
) -> DiffusionKDEResult:
    """Run Algorithm 1 followed by the plug-in diffusion KDE."""

    sample = validate_sample(data)
    isj = algorithm1_isj(sample, stages=5, grid_size=isj_grid_size, domain=domain)
    x_grid, initial_density, dx = empirical_initial_density(sample, domain, n_grid)
    pilot_density = gaussian_kde(x_grid, sample, isj.t_hat)
    pilot_density = np.maximum(pilot_density, pilot_floor)
    roughness, _ = estimate_roughness(initial_density, pilot_density, dx, isj.t2_hat, floor=pilot_floor)
    t_star = calculate_t_star(sample.size, roughness)
    final_density = solve_diffusion(initial_density, pilot_density, dx, t_star, floor=pilot_floor)
    validate_density(final_density, dx, name="final diffusion density")
    validate_density(pilot_density, dx, name="pilot density", require_unit_integral=False)
    return DiffusionKDEResult(
        x_grid=x_grid,
        final_density=final_density,
        pilot_density=pilot_density,
        initial_density=initial_density,
        t_hat=isj.t_hat,
        t2_hat=isj.t2_hat,
        t_star=t_star,
        estimated_roughness=roughness,
        isj_converged=isj.converged,
        isj_iterations=isj.iterations,
    )


def reflected_boundary_kde(
    x_grid: Array,
    data: Array,
    t: float,
    truncation_tol: float = 1e-12,
    chunk_size: int = 512,
) -> Array:
    """Reflected estimator on [0, 1] using the kernel kappa from equation (4)."""

    sample = validate_sample(data)
    x = np.asarray(x_grid, dtype=float).ravel()
    radius = sqrt(2.0 * t * np.log(1.0 / truncation_tol))
    max_reflection = max(1, int(ceil((1.0 + radius) / 2.0)) + 1)
    density = np.zeros_like(x)
    for start in range(0, sample.size, chunk_size):
        chunk = sample[start : start + chunk_size]
        partial = np.zeros((x.size, chunk.size), dtype=float)
        for k in range(-max_reflection, max_reflection + 1):
            partial += gaussian_kernel(x[:, None], (2.0 * k + chunk)[None, :], t)
            partial += gaussian_kernel(x[:, None], (2.0 * k - chunk)[None, :], t)
        density += partial.sum(axis=1)
    return density / sample.size


def calculate_ise(x_grid: Array, estimated_density: Array, true_density: Array) -> float:
    """Integrated squared error using numerical quadrature."""

    x = np.asarray(x_grid, dtype=float)
    estimate = np.asarray(estimated_density, dtype=float)
    truth = np.asarray(true_density, dtype=float)
    return float(np.trapezoid((estimate - truth) ** 2, x))


def beta_1_4_density(x: Array) -> Array:
    """Beta(1, 4) density: f(x)=4(1-x)^3 on [0, 1]."""

    values = np.asarray(x, dtype=float)
    density = np.zeros_like(values)
    mask = (values >= 0.0) & (values <= 1.0)
    density[mask] = 4.0 * (1.0 - values[mask]) ** 3
    return density


def validate_density(
    density: Array,
    dx: float,
    name: str = "density",
    require_unit_integral: bool = True,
    integral_tol: float = 5e-3,
) -> None:
    """Check finite values, small negative errors, and optional normalization."""

    values = np.asarray(density, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.min(values) < -1e-8:
        raise ValueError(f"{name} has negative values below numerical tolerance.")
    if require_unit_integral:
        integral = float(np.sum(values) * dx)
        if abs(integral - 1.0) > integral_tol:
            raise ValueError(f"{name} integrates to {integral:.6g}, not approximately 1.")


def sample_beta_1_4(rng: np.random.Generator, sample_size: int) -> Array:
    """Generate from Beta(1, 4)."""

    return rng.beta(1.0, 4.0, size=sample_size)
