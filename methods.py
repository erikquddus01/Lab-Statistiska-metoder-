"""The mathematical steps of Botev, Grotowski and Kroese (2010)."""

import numpy as np
from scipy.fft import dct
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


# --------------------------------------------------
# 1. Gaussian kernel
# --------------------------------------------------

def gaussian_kernel(x, y, t):
    """Gaussian kernel phi(x, y; t), with bandwidth h = sqrt(t)."""
    distance = x - y
    return np.exp(-distance**2 / (2 * t)) / np.sqrt(2 * np.pi * t)


# --------------------------------------------------
# 2. Gaussian KDE
# --------------------------------------------------

def gaussian_kde(data, x_grid, t):
    """Average the Gaussian kernels centred at the observations (equation 1)."""
    density = np.zeros_like(x_grid, dtype=float)
    # Each observation contributes one kernel. Their average is the KDE.
    for observation in data:
        density += gaussian_kernel(x_grid, observation, t)
    return density / len(data)


# --------------------------------------------------
# 3. Algorithm 1 / Improved Sheather-Jones (ISJ)
# --------------------------------------------------

def derivative_functional(coefficients_squared, j, t):
    """Estimate the integral of [f^(j)(x)]^2 on the scaled interval [0, 1]."""
    # The cosine coefficients represent the binned empirical density.
    # Gaussian smoothing damps each frequency; differentiation weights it
    # by (pi*k)^j. Squaring and integrating gives the derivative functional.
    k_squared = np.arange(1, len(coefficients_squared) + 1, dtype=float)**2
    smoothing = np.exp(-k_squared * np.pi**2 * t)
    roughness = 2 * np.pi**(2 * j) * np.sum(
        k_squared**j * coefficients_squared * smoothing
    )
    if not np.isfinite(roughness) or roughness <= 0:
        raise ValueError("ISJ derivative functional is zero or non-finite.")
    return float(roughness)


def gamma_j(j, t_next, n, coefficients_squared):
    """Use t_(j+1) to calculate t_j, following equation (29)."""
    roughness_next = derivative_functional(coefficients_squared, j + 1, t_next)
    # Equation (29) needs the product 1 * 3 * ... * (2*j - 1).
    odd_product = 1
    for odd_number in range(1, 2 * j, 2):
        odd_product *= odd_number
    constant = (1 + 2**(-(j + 0.5))) / 3
    numerator = 2 * constant * odd_product / np.sqrt(2 * np.pi)
    return (numerator / (n * roughness_next))**(2 / (3 + 2 * j))


def algorithm1_isj(data, domain=None, grid_size=4096, max_iter=200):
    """Return t_pilot, t2, and the number of fixed-point iterations."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    # DCT calculations use [0, 1]. Keep the interval width so that the
    # bandwidths can be converted back to the original squared data units.
    if domain is None:
        padding = 0.05 * (np.max(data) - np.min(data))
        domain = (np.min(data) - padding, np.max(data) + padding)
    left, right = domain
    width = right - left
    unit_data = (data - left) / width
    tolerance = np.finfo(float).eps
    unit_data = np.clip(unit_data, tolerance, 1 - tolerance)
    counts, _ = np.histogram(unit_data, bins=grid_size, range=(0, 1))
    masses = counts / counts.sum()
    coefficients = dct(masses, type=2)
    coefficients_squared = (coefficients[1:] / 2)**2

    # Solve t = xi * gamma_1(gamma_2(...gamma_5(t))) (equation 30).
    # Five stages are the paper's recommendation, not a normal-reference rule.
    stages = 5
    xi = ((6 * np.sqrt(2) - 3) / 7)**(2 / 5)
    t = tolerance
    for iterations in range(1, max_iter + 1):
        t_next = t
        for j in range(stages, 0, -1):
            t_next = gamma_j(j, t_next, n, coefficients_squared)
        t_next = xi * t_next
        if not np.isfinite(t_next):
            raise ValueError("Algorithm 1 produced a non-finite bandwidth.")
        if abs(t_next - t) < tolerance:
            t = t_next
            break
        t = t_next
    else:
        raise RuntimeError("Algorithm 1 did not converge within max_iter.")

    # t_pilot builds the pilot KDE p(x).
    # t2 is an auxiliary bandwidth for the roughness calculation in Algorithm 2.
    # Stop the same recursion at gamma_2 to retain t_2 rather than t_1.
    t2 = t
    for j in range(stages, 1, -1):
        t2 = gamma_j(j, t2, n, coefficients_squared)
    t_pilot = t * width**2
    t2 = t2 * width**2
    return t_pilot, t2, iterations


# --------------------------------------------------
# 4. Pilot density
# --------------------------------------------------

# The pilot is p(x) = gaussian_kde(data, x_grid, t_pilot).
# It is only a rough first estimate used to guide the diffusion.
# This calculation appears directly in Algorithm 2 below; it needs no helper.


# --------------------------------------------------
# 5. Diffusion PDE
# --------------------------------------------------

def empirical_initial_density(data, domain, n_grid):
    """Represent the empirical sample by cell densities of total mass one."""
    left, right = domain
    edges = np.linspace(left, right, n_grid + 1)
    dx = (right - left) / n_grid
    x_grid = left + (np.arange(n_grid) + 0.5) * dx
    counts, _ = np.histogram(data, bins=edges)
    initial_density = counts / (len(data) * dx)
    return x_grid, initial_density, dx


def diffusion_rhs(g, p, dx):
    """Calculate dg/dt = 1/2 d/dx [p(x) d/dx(g/p)] on the cell grid."""
    ratio = g / p
    # Approximate the derivative of g/p at the internal cell faces.
    ratio_diff = (ratio[1:] - ratio[:-1]) / dx
    p_at_faces = (p[1:] + p[:-1]) / 2

    # Flux points towards decreasing g/p. Zero boundary flux prevents
    # probability from escaping the interval.
    flux = np.zeros(len(g) + 1)
    flux[1:-1] = -0.5 * p_at_faces * ratio_diff

    # Each cell gains incoming probability and loses outgoing probability.
    dg_dt = (flux[:-1] - flux[1:]) / dx
    return dg_dt


def solve_diffusion(initial_density, p, dx, t):
    """Evolve the empirical density from time 0 to time t."""
    # The PDE is linear because p is fixed. Column i of its matrix is the
    # rate of change when only cell i has density one. The flux calculation
    # above affects only that cell and its two neighbours: three diagonals.
    n_grid = len(p)
    diagonal = np.zeros(n_grid)
    upper = np.zeros(n_grid - 1)
    lower = np.zeros(n_grid - 1)
    for i in range(n_grid):
        g = np.zeros(n_grid)
        g[i] = 1
        dg_dt = diffusion_rhs(g, p, dx)
        diagonal[i] = dg_dt[i]
        if i > 0:
            upper[i - 1] = dg_dt[i - 1]
        if i < n_grid - 1:
            lower[i] = dg_dt[i + 1]
    operator = diags([lower, diagonal, upper], [-1, 0, 1], format="csr")

    # For dg/dt = operator @ g, the solution is exp(t * operator) @ g(0).
    # SciPy evaluates this without constructing the full matrix exponential.
    g0 = initial_density / (np.sum(initial_density) * dx)
    density = expm_multiply(t * operator, g0)

    # Check the solver output before correcting tiny floating-point errors.
    integral = np.sum(density) * dx
    if not np.all(np.isfinite(density)):
        raise ValueError("Diffusion produced NaN or infinite density values.")
    if abs(integral - 1) > 5e-3 or np.min(density) < -1e-8:
        raise ValueError(f"Invalid diffusion density: integral = {integral:.6g}.")
    density = np.maximum(density, 0)
    return density / (np.sum(density) * dx)


# --------------------------------------------------
# 6. Roughness estimate
# --------------------------------------------------

def estimate_roughness(initial_density, p, dx, t2):
    """Estimate ||Lf||^2 from the time derivative near t2 (Section 6)."""
    epsilon = max(t2 * 1e-3, 1e-7)
    # Run the diffusion at t2 and a very slightly later time. Their
    # difference approximates dg/dt = Lg, which estimates Lf at this scale.
    g_at_t2 = solve_diffusion(initial_density, p, dx, t2)
    g_after_t2 = solve_diffusion(initial_density, p, dx, t2 + epsilon)
    dg_dt = (g_after_t2 - g_at_t2) / epsilon

    # Integrating the squared derivative gives ||Lf||^2 for the final bandwidth.
    roughness = np.sum(dg_dt**2) * dx
    if not np.isfinite(roughness) or roughness <= 0:
        raise ValueError("Roughness is zero or non-finite.")
    return float(roughness)


# --------------------------------------------------
# 7. Final t_star
# --------------------------------------------------

def calculate_t_star(n, roughness):
    """Equation (23), with a(x) = p(x), hence sigma(x) = 1."""
    if not np.isfinite(roughness) or roughness <= 0:
        raise ValueError("Roughness must be positive and finite.")
    # This is the final diffusion time. It determines how long we run
    # the final diffusion, starting again from the empirical density.
    t_star = (
        1.0 /
        (2 * n * np.sqrt(np.pi) * roughness)
    ) ** (2.0 / 5.0)
    return t_star


# --------------------------------------------------
# 8. Final diffusion KDE / Algorithm 2
# --------------------------------------------------

def algorithm2_diffusion_kde(data, domain=(0.0, 1.0), n_grid=301,
                             isj_grid_size=4096):
    """Run the full method and return the density and named diagnostics."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    t_pilot, t2, iterations = algorithm1_isj(data, domain, isj_grid_size)
    x_grid, initial_density, dx = empirical_initial_density(data, domain, n_grid)

    # This is only a rough first estimate of the density.
    # It guides the diffusion and is not the final result.
    pilot = gaussian_kde(data, x_grid, t_pilot)
    # Retain the original numerical floor to avoid dividing by underflowed p.
    p = np.maximum(pilot, 1e-12)

    roughness = estimate_roughness(initial_density, p, dx, t2)
    t_star = calculate_t_star(n, roughness)
    # Start again at the raw empirical density, rather than at the pilot
    # or at g(t2). This is where the final estimator is produced.
    density = solve_diffusion(initial_density, p, dx, t_star)
    integral = np.sum(density) * dx

    # A normal dictionary keeps the arrays and diagnostics accessible by name.
    return {
        "x_grid": x_grid,
        "density": density,
        "pilot": pilot,
        "initial_density": initial_density,
        "t_pilot": t_pilot,
        "t2": t2,
        "t_star": t_star,
        "roughness": roughness,
        "integral": integral,
        "isj_iterations": iterations,
    }


# --------------------------------------------------
# 9. Integrated squared error (ISE)
# --------------------------------------------------

def calculate_ise(x_grid, density, true_density):
    """Numerically integrate the squared estimation error."""
    squared_error = (density - true_density)**2
    return float(np.trapezoid(squared_error, x_grid))


# --------------------------------------------------
# 10. Reflected estimator for Figure 1
# --------------------------------------------------

def reflected_boundary_kde(data, x_grid, t):
    """Reflection kernel on [0, 1], from equation (4)."""
    # Equation (4) sums kernels at 2*k + X_i and 2*k - X_i.
    # Include enough reflected copies that omitted Gaussian tails are negligible.
    radius = np.sqrt(2 * t * np.log(1e12))
    max_reflection = max(1, int(np.ceil((1 + radius) / 2)) + 1)
    density = np.zeros_like(x_grid, dtype=float)
    for k in range(-max_reflection, max_reflection + 1):
        density += gaussian_kde(2 * k + data, x_grid, t)
        density += gaussian_kde(2 * k - data, x_grid, t)
    return density
