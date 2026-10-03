"""Generate data, run the methods, and save the experimental results."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods import (
    algorithm2_diffusion_kde,
    calculate_ise,
    gaussian_kde,
    reflected_boundary_kde,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_SEED = 2935


def reproduce_figure1(seed=RANDOM_SEED):
    """Compare ordinary and reflected KDE using the paper's given bandwidth."""
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    # Figure 1 uses Beta(1,4) data and a fixed bandwidth, not Algorithm 1.
    n = 1000
    data = rng.beta(1, 4, size=n)
    h = 0.05248
    t = h**2
    x_grid = np.linspace(0, 1, 700)
    true_density = 4 * (1 - x_grid)**3
    ordinary = gaussian_kde(data, x_grid, t)
    reflected = reflected_boundary_kde(data, x_grid, t)

    ordinary_ise = calculate_ise(x_grid, ordinary, true_density)
    reflected_ise = calculate_ise(x_grid, reflected, true_density)
    print(f"Figure 1: h = {h}, t = {t:.8g}")
    print(f"  ordinary ISE  = {ordinary_ise:.6g}")
    print(f"  reflected ISE = {reflected_ise:.6g}")

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x_grid, true_density, color="black", linewidth=2.1, label="True Beta(1,4)")
    ax.plot(x_grid, ordinary, color="#be3a34", linewidth=1.7, label="Gaussian KDE")
    ax.plot(x_grid, reflected, color="#2779a7", linewidth=1.7, label="Reflected diffusion")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Boundary bias for Beta(1,4), N=1000")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure1_reproduction.pdf")
    plt.close(fig)
    return {"ordinary_ise": ordinary_ise, "reflected_ise": reflected_ise,
            "bandwidth_t": t}


def run_sample_size_experiment(sample_sizes=None, repetitions=20,
                               seed=RANDOM_SEED, n_grid=251, isj_grid_size=2048):
    """Repeat the diffusion estimate for each N and summarize its ISE."""
    RESULTS_DIR.mkdir(exist_ok=True)
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 5000]
    # Use a separate fixed seed so Figure 1 does not affect these samples.
    rng = np.random.default_rng(seed + 1)
    individual_rows = []
    summary_rows = []

    # Repeat for each sample size to see how error changes with more data.
    for n in sample_sizes:
        ise_values = []
        for repetition in range(1, repetitions + 1):
            data = rng.beta(1, 4, size=n)
            estimate = algorithm2_diffusion_kde(
                data, n_grid=n_grid, isj_grid_size=isj_grid_size
            )
            x_grid = estimate["x_grid"]
            density = estimate["density"]
            true_density = 4 * (1 - x_grid)**3
            ise = calculate_ise(x_grid, density, true_density)
            ise_values.append(ise)

            # Keep bandwidths and roughness alongside the error so each
            # repetition can be checked and discussed in the presentation.
            individual_rows.append({
                "sample_size": n,
                "repetition": repetition,
                "ise": ise,
                "t_pilot": estimate["t_pilot"],
                "t2": estimate["t2"],
                "t_star": estimate["t_star"],
                "roughness": estimate["roughness"],
                "density_integral": estimate["integral"],
                "isj_iterations": estimate["isj_iterations"],
            })
            print(f"N={n}, repetition={repetition}/{repetitions}, ISE={ise:.6g}")
            print("  Algorithm 1:")
            print(f"    t_pilot = {estimate['t_pilot']:.6g}")
            print(f"    t2      = {estimate['t2']:.6g}")
            print("  Algorithm 2:")
            print(f"    roughness = {estimate['roughness']:.6g}")
            print(f"    t_star    = {estimate['t_star']:.6g}")
            print(f"    integral of final density = {estimate['integral']:.12g}")

        # The standard deviation describes variation between independent
        # samples. ddof=1 gives the sample standard deviation, as before.
        mean_ise = float(np.mean(ise_values))
        std_ise = float(np.std(ise_values, ddof=1)) if repetitions > 1 else 0.0
        summary_rows.append({
            "sample_size": n,
            "repetitions": repetitions,
            "mean_ise": mean_ise,
            "std_ise": std_ise,
            "min_ise": min(ise_values),
            "max_ise": max(ise_values),
        })
        print(f"N={n}: mean ISE = {mean_ise:.6g}, std = {std_ise:.6g}")

    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (RESULTS_DIR / "individual_ise.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(individual_rows[0]))
        writer.writeheader()
        writer.writerows(individual_rows)

    # Plot the mean error and its spread against sample size.
    means = [row["mean_ise"] for row in summary_rows]
    stds = [row["std_ise"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(sample_sizes, means, yerr=stds, marker="o", linewidth=1.8,
                capsize=4, color="#2f6f4e")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size N")
    ax.set_ylabel("mean ISE")
    ax.set_title("Diffusion KDE sample-size experiment")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_size_experiment.pdf")
    plt.close(fig)
    return summary_rows
