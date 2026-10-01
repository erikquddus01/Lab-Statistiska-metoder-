"""Simulation and plotting code for the diffusion KDE assignment."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from methods import (
    algorithm2_diffusion_kde,
    beta_1_4_density,
    calculate_ise,
    gaussian_kde,
    reflected_boundary_kde,
    sample_beta_1_4,
)


RESULTS_DIR = Path("results")
RANDOM_SEED = 2935


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def reproduce_figure1(seed: int = RANDOM_SEED) -> dict[str, float]:
    """Reproduce the paper's boundary-bias Figure 1 experiment."""

    _ensure_results_dir()
    rng = np.random.default_rng(seed)
    n = 1000
    data = sample_beta_1_4(rng, n)
    t = 0.05248**2
    x_grid = np.linspace(0.0, 1.0, 700)
    true_density = beta_1_4_density(x_grid)
    ordinary = gaussian_kde(x_grid, data, t)
    reflected = reflected_boundary_kde(x_grid, data, t)

    ordinary_ise = calculate_ise(x_grid, ordinary, true_density)
    reflected_ise = calculate_ise(x_grid, reflected, true_density)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x_grid, true_density, color="black", linewidth=2.1, label="True Beta(1,4)")
    ax.plot(x_grid, ordinary, color="#be3a34", linewidth=1.7, label="Gaussian KDE")
    ax.plot(x_grid, reflected, color="#2779a7", linewidth=1.7, label="Reflected diffusion")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    ax.set_title("Boundary bias for Beta(1,4), N=1000")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure1_reproduction.pdf")
    plt.close(fig)

    return {
        "ordinary_ise": ordinary_ise,
        "reflected_ise": reflected_ise,
        "bandwidth_t": t,
    }


def run_sample_size_experiment(
    sample_sizes: list[int] | None = None,
    repetitions: int = 20,
    seed: int = RANDOM_SEED,
    n_grid: int = 251,
    isj_grid_size: int = 2048,
) -> list[dict[str, float]]:
    """Run the sample-size variation and save summary plus individual results."""

    _ensure_results_dir()
    sizes = sample_sizes or [50, 100, 250, 500, 1000, 5000]
    rng = np.random.default_rng(seed + 1)
    individual_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for sample_size in sizes:
        ise_values: list[float] = []
        for repetition in range(1, repetitions + 1):
            data = sample_beta_1_4(rng, sample_size)
            result = algorithm2_diffusion_kde(
                data,
                domain=(0.0, 1.0),
                n_grid=n_grid,
                isj_grid_size=isj_grid_size,
            )
            true_density = beta_1_4_density(result.x_grid)
            ise = calculate_ise(result.x_grid, result.final_density, true_density)
            ise_values.append(ise)
            individual_rows.append(
                {
                    "sample_size": sample_size,
                    "repetition": repetition,
                    "ise": ise,
                    "t_hat": result.t_hat,
                    "t2_hat": result.t2_hat,
                    "t_star": result.t_star,
                    "estimated_roughness": result.estimated_roughness,
                    "isj_converged": int(result.isj_converged),
                    "isj_iterations": result.isj_iterations,
                }
            )
            print(
                f"N={sample_size:5d} rep={repetition:02d}/{repetitions} "
                f"ISE={ise:.6g} t*={result.t_star:.6g}"
            )

        summary_rows.append(
            {
                "sample_size": sample_size,
                "repetitions": repetitions,
                "mean_ise": mean(ise_values),
                "std_ise": stdev(ise_values) if len(ise_values) > 1 else 0.0,
                "min_ise": min(ise_values),
                "max_ise": max(ise_values),
            }
        )

    with (RESULTS_DIR / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (RESULTS_DIR / "individual_ise.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(individual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(individual_rows)

    _plot_sample_size_summary(summary_rows)
    return summary_rows


def _plot_sample_size_summary(summary_rows: list[dict[str, float]]) -> None:
    sample_sizes = np.array([row["sample_size"] for row in summary_rows], dtype=float)
    means = np.array([row["mean_ise"] for row in summary_rows], dtype=float)
    stds = np.array([row["std_ise"] for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.errorbar(
        sample_sizes,
        means,
        yerr=stds,
        marker="o",
        linewidth=1.8,
        capsize=4,
        color="#2f6f4e",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size N")
    ax.set_ylabel("mean ISE")
    ax.set_title("Diffusion KDE sample-size experiment")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_size_experiment.pdf")
    plt.close(fig)


def run_all() -> None:
    """Regenerate every required result."""

    figure1 = reproduce_figure1()
    print(
        "Figure 1 ISE: "
        f"ordinary={figure1['ordinary_ise']:.6g}, "
        f"reflected={figure1['reflected_ise']:.6g}"
    )
    rows = run_sample_size_experiment()
    print("Sample-size summary:")
    for row in rows:
        print(
            f"N={row['sample_size']:5.0f} "
            f"mean ISE={row['mean_ise']:.6g} "
            f"std={row['std_ise']:.6g}"
        )


if __name__ == "__main__":
    run_all()
