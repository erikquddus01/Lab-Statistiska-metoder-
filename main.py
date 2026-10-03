"""Regenerate all results for the diffusion KDE project."""

from experiments import reproduce_figure1
from experiments import run_sample_size_experiment


def main():
    reproduce_figure1()
    run_sample_size_experiment()


if __name__ == "__main__":
    main()
