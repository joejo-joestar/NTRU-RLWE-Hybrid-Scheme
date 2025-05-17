"""
NTRU-RLWE Hybrid Scheme Demonstration
"""

from benchmarks import FHEBenchmark
# from hybrid import NTRU_RLWE_Hybrid
# from utils import visualize_noise_distribution, calculate_operational_depth


def noise_comparison_demo():
    """Demonstrate noise comparison across schemes"""
    print("\n=== Noise Comparison Across FHE Schemes ===")

    # Run noise comparison benchmarks
    noise_results = FHEBenchmark._benchmark_noise_comparison(iterations=3)

    # Print summary
    print("\nNoise Comparison Results:")
    print(
        noise_results.groupby(["scheme", "operation"])["noise"]
        .mean()
        .unstack()
        .to_markdown()
    )

    # Generate comparison plots
    FHEBenchmark.plot_noise_comparison(noise_results)


def main():
    print("\n=== NTRU-RLWE Hybrid Scheme Evaluation ===")

    # Run standard benchmarks
    benchmark_df = FHEBenchmark.run_benchmarks(iterations=5)
    print("\nBenchmark Results:")
    print(benchmark_df.to_markdown(tablefmt="grid"))

    # Run noise comparison demo
    noise_comparison_demo()

    print("\nEvaluation complete. Results saved to 'results/' directory")


if __name__ == "__main__":
    main()
