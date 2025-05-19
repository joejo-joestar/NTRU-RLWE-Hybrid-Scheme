# `demo.py`

This file serves as a demonstration script to showcase the functionalities of the implemented FHE schemes, particularly the NTRU-RLWE hybrid and its comparison with other schemes.

* **`noise_comparison_demo()`**
  * **Purpose**: Demonstrates and visualizes noise comparison across different FHE schemes.
  * **Details**: Runs `FHEBenchmark._benchmark_noise_comparison` to obtain noise data. Prints a summary of this data (mean noise by scheme and operation type) and then uses `FHEBenchmark.plot_noise_comparison` to generate plots.
  * **Math Used**:
    * Calculates the mean of noise values for summarization: `results_df.groupby(['Scheme', 'Operation'])['Noise'].mean()`.
* **`main()`**
  * **Purpose**: The main entry point for the demonstration script.
  * **Details**: Executes standard performance benchmarks using `FHEBenchmark.run_benchmarks` and prints the results. It then calls `noise_comparison_demo` for the noise analysis.
  * **Math Used**: Primarily orchestrates calls to other functions that perform mathematical computations or data analysis.
