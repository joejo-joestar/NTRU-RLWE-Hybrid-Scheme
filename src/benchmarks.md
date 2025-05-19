# `benchmarks.py`

This file is designed to benchmark the performance and noise characteristics of different Fully Homomorphic Encryption (FHE) schemes, including the custom NTRU-RLWE hybrid scheme, and compare them against established schemes like CKKS and BFV from the `tenseal` library. It also includes utilities for visualizing these benchmark results.

**Class `FHEBenchmark`**

This class encapsulates all benchmarking logic as static methods.

* **`run_benchmarks(iterations=3)`**
  * **Purpose**: Orchestrates the execution of all defined performance and noise benchmarks.
  * **Details**: Calls internal static methods `_benchmark_ckks`, `_benchmark_bfv`, `_benchmark_ntru_rlwe`, and `_benchmark_noise_comparison`. It collects the results (operation times, feature support, noise levels) into a pandas DataFrame for further analysis and plotting.
  * **Math Used**: No direct complex math, but relies on aggregation of numerical results (timings, noise values) returned by other benchmark methods.
* **`plot_operation_times(results_df, save_dir="results")`**
  * **Purpose**: Generates and saves a bar chart comparing the average times for key FHE operations (encryption, homomorphic addition, homomorphic multiplication) across the benchmarked schemes.
  * **Details**: Uses `matplotlib` and `seaborn` for plotting. The input `results_df` contains the timing data.
  * **Math Used**:
    * Implicitly uses the mean of timings (calculated during benchmarking) for bar heights.
    * Pandas `melt` function is used for data reshaping, which is a data manipulation technique.
* **`test_multiparty(iterations=3)`**
  * **Purpose**: Intended to benchmark or test multiparty decryption support for the NTRU-RLWE hybrid scheme.
  * **Details**: It simulates a scenario with three parties. Party 0 encrypts a message. Each party performs a partial decryption of the ciphertext. These partials are then combined to attempt to retrieve the original message. The code contains a comment "NOTE: Doesnt work!!", **indicating** incompleteness.
  * **Math Used**:
    * Polynomial generation (`generate_poly`).
    * Encryption using the hybrid scheme: `parties.encrypt(m)`.
    * Partial decryption by each party: `p.decrypt(ct, partial=True)`.
    * Simulated combination of partial decryptions: $\text{combined\_sum} = \sum_{i} partial_i$
This is done using `np.sum(partials, axis=0)`.
    * Modular reduction of the combined sum: $combined = \text{combined\_sum} \pmod p$
This is `poly_mod(combined_sum, params["p"])`.
  * **Simulated Aspect**: The method of combining partial decryptions (`np.sum(partials, axis=0)`) is a highly simplified placeholder. Real multiparty FHE decryption involves complex cryptographic protocols tailored to how the secret key shares are used and how the partially decrypted components (which are themselves often polynomials or ring elements) are mathematically combined to reconstruct the plaintext without revealing the key shares. This usually involves specific algebraic properties of the scheme (e.g., Shamir's Secret Sharing applied to key components, or specific homomorphic properties of the decryption function).
* **`_benchmark_ckks(iterations)`**
  * **Purpose**: Benchmarks the performance (encryption, addition, multiplication times) of the CKKS (Cheon-Kim-Kim-Song) FHE scheme.
  * **Details**: Uses the `tenseal` library for CKKS implementation. CKKS is optimized for approximate arithmetic on real or complex numbers.
  * **Math Used**:
    * Generation of random vectors (`np.random.randn`) as plaintexts.
    * CKKS operations (encryption, addition, multiplication) provided by `tenseal`. These involve computations in polynomial rings with complex numbers, often using specific encoding/decoding techniques (like canonical embedding) and careful noise management for approximate results. The underlying math includes polynomial arithmetic (addition, multiplication) modulo a cyclotomic polynomial and a large integer modulus $q$.
* **`_benchmark_bfv(iterations)`**
  * **Purpose**: Benchmarks the performance of the BFV (Brakerski/Fan-Vercauteren) FHE scheme.
  * **Details**: Uses `tenseal` for BFV. BFV is designed for exact arithmetic over finite fields or integers.
  * **Math Used**:
    * Generation of random **integer** vectors (`np.random.randint`) as plaintexts.
    * BFV operations from `tenseal`. These involve polynomial arithmetic (addition, multiplication) in a polynomial ring $R_q = \mathbb{Z}_q[x]/(x^N+1)$, with plaintexts in $R_p = \mathbb{Z}_p[x]/(x^N+1)$.
* **`_benchmark_ntru_rlwe(iterations)`**
  * **Purpose**: Benchmarks the performance of the custom NTRU-RLWE hybrid scheme implemented in `hybrid.py`.
  * **Details**: Measures encryption, addition, and multiplication times for the custom scheme.
  * **Math Used**: Relies on the mathematical operations defined within the `NTRU_RLWE_Hybrid` class (see `hybrid.py` detailed below).
* **`_benchmark_noise_growth(iterations=3)`**
  * **Purpose**: Benchmarks how noise accumulates in ciphertexts after repeated homomorphic operations, focusing on the NTRU-RLWE scheme.
  * **Details**: Calls `scheme.noise_growth_benchmark` for the NTRU-RLWE scheme. Notes that similar implementation for CKKS/BFV could be added.
  * **Math Used**: Relies on the noise estimation and operational methods of the FHE schemes themselves (particularly `NTRU_RLWE_Hybrid.noise_growth_benchmark`).
* **`_estimate_ckks_noise(encrypted)`**
  * **Purpose**: Attempts to estimate the noise in a CKKS ciphertext obtained from `tenseal`.
  * **Details**: Tries to use the `encrypted.noise()` method if available. As a fallback, it calculates the standard deviation of the decrypted values and scales it by an arbitrary factor (1000).
  * **Math Used**:
    * Standard deviation: $\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N}}$.
    * Arbitrary scaling: `value * 1000`.
  * **Simulated Aspect**: The fallback calculation `np.std(encrypted.decrypt()) * 1000` is a rough proxy. Actual CKKS noise refers to the error relative to the message scale. The `tenseal` library's `noise()` or similar (if available) provides a more scheme-accurate measure.
* **`_estimate_bfv_noise(encrypted)`**
  * **Purpose**: Estimates noise in a BFV ciphertext from `tenseal`.
  * **Details**: Since `tenseal` might not directly expose BFV noise, it uses a proxy: the standard deviation of the difference between decrypted values and their rounded integer counterparts.
  * **Math Used**:
    * Rounding: `np.round(values)`.
    * Difference: `decrypted_values - np.round(decrypted_values)`.
    * Standard deviation of these differences.
  * **Simulated Aspect**: This measures the magnitude of the fractional parts of decrypted coefficients, which can be related to how close the underlying plaintext (plus error) is to overflowing the plaintext modulus or how much error has accumulated beyond the integer message.
* **`_benchmark_noise_comparison(iterations=3)`**
  * **Purpose**: Compares noise characteristics across NTRU-RLWE, CKKS, and BFV for encryption, addition, and multiplication.
  * **Details**: Initializes scheme contexts with specific parameters (polynomial modulus degree, coefficient moduli, plaintext modulus). Uses scheme-specific noise estimation methods (`scheme.estimate_noise` for NTRU-RLWE, and the `_estimate_ckks_noise`, `_estimate_bfv_noise` proxies for the others).
  * **Math Used**: The mathematics involved in the respective noise estimation functions of each scheme. Parameter choices like polynomial degree ($N$), ciphertext modulus ($q$), and plaintext modulus ($p$) are fundamental to FHE security and correctness.
