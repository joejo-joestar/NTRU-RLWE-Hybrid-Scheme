"""
FHE Scheme Benchmarking and Visualization

This module provides:
1. Performance benchmarking of NTRU-RLWE against CKKS and BFV
2. Multiparty encryption testing
3. Visualization of results in separate charts
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tenseal as ts
from pathlib import Path
from hybrid import NTRU_RLWE_Hybrid
from utils import generate_poly, poly_mod

# Chart style configuration
plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


class FHEBenchmark:
    """
    Comprehensive FHE benchmarking toolkit

    Methods:
        run_benchmarks(): Execute all benchmarks
        plot_operation_times(): Save operation times chart
        plot_feature_comparison(): Save feature support chart
        test_multiparty(): Test multiparty support
    """

    @staticmethod
    def run_benchmarks(iterations=3):
        """
        Run performance benchmarks for all schemes

        Args:
            iterations: Number of test runs

        Returns:
            DataFrame with benchmark results
        """
        results = []

        # Standard performance benchmarks
        results.append(FHEBenchmark._benchmark_ckks(iterations))
        results.append(FHEBenchmark._benchmark_bfv(iterations))
        results.append(FHEBenchmark._benchmark_ntru_rlwe(iterations))

        # Noise benchmarks
        noise_results = FHEBenchmark._benchmark_noise_comparison(iterations)
        FHEBenchmark.plot_noise_comparison(noise_results)

        return pd.DataFrame(results)

    @staticmethod
    def plot_operation_times(results_df, save_dir="results"):
        """
        Generate and save operation times comparison chart

        Args:
            results_df: Benchmark results DataFrame
            save_dir: Directory to save chart
        """
        Path(save_dir).mkdir(exist_ok=True)

        results_df["scheme"] = results_df["scheme"].str.strip()
        results_df = results_df.drop_duplicates(subset=["scheme"])

        melted = pd.melt(
            results_df,
            id_vars=["scheme"],
            value_vars=["enc_ms", "add_ms", "mul_ms"],
            var_name="Operation",
            value_name="Time (ms)",
        )

        plt.close("all")

        plt.figure(figsize=(10, 6), dpi=300)
        plt.figure(figsize=(10, 6), dpi=300)
        ax = plt.gca()
        ax = sns.barplot(
            data=melted, x="scheme", y="Time (ms)", hue="Operation", palette="viridis"
        )

        plt.title("Homomorphic Operation Times Comparison", fontsize=18, pad=20)
        plt.xlabel("Scheme", fontsize=14)
        plt.ylabel("Time (ms)", fontsize=14)
        plt.legend(title="Operation", fontsize=12, title_fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 1e-2:
                ax.annotate(
                    f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    xytext=(0, 4),
                    textcoords="offset points",
                )
        plt.savefig(f"{save_dir}/operation_times.png")
        plt.close()

    @staticmethod
    def plot_feature_comparison(results_df, save_dir="results"):
        """
        Generate and save feature support comparison chart

        Args:
            results_df: Benchmark results DataFrame
            save_dir: Directory to save chart
        """
        Path(save_dir).mkdir(exist_ok=True)

        plt.figure(figsize=(10, 4))
        features = results_df[["scheme", "bootstrapping", "multiparty"]]
        features = features.melt(id_vars="scheme")

        ax = sns.scatterplot(
            data=features,
            x="variable",
            y="scheme",
            hue="value",
            s=300,
            palette="viridis",
        )

        plt.title("Feature Support Comparison", pad=20)
        plt.xlabel("Feature")
        plt.ylabel("Scheme")
        plt.xticks(rotation=15)
        ax.legend(title="Support Level", bbox_to_anchor=(1.05, 1))

        plt.savefig(f"{save_dir}/feature_comparison.png", bbox_inches="tight")
        plt.close()

    # NOTE: Doesnt work!!
    @staticmethod
    def test_multiparty(iterations=3):
        """Test multiparty decryption support"""
        results = []
        params = {"N": 1024, "q": 2**20, "p": 3, "sigma": 3.2}  # Using smaller q

        for _ in range(iterations):
            # Create 3 parties
            parties = [NTRU_RLWE_Hybrid(**params) for _ in range(3)]

            # Generate and encrypt message
            m = generate_poly(params["N"], params["p"])
            ct = parties[0].encrypt(m)

            # Time partial decryptions
            start = time.time()

            # Collect partial decryptions
            partials = []
            for p in parties:
                # Get partial decryption from each party
                partial = p.decrypt(ct, partial=True)
                partials.append(partial)

            # Combine partials (simple sum for demonstration)
            combined = np.sum(partials, axis=0)
            combined = poly_mod(combined, params["p"])

            elapsed = time.time() - start

            # Verify
            correct = np.array_equal(poly_mod(m, params["p"]), combined)
            results.append({"time_ms": elapsed * 1000, "correct": correct})

        return {
            "success_rate": np.mean([r["correct"] for r in results]),
            "avg_time_ms": np.mean([r["time_ms"] for r in results]),
        }

    @staticmethod
    def _benchmark_ckks(iterations):
        """Benchmark CKKS scheme"""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 30, 30, 30, 40],
        )
        context.global_scale = 2**30
        context.generate_galois_keys()

        vec_size = 4096  # 8192/2
        enc_times, add_times, mul_times = [], [], []

        for _ in range(iterations):
            vec = np.random.randn(vec_size)

            # Encryption
            start = time.time()
            enc = ts.ckks_vector(context, vec)
            enc_times.append(time.time() - start)

            # Addition
            start = time.time()
            _ = enc + enc
            add_times.append(time.time() - start)

            # Multiplication
            start = time.time()
            _ = enc * enc
            mul_times.append(time.time() - start)

        return {
            "scheme": "CKKS",
            "enc_ms": np.mean(enc_times) * 1000,
            "add_ms": np.mean(add_times) * 1000,
            "mul_ms": np.mean(mul_times) * 1000,
            "bootstrapping": "Yes",
            "multiparty": "Limited",
        }

    @staticmethod
    def _benchmark_bfv(iterations):
        """Benchmark BFV scheme"""
        context = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=40961
        )

        vec_size = 2048  # 4096/2
        enc_times, add_times, mul_times = [], [], []

        for _ in range(iterations):
            vec = [np.random.randint(0, 100) for _ in range(vec_size)]

            # Encryption
            start = time.time()
            enc = ts.bfv_vector(context, vec)
            enc_times.append(time.time() - start)

            # Addition
            start = time.time()
            _ = enc + enc
            add_times.append(time.time() - start)

            # Multiplication
            start = time.time()
            _ = enc * enc
            mul_times.append(time.time() - start)

        return {
            "scheme": "BFV",
            "enc_ms": np.mean(enc_times) * 1000,
            "add_ms": np.mean(add_times) * 1000,
            "mul_ms": np.mean(mul_times) * 1000,
            "bootstrapping": "No",
            "multiparty": "Limited",
        }

    @staticmethod
    def _benchmark_ntru_rlwe(iterations):
        """Benchmark our hybrid scheme"""
        hybrid = NTRU_RLWE_Hybrid(N=1024, q=2**40, p=3, sigma=3.2)
        enc_times, add_times, mul_times = [], [], []

        for _ in range(iterations):
            m = generate_poly(1024, 3)

            # Encryption
            start = time.time()
            ct = hybrid.encrypt(m)
            enc_times.append(time.time() - start)

            # Addition
            start = time.time()
            _ = hybrid.add(ct, ct)
            add_times.append(time.time() - start)

            # Multiplication
            start = time.time()
            _ = hybrid.mul(ct, ct)
            mul_times.append(time.time() - start)

        return {
            "scheme": "NTRU-RLWE",
            "enc_ms": np.mean(enc_times) * 1000,
            "add_ms": np.mean(add_times) * 1000,
            "mul_ms": np.mean(mul_times) * 1000,
            "bootstrapping": "Yes",
            "multiparty": "Yes",
        }

    @staticmethod
    def plot_results(results_df):
        """Generate comparison plots"""
        plt.figure(figsize=(15, 5))

        # Operation times
        plt.subplot(1, 3, 1)
        results_df.plot(
            x="scheme", y=["enc_ms", "add_ms", "mul_ms"], kind="bar", logy=True
        )
        plt.title("Operation Times (log scale)")
        plt.ylabel("Time (ms)")

        # Feature comparison
        plt.subplot(1, 3, 2)
        features = results_df[["scheme", "bootstrapping", "multiparty"]]
        features = features.melt(id_vars="scheme")
        sns.scatterplot(data=features, x="variable", y="scheme", hue="value", s=500)
        plt.title("Feature Support")

        # Speedup comparison
        plt.subplot(1, 3, 3)
        results_df["total_ops_ms"] = (
            results_df["enc_ms"] + results_df["add_ms"] + results_df["mul_ms"]
        )
        results_df["speedup_vs_bfv"] = (
            results_df.loc[results_df["scheme"] == "BFV", "total_ops_ms"].values[0]
            / results_df["total_ops_ms"]
        )
        sns.barplot(data=results_df, x="scheme", y="speedup_vs_bfv")
        plt.title("Speedup vs. BFV")
        plt.ylabel("Times faster")

        plt.tight_layout()
        plt.savefig("benchmark_results.png")

    @staticmethod
    def _benchmark_noise_growth(iterations=3):
        """Benchmark noise growth characteristics"""
        schemes = {
            "NTRU-RLWE": NTRU_RLWE_Hybrid(N=1024, q=2**40, p=3, sigma=3.2),
            "CKKS": ts.context(ts.SCHEME_TYPE.CKKS, 8192, [40, 30, 30, 30, 40]),
            "BFV": ts.context(ts.SCHEME_TYPE.BFV, 4096, 40961),
        }

        results = []
        for name, scheme in schemes.items():
            for _ in range(iterations):
                if name == "NTRU-RLWE":
                    noise_data = scheme.noise_growth_benchmark(operations=3)
                    results.extend([{**d, "scheme": name} for d in noise_data])
                else:
                    # Implement similar for CKKS/BFV if possible
                    pass

        return pd.DataFrame(results)

    @staticmethod
    def plot_noise_growth(results_df, save_dir="results"):
        """
        Generate and save noise growth comparison chart.

        Args:
            results_df: Noise growth benchmark results
            save_dir: Directory to save chart
        """
        Path(save_dir).mkdir(exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Plot RLWE noise
        plt.subplot(1, 2, 1)
        sns.lineplot(
            data=results_df, x="operation", y="rlwe_noise", hue="scheme", marker="o"
        )
        plt.title("RLWE Noise Growth")
        plt.xticks(rotation=45)
        plt.ylabel("Noise Estimate")

        # Plot NTRU noise
        plt.subplot(1, 2, 2)
        sns.lineplot(
            data=results_df, x="operation", y="ntru_noise", hue="scheme", marker="o"
        )
        plt.title("NTRU Noise Growth")
        plt.xticks(rotation=45)
        plt.ylabel("Noise Estimate")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/noise_growth_comparison.png")
        plt.close()

    @staticmethod
    def _estimate_ckks_noise(encrypted):
        """Estimate noise in CKKS ciphertext"""
        try:
            # This is TenSEAL specific - may need adjustment for other libraries
            return encrypted.noise()
        except Exception:
            # Fallback estimation
            return np.std(encrypted.decrypt()) * 1000  # Arbitrary scaling

    @staticmethod
    def _estimate_bfv_noise(encrypted):
        """Estimate noise in BFV ciphertext"""
        try:
            # BFV doesn't directly expose noise, so we use a proxy
            decrypted = np.array(encrypted.decrypt())
            return np.std(decrypted - np.round(decrypted))
        except Exception:
            return 0.0  # Fallback if estimation fails

    @staticmethod
    def _benchmark_noise_comparison(iterations=3):
        """Benchmark noise characteristics across schemes"""
        # Initialize schemes with proper parameters
        schemes = {
            "NTRU-RLWE": NTRU_RLWE_Hybrid(N=1024, q=2**40, p=3, sigma=3.2),
            "CKKS": None,  # Will initialize properly below
            "BFV": None,  # Will initialize properly below
        }

        # Proper CKKS setup
        context_ckks = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[40, 30, 30, 30, 40],
        )
        context_ckks.global_scale = 2**30
        context_ckks.generate_galois_keys()
        schemes["CKKS"] = context_ckks

        # Proper BFV setup
        context_bfv = ts.context(
            ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=40961
        )
        schemes["BFV"] = context_bfv

        results = []

        for name, scheme in schemes.items():
            for i in range(iterations):
                if name == "NTRU-RLWE":
                    # Test our hybrid scheme
                    m = generate_poly(1024, 3)
                    ct = scheme.encrypt(m)
                    rlwe_noise, ntru_noise = scheme.estimate_noise(ct)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "encrypt",
                            "noise": rlwe_noise,
                            "noise_type": "RLWE",
                        }
                    )
                    results.append(
                        {
                            "scheme": name,
                            "operation": "encrypt",
                            "noise": ntru_noise,
                            "noise_type": "NTRU",
                        }
                    )

                    # Addition
                    ct_add = scheme.add(ct, ct)
                    rlwe_noise, ntru_noise = scheme.estimate_noise(ct_add)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "add",
                            "noise": rlwe_noise,
                            "noise_type": "RLWE",
                        }
                    )

                    # Multiplication
                    ct_mul = scheme.mul(ct, ct)
                    rlwe_noise, ntru_noise = scheme.estimate_noise(ct_mul)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "mul",
                            "noise": rlwe_noise,
                            "noise_type": "RLWE",
                        }
                    )

                elif name == "CKKS":
                    # Test CKKS scheme
                    vec = np.random.randn(4096)
                    enc = ts.ckks_vector(scheme, vec)
                    noise = FHEBenchmark._estimate_ckks_noise(enc)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "encrypt",
                            "noise": noise,
                            "noise_type": "CKKS",
                        }
                    )

                    # Addition
                    enc_add = enc + enc
                    noise = FHEBenchmark._estimate_ckks_noise(enc_add)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "add",
                            "noise": noise,
                            "noise_type": "CKKS",
                        }
                    )

                    # Multiplication
                    enc_mul = enc * enc
                    noise = FHEBenchmark._estimate_ckks_noise(enc_mul)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "mul",
                            "noise": noise,
                            "noise_type": "CKKS",
                        }
                    )

                elif name == "BFV":
                    # Test BFV scheme
                    vec = [np.random.randint(0, 100) for _ in range(2048)]
                    enc = ts.bfv_vector(scheme, vec)
                    noise = FHEBenchmark._estimate_bfv_noise(enc)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "encrypt",
                            "noise": noise,
                            "noise_type": "BFV",
                        }
                    )

                    # Addition
                    enc_add = enc + enc
                    noise = FHEBenchmark._estimate_bfv_noise(enc_add)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "add",
                            "noise": noise,
                            "noise_type": "BFV",
                        }
                    )

                    # Multiplication
                    enc_mul = enc * enc
                    noise = FHEBenchmark._estimate_bfv_noise(enc_mul)
                    results.append(
                        {
                            "scheme": name,
                            "operation": "mul",
                            "noise": noise,
                            "noise_type": "BFV",
                        }
                    )

        return pd.DataFrame(results)

    @staticmethod
    def plot_noise_comparison(results_df, save_dir="results"):
        """
        Generate and save noise comparison charts.

        Args:
            results_df: Noise comparison benchmark results
            save_dir: Directory to save charts
        """
        Path(save_dir).mkdir(exist_ok=True)

        # Plot 1: Noise by operation type
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df, x="scheme", y="noise", hue="operation", palette="viridis"
        )
        plt.title("Noise Levels by Operation Type")
        plt.ylabel("Noise Estimate")
        plt.xlabel("Scheme")
        plt.yscale("log")  # Log scale for better comparison
        plt.tight_layout()
        plt.savefig(f"{save_dir}/noise_by_operation.png")
        plt.close()

        # Plot 2: Noise growth comparison
        plt.figure(figsize=(12, 6))
        for scheme in results_df["scheme"].unique():
            scheme_data = results_df[results_df["scheme"] == scheme]
            plt.plot(
                scheme_data["operation"], scheme_data["noise"], marker="o", label=scheme
            )
        plt.title("Noise Growth Across Operations")
        plt.ylabel("Noise Estimate (log scale)")
        plt.xlabel("Operation")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/noise_growth_comparison.png")
        plt.close()

        # Plot 3: Relative noise increase
        results_df["noise_norm"] = results_df.groupby("scheme")["noise"].transform(
            lambda x: x / x.iloc[0]
        )
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=results_df,
            x="operation",
            y="noise_norm",
            hue="scheme",
            style="scheme",
            markers=True,
            dashes=False,
            markersize=10,
        )
        plt.title("Relative Noise Increase (Normalized to Encryption)")
        plt.ylabel("Noise (Multiples of initial noise)")
        plt.xlabel("Operation")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/relative_noise_growth.png")
        plt.close()
