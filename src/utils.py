"""
Utility functions for NTRU-RLWE hybrid scheme implementation.

Includes polynomial operations, number generation, and other helper functions.
"""

from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.stats import norm


def generate_poly(N: int, q: int) -> np.ndarray:
    """
    Generate a random polynomial with coefficients in [-q/2, q/2].

    Args:
        N: Degree of polynomial
        q: Modulus for coefficients

    Returns:
        Random polynomial as numpy array
    """
    return np.random.randint(-q // 2, q // 2 + 1, N)


def generate_gaussian_poly(N: int, sigma: float) -> np.ndarray:
    """
    Generate a polynomial with coefficients sampled from a discrete Gaussian.

    Args:
        N: Degree of polynomial
        sigma: Standard deviation of Gaussian distribution

    Returns:
        Polynomial with Gaussian coefficients
    """
    return np.round(norm.rvs(0, sigma, N)).astype(int)


def poly_mod(poly: np.ndarray, q: int) -> np.ndarray:
    """
    Reduce polynomial coefficients modulo q to range [-q/2, q/2].

    Args:
        poly: Input polynomial
        q: Modulus

    Returns:
        Polynomial with coefficients reduced mod q
    """
    return np.mod(poly + q // 2, q) - q // 2


def poly_add(poly1: np.ndarray, poly2: np.ndarray, q: int) -> np.ndarray:
    """
    Add two polynomials modulo q.

    Args:
        poly1: First polynomial
        poly2: Second polynomial
        q: Modulus

    Returns:
        Sum of polynomials mod q
    """
    return poly_mod(poly1 + poly2, q)


def poly_mul(poly1: np.ndarray, poly2: np.ndarray, q: int, N: int) -> np.ndarray:
    """
    Multiply two polynomials modulo (x^N + 1) and q.

    Args:
        poly1: First polynomial
        poly2: Second polynomial
        q: Coefficient modulus
        N: Ring degree (x^N + 1)

    Returns:
        Product of polynomials mod (x^N + 1) and q
    """
    result = poly.polymul(poly1, poly2)
    # Modulo x^N + 1
    remainder = np.zeros(N)
    for i in range(len(result)):
        remainder[i % N] += (-1) ** (i // N) * result[i]
    return poly_mod(remainder, q)


def poly_inverse_ntru(poly: np.ndarray, q: int) -> np.ndarray:
    """
    Simplified NTRU-style polynomial inversion for POC.
    In production, use proper NTRU inversion.

    Args:
        poly: Polynomial to invert
        q: Modulus

    Returns:
        Approximate inverse polynomial
    """
    # This is a placeholder - real implementation would use:
    # 1. Check if invertible
    # 2. Use extended Euclidean algorithm for polynomials
    return np.ones_like(poly)


def estimate_noise_ntru(
    ciphertext: np.ndarray, secret_key: np.ndarray, q: int, N: int
) -> float:
    """
    Estimate noise level in NTRU ciphertext.

    Args:
        ciphertext: NTRU ciphertext polynomial
        secret_key: Private key polynomial f
        q: Coefficient modulus
        N: Polynomial degree

    Returns:
        Estimated noise magnitude
    """
    # For NTRU: noise ≈ ciphertext - m (mod q)
    # Since we don't know m, we can only estimate the distribution
    centered = poly_mod(ciphertext, q)
    return np.std(centered)


def estimate_noise_rlwe(
    ciphertext: tuple, secret_key: np.ndarray, q: int, N: int
) -> float:
    """
    Estimate noise level in RLWE ciphertext.

    Args:
        ciphertext: RLWE ciphertext tuple (a, b)
        secret_key: Secret key polynomial s
        q: Coefficient modulus
        N: Polynomial degree

    Returns:
        Estimated noise magnitude
    """
    a, b = ciphertext
    # For RLWE: noise = b - a*s - m ≈ e
    # Since we don't know m, we can only estimate the distribution
    s_times_a = poly_mul(a, secret_key, q, N)
    noise_estimate = poly_add(b, -s_times_a, q)
    return np.std(poly_mod(noise_estimate, q))


def visualize_noise_distribution(
    ciphertext, secret_key=None, q=None, N=None, scheme="rlwe"
):
    """
    Plot histogram of noise distribution in ciphertext.

    Args:
        ciphertext: The ciphertext to analyze
        secret_key: Secret key for noise estimation
        q: Modulus
        N: Polynomial degree
        scheme: 'rlwe' or 'ntru'
    """
    if scheme == "rlwe":
        noise = estimate_noise_rlwe(ciphertext, secret_key, q, N)
        plt.hist(poly_mod(ciphertext[1], q), bins=50, alpha=0.7)
    else:  # ntru
        noise = estimate_noise_ntru(ciphertext, secret_key, q, N)
        plt.hist(poly_mod(ciphertext, q), bins=50, alpha=0.7)

    plt.title(f"Noise Distribution (estimated σ={noise:.2f})")
    plt.xlabel("Coefficient value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"noise_dist_{scheme}.png")
    plt.close()


def calculate_operational_depth(hybrid, max_operations=10):
    """
    Determine maximum operational depth before decryption fails.

    Args:
        hybrid: Initialized NTRU_RLWE_Hybrid instance
        max_operations: Maximum operations to test

    Returns:
        Dictionary with results
    """
    m = generate_poly(hybrid.N, hybrid.p)
    ct = hybrid.encrypt(m)
    results = []

    for i in range(max_operations):
        ct = hybrid.mul(ct, ct)
        try:
            decrypted = hybrid.decrypt(ct)
            error = np.mean(np.abs(decrypted - (m ** (2 ** (i + 1)) % hybrid.p)))
            results.append(
                {"operations": 2 ** (i + 1), "error": error, "success": True}
            )
        except Exception:
            results.append(
                {"operations": 2 ** (i + 1), "error": float("inf"), "success": False}
            )
            break

    return results
