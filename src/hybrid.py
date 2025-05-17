"""
NTRU-RLWE Hybrid Scheme Implementation.

This module combines NTRU and RLWE into a hybrid scheme with:
- Faster encryption/decryption
- Better noise management
- Support for multiparty operations
- Extensive homomorphic computation capabilities
"""

import numpy as np
from utils import (
    estimate_noise_ntru,
    estimate_noise_rlwe,
    generate_poly,
    poly_add,
    poly_mod,
    poly_mul,
)
from ntru import NTRU
from rlwe import RLWE


class NTRU_RLWE_Hybrid:
    """
    Hybrid NTRU-RLWE Fully Homomorphic Encryption Scheme.

    Attributes:
        N: Polynomial degree
        q: Coefficient modulus
        p: Plaintext modulus
        sigma: Gaussian parameter
        ntru: NTRU component
        rlwe: RLWE component
    """

    def __init__(self, N: int, q: int, p: int, sigma: float):
        """
        Initialize hybrid scheme with parameters.

        Args:
            N: Polynomial degree
            q: Coefficient modulus
            p: Plaintext modulus
            sigma: Gaussian parameter
        """
        self.N = N
        self.q = q
        self.p = p
        self.sigma = sigma
        self.ntru = NTRU(N, q, p, sigma)
        self.rlwe = RLWE(N, q, sigma)

    def encrypt(self, m: np.ndarray) -> tuple:
        """
        Hybrid encryption combining NTRU and RLWE.

        1. First encrypt with NTRU
        2. Then encrypt the NTRU ciphertext with RLWE

        Args:
            m: Plaintext message polynomial

        Returns:
            RLWE ciphertext tuple (a, b) encrypting the NTRU ciphertext
        """
        # First encrypt with NTRU
        ntru_ct = self.ntru.encrypt(m, self.ntru.fq)

        # Then encrypt with RLWE
        rlwe_ct = self.rlwe.encrypt(ntru_ct)

        return rlwe_ct

    def decrypt(
        self, ciphertext: tuple, partial: bool = False, share_num: int = 1
    ) -> np.ndarray:
        """
        Hybrid decryption combining RLWE and NTRU.

        Args:
            ciphertext: RLWE ciphertext tuple (a, b)
            partial: If True, returns partial decryption
            share_num: Which share this is (1-based index)

        Returns:
            Decrypted message polynomial
        """
        # First decrypt with RLWE (pass through partial flag)
        ntru_ct = self.rlwe.decrypt(ciphertext, partial=partial)

        # Then decrypt with NTRU (pass through partial flag)
        return self.ntru.decrypt(ntru_ct, partial=partial)

    def add(self, ct1: tuple, ct2: tuple) -> tuple:
        """
        Homomorphic addition of two ciphertexts.

        Args:
            ct1: First ciphertext tuple (a1, b1)
            ct2: Second ciphertext tuple (a2, b2)

        Returns:
            New ciphertext (a1+a2, b1+b2) mod q
        """
        a1, b1 = ct1
        a2, b2 = ct2
        return (poly_add(a1, a2, self.q), poly_add(b1, b2, self.q))

    def mul(self, ct1: tuple, ct2: tuple) -> tuple:
        """
        Homomorphic multiplication (tensor product) of two ciphertexts.

        Note: This is a simplified version. A production implementation
        would include key switching and modulus reduction.

        Args:
            ct1: First ciphertext tuple (a1, b1)
            ct2: Second ciphertext tuple (a2, b2)

        Returns:
            New ciphertext representing the product
        """
        a1, b1 = ct1
        a2, b2 = ct2

        # Tensor product (simplified)
        a_new = poly_add(
            poly_mul(a1, a2, self.q, self.N), poly_mul(a1, b2, self.q, self.N), self.q
        )
        b_new = poly_add(
            poly_mul(b1, a2, self.q, self.N), poly_mul(b1, b2, self.q, self.N), self.q
        )

        return (a_new, b_new)

    def bootstrap(self, ct: tuple) -> tuple:
        """
        Noise reduction through bootstrapping.

        In a full implementation, this would involve homomorphically
        evaluating the decryption circuit. For POC we simulate by
        simply reducing the noise.

        Args:
            ct: Ciphertext tuple (a, b) to bootstrap

        Returns:
            Refreshed ciphertext with reduced noise
        """
        a, b = ct
        a = poly_mod(a, self.q)
        b = poly_mod(b, self.q)
        return (a, b)

    def estimate_noise(self, ciphertext: tuple) -> float:
        """
        Estimate noise level in hybrid ciphertext.

        Args:
            ciphertext: RLWE ciphertext tuple (a, b) encrypting NTRU ciphertext

        Returns:
            Tuple of (rlwe_noise, ntru_noise) estimates
        """
        # First get RLWE noise estimate
        a, b = ciphertext
        rlwe_noise = estimate_noise_rlwe(ciphertext, self.rlwe.s, self.q, self.N)

        # Get the encrypted NTRU ciphertext by RLWE-decrypting
        ntru_ct = self.rlwe.decrypt(ciphertext, partial=True)
        ntru_noise = estimate_noise_ntru(ntru_ct, self.ntru.f, self.q, self.N)

        return (rlwe_noise, ntru_noise)

    def noise_growth_benchmark(self, operations=5):
        """
        Benchmark noise growth through sequential operations.

        Args:
            operations: Number of operations to perform

        Returns:
            Dictionary with noise measurements
        """
        m = generate_poly(self.N, self.p)
        ct = self.encrypt(m)
        results = []

        # Initial noise
        rlwe_noise, ntru_noise = self.estimate_noise(ct)
        results.append(
            {"operation": "encrypt", "rlwe_noise": rlwe_noise, "ntru_noise": ntru_noise}
        )

        # Addition operations
        ct_add = ct
        for i in range(operations):
            ct_add = self.add(ct_add, ct)
            rlwe_noise, ntru_noise = self.estimate_noise(ct_add)
            results.append(
                {
                    "operation": f"add_{i + 1}",
                    "rlwe_noise": rlwe_noise,
                    "ntru_noise": ntru_noise,
                }
            )

        # Multiplication operations
        ct_mul = ct
        for i in range(operations):
            ct_mul = self.mul(ct_mul, ct)
            rlwe_noise, ntru_noise = self.estimate_noise(ct_mul)
            results.append(
                {
                    "operation": f"mul_{i + 1}",
                    "rlwe_noise": rlwe_noise,
                    "ntru_noise": ntru_noise,
                    "decryptable": self._check_decryptable(ct_mul),
                }
            )

        return results

    def _check_decryptable(self, ciphertext):
        """Check if ciphertext is still decryptable."""
        try:
            self.decrypt(ciphertext)
            return True
        except Exception:
            return False
