"""
NTRU Public Key Cryptosystem Implementation.

This module implements the NTRU encryption scheme which forms one component
of the NTRU-RLWE hybrid scheme.
"""

import numpy as np
from utils import (
    generate_gaussian_poly,
    poly_mul,
    poly_add,
    poly_mod,
    poly_inverse_ntru,
)


class NTRU:
    """
    NTRU Public Key Cryptosystem.

    Attributes:
        N: Polynomial degree
        q: Coefficient modulus
        p: Plaintext modulus
        sigma: Gaussian parameter
        f: Private key polynomial
        fp: Inverse of f mod p
        fq: Inverse of f mod q
    """

    def __init__(self, N: int, q: int, p: int, sigma: float):
        """
        Initialize NTRU with parameters and generate key pair.

        Args:
            N: Polynomial degree
            q: Coefficient modulus
            p: Plaintext modulus
            sigma: Gaussian parameter for key generation
        """
        self.N = N
        self.q = q
        self.p = p
        self.sigma = sigma
        self.f, self.fp, self.fq = self._generate_key_pair()

    def _generate_key_pair(self) -> tuple:
        """
        Generate NTRU key pair (private key f and public key h).

        Returns:
            Tuple of (f, fp, fq) where:
            - f is the private key
            - fp is inverse of f mod p
            - fq is inverse of f mod q (used to compute public key)
        """
        while True:
            f = generate_gaussian_poly(self.N, self.sigma)
            g = generate_gaussian_poly(self.N, self.sigma)

            # Check if f is invertible mod p (simplified for POC)
            if np.sum(f) % self.p != 0:
                fp = poly_inverse_ntru(f, self.p)
                fq = poly_inverse_ntru(f, self.q)

                # Compute public key h = p * fq * g mod q
                h = poly_mul(fq, g, self.q, self.N)
                h = poly_mul([self.p], h, self.q, self.N)

                return f, fp, h

    def encrypt(self, m: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Encrypt a message using NTRU.

        Args:
            m: Plaintext message polynomial
            h: Public key polynomial

        Returns:
            Ciphertext polynomial
        """
        r = generate_gaussian_poly(self.N, self.sigma)
        e = poly_add(poly_mul(h, r, self.q, self.N), m, self.q)
        return e

    def decrypt(self, e: np.ndarray, partial: bool = False) -> np.ndarray:
        """
        Decrypt a ciphertext using NTRU.

        Args:
            e: Ciphertext polynomial
            partial: If True, returns intermediate value before final mod p

        Returns:
            Decrypted message polynomial
        """
        a = poly_mul(self.f, e, self.q, self.N)
        if partial:
            return a
        m = poly_mul(self.fp, a, self.q, self.N)
        return poly_mod(m, self.p)
