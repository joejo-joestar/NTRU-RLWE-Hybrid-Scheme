"""
RLWE (Ring Learning With Errors) Implementation.

This module implements the RLWE encryption scheme which forms one component
of the NTRU-RLWE hybrid scheme.
"""

import numpy as np
from utils import generate_gaussian_poly, generate_poly, poly_mod, poly_mul, poly_add


class RLWE:
    """
    RLWE Public Key Cryptosystem.

    Attributes:
        N: Polynomial degree
        q: Coefficient modulus
        sigma: Gaussian parameter
        s: Secret key polynomial
    """

    def __init__(self, N: int, q: int, sigma: float):
        """
        Initialize RLWE with parameters and generate secret key.

        Args:
            N: Polynomial degree
            q: Coefficient modulus
            sigma: Gaussian parameter for error generation
        """
        self.N = N
        self.q = q
        self.sigma = sigma
        self.s = generate_gaussian_poly(N, sigma)

    def encrypt(self, m: np.ndarray) -> tuple:
        """
        Encrypt a message using RLWE.

        Args:
            m: Plaintext message polynomial

        Returns:
            Tuple (a, b) representing the ciphertext:
            - a: Random polynomial
            - b: a*s + e + m mod q
        """
        a = generate_poly(self.N, self.q)
        e = generate_gaussian_poly(self.N, self.sigma)
        b = poly_add(poly_mul(a, self.s, self.q, self.N), e, self.q)
        b = poly_add(b, m, self.q)
        return (a, b)

    def decrypt(self, ciphertext: tuple, partial: bool = False) -> np.ndarray:
        """
        Decrypt a ciphertext using RLWE.

        Args:
            ciphertext: Tuple (a, b) to decrypt
            partial: If True, returns raw polynomial before final mod q

        Returns:
            Decrypted message polynomial
        """
        a, b = ciphertext
        m = poly_add(b, -poly_mul(a, self.s, self.q, self.N), self.q)
        return m if partial else poly_mod(m, self.q)
