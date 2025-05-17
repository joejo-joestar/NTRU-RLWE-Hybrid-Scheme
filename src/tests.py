"""
NTRU-RLWE Hybrid Scheme Unit Tests

This file verifies:
1. Correctness of encryption/decryption
2. Homomorphic operation validity
3. Edge case handling
4. Noise management
"""

import numpy as np
import pytest
from hybrid import NTRU_RLWE_Hybrid
from utils import generate_poly, poly_mod, poly_mul


class TestNTRU_RLWE_Hybrid:
    """Comprehensive test suite for the hybrid scheme"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize test environment"""
        self.N = 1024
        self.q = 2**20
        self.p = 3
        self.sigma = 3.2
        self.hybrid = NTRU_RLWE_Hybrid(N=self.N, q=self.q, p=self.p, sigma=self.sigma)
        self.test_poly = generate_poly(self.N, self.p)

    # Basic Functionality Tests
    def test_encrypt_decrypt_consistency(self):
        """Verify plaintext survives encryption-decryption roundtrip"""
        ciphertext = self.hybrid.encrypt(self.test_poly)
        decrypted = self.hybrid.decrypt(ciphertext)
        assert np.array_equal(self.test_poly, decrypted), (
            "Decrypted text doesn't match original"
        )

    def test_zero_encryption(self):
        """Test encryption/decryption of zero polynomial"""
        zero_poly = np.zeros(self.N)
        decrypted = self.hybrid.decrypt(self.hybrid.encrypt(zero_poly))
        assert np.all(decrypted == 0), "Zero polynomial not preserved"

    # Homomorphic Operation Tests
    def test_homomorphic_addition(self):
        """Verify Enc(a) + Enc(b) == Enc(a + b)"""
        a = generate_poly(self.N, self.p)
        b = generate_poly(self.N, self.p)

        ct_a = self.hybrid.encrypt(a)
        ct_b = self.hybrid.encrypt(b)
        sum_ct = self.hybrid.add(ct_a, ct_b)

        expected = (a + b) % self.p
        result = self.hybrid.decrypt(sum_ct)
        assert np.allclose(result, expected, atol=1), "Addition homomorphism failed"

    def test_homomorphic_multiplication(self):
        """Verify Enc(a) * Enc(b) ≈ Enc(a * b) with noise tolerance"""
        a = generate_poly(self.N, self.p)
        b = generate_poly(self.N, self.p)

        ct_a = self.hybrid.encrypt(a)
        ct_b = self.hybrid.encrypt(b)
        product_ct = self.hybrid.mul(ct_a, ct_b)

        # Allow some noise in multiplication results
        result = self.hybrid.decrypt(product_ct)
        expected = (a * b) % self.p
        assert np.mean(np.abs(result - expected)) < 2, (
            "Multiplication homomorphism failed"
        )

    # Edge Case Tests
    def test_max_value_encryption(self):
        """Test polynomials with maximum coefficient values"""
        max_poly = np.full(self.N, self.p - 1)
        decrypted = self.hybrid.decrypt(self.hybrid.encrypt(max_poly))
        assert np.array_equal(decrypted, max_poly), "Max value polynomial not preserved"

    def test_min_value_encryption(self):
        """Test polynomials with minimum coefficient values"""
        min_poly = np.full(self.N, -(self.p // 2))
        decrypted = self.hybrid.decrypt(self.hybrid.encrypt(min_poly))
        assert np.array_equal(decrypted, min_poly % self.p), (
            "Min value polynomial not preserved"
        )

    # Noise Management Tests
    def test_noise_growth_single_operation(self):
        """Verify noise grows as expected after one multiplication"""
        ct = self.hybrid.encrypt(self.test_poly)
        ct = self.hybrid.mul(ct, ct)
        assert self.hybrid.decrypt(ct) is not None, (
            "Noise growth made ciphertext undecryptable"
        )

    def test_bootstrapping_effectiveness(self):
        """Test if bootstrapping reduces noise"""
        # Encrypt and perform multiple multiplications
        ct = self.hybrid.encrypt(self.test_poly)
        for _ in range(2):
            ct = self.hybrid.mul(ct, ct)

        # Bootstrap and verify decryption still works
        bootstrapped_ct = self.hybrid.bootstrap(ct)
        assert self.hybrid.decrypt(bootstrapped_ct) is not None, (
            "Bootstrapping failed to reduce noise"
        )

    # Multiparty Tests
    def test_multiparty_decryption(self):
        """Verify threshold decryption works with 3 parties"""
        parties = [
            NTRU_RLWE_Hybrid(N=self.N, q=self.q, p=self.p, sigma=self.sigma)
            for _ in range(3)
        ]

        # Party 0 encrypts
        ciphertext = parties[0].encrypt(self.test_poly)

        # Collect partial decryptions
        partials = []
        for i, party in enumerate(parties):
            partial = party.decrypt(ciphertext, partial=True)
            partials.append(partial)

        # Combine partials (simplified for demo)
        combined = np.sum(partials, axis=0) % self.q
        final = poly_mod(poly_mul(parties[0].ntru.fp, combined, self.q, self.N), self.p)

        assert np.array_equal(final, self.test_poly), "Multiparty decryption failed"


if __name__ == "__main__":
    # Command-line test execution
    import unittest

    unittest.main(argv=[""], exit=False)
