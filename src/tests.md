# `tests.py`

This file contains unit tests for the `NTRU_RLWE_Hybrid` scheme, designed to verify its correctness, homomorphic properties, and behavior under noise. It uses the `pytest` framework.

**Class `TestNTRU_RLWE_Hybrid`**

* **`setup(self)`**
  * **Purpose**: A `pytest` fixture that runs before each test method in the class.
  * **Details**: Initializes common parameters (`N`, `q`, `p`, `sigma`), creates an instance of `NTRU_RLWE_Hybrid`, and a sample `test_poly`.
  * **Math Used**: Parameter instantiation.
* **`test_encrypt_decrypt_consistency(self)`**
  * **Purpose**: Checks if encrypting a polynomial and then decrypting it returns the original polynomial.
  * **Math Used**: Relies on `hybrid.encrypt` and `hybrid.decrypt`. Comparison uses `np.array_equal(decrypted, self.test_poly % self.p)`. The modulo `p` ensures the comparison is in the plaintext space.
* **`test_zero_encryption(self)`**
  * **Purpose**: Tests encryption and decryption of a zero polynomial.
  * **Math Used**: `hybrid.encrypt`, `hybrid.decrypt`. Checks `np.all(decrypted == 0)`.
* **`test_homomorphic_addition(self)`**
  * **Purpose**: Verifies the homomorphic addition property: $Dec(Enc(poly1) + Enc(poly2)) = (poly1 + poly2) \pmod p$.
  * **Details**: Encrypts two polynomials, homomorphically adds their ciphertexts, decrypts the result, and compares it to the sum of the original polynomials (modulo `p`).
  * **Math Used**:
    * `hybrid.encrypt`, `hybrid.add`, `hybrid.decrypt`.
    * Expected result: `(poly1 + poly2) % self.p`.
    * Comparison: `np.allclose(decrypted_sum, expected_sum_mod_p)`, which allows for small floating-point differences if noise was a factor, though for addition, it's often exact.
* **`test_homomorphic_multiplication(self)`**
  * **Purpose**: Verifies the homomorphic multiplication property: $Dec(Enc(poly1) * Enc(poly2))$ should be related to $(poly1 \cdot poly2) \pmod p$.
  * **Details**: Encrypts two polynomials, homomorphically multiplies their ciphertexts, decrypts, and compares. Uses `np.mean(np.abs(result - expected)) < 2` for tolerance, acknowledging noise or approximation from the simplified multiplication.
  * **Math Used**:
    * `hybrid.encrypt`, `hybrid.mul`, `hybrid.decrypt`.
    * Expected result: `(poly1 * poly2) % self.p`. (Note: standard polynomial multiplication, not necessarily ring multiplication if `poly1`, `poly2` are just arrays of coefficients).
    * Error metric: Mean Absolute Error $\frac{1}{N} \sum_{i=0}^{N-1} | \text{result}_i - \text{expected}_i |$.
* **`test_max_value_encryption(self)`**
  * **Purpose**: Tests encryption/decryption of a polynomial with all coefficients at the maximum plaintext value (`self.p - 1`).
  * **Math Used**: `hybrid.encrypt`, `hybrid.decrypt`.
* **`test_min_value_encryption(self)`**
  * **Purpose**: Tests encryption/decryption with minimum coefficient values, typically negative values that map into $\mathbb{Z}_p$. Uses `min_poly = np.full(self.N, -(self.p // 2))` and checks against `min_poly % self.p`.
  * **Math Used**: `hybrid.encrypt`, `hybrid.decrypt`. Modulo `p` for expected result.
* **`test_noise_growth_single_operation(self)`**
  * **Purpose**: Checks if a ciphertext remains decryptable after one homomorphic multiplication (squaring).
  * **Math Used**: `hybrid.encrypt`, `hybrid.mul` (squaring), `hybrid.decrypt`.
* **`test_bootstrapping_effectiveness(self)`**
  * **Purpose**: Aims to test if `hybrid.bootstrap` helps maintain decryptability after several multiplications.
  * **Details**: Performs two multiplications (squarings), then applies bootstrapping.
  * **Math Used**: `hybrid.encrypt`, `hybrid.mul`, `hybrid.bootstrap` (simulated), `hybrid.decrypt`.
  * **Note**: Given `hybrid.bootstrap` is simulated and doesn't reduce cryptographic noise, this test's outcome depends on whether two multiplications already exceed the noise budget without true bootstrapping.
* **`test_multiparty_decryption(self)`**
  * **Purpose**: Tests a simulated multiparty decryption scenario.
  * **Details**:

1. An "Aggregator" party (conceptually) owns the full scheme parameters.
2. Three parties are created, each potentially having a share of keys (though in this setup, they might get full keys if not properly sharded).
3. Party 0 encrypts `self.test_poly`.
4. Each party generates a partial decryption: `partials.append(party.decrypt(ciphertext, partial=True))`.
5. Partials are combined: `combined = np.sum(partials, axis=0) % self.q`. This is a simple sum and modulo `q`.
6. Final result calculation: `final_decryption = poly_mod(poly_mul(parties.ntru.fp, combined, self.q, self.N), self.p)`. This uses `fp` (the simulated NTRU private key component) from one party to process the sum.
    * **Math Used**: `hybrid.encrypt`, `hybrid.decrypt(partial=True)`. Summation modulo `q`. Polynomial multiplication and reduction modulo `p`.
    * **Simulated Aspect**: As with `FHEBenchmark.test_multiparty`, the combination logic (`np.sum` and then multiplication by one party's `fp`) is a significant simplification. Real multiparty decryption for a hybrid scheme like this would involve complex protocols for combining partial decryptions from both the RLWE and NTRU layers, often using techniques like threshold cryptography. For instance, RLWE decryption $b - as$ requires handling shares of $s$. NTRU decryption $f_p \cdot (f \cdot e \pmod q) \pmod p$ requires handling shares of $f$ and $f_p$. A simple sum of the `partial=True` outputs from the hybrid decrypt (which are themselves results of partial NTRU decryptions of partial RLWE decryptions) is unlikely to be mathematically sound for standard multiparty schemes without very specific (and likely insecure) key sharing.
