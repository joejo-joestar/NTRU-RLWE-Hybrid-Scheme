# `hybrid.py`

This file implements the core logic for the NTRU-RLWE Hybrid FHE scheme. This scheme combines elements from both NTRU and RLWE cryptosystems, likely aiming to leverage strengths from both.

**Class `NTRU_RLWE_Hybrid`**

* **`__init__(self, N: int, q: int, p: int, sigma: float)`**
  * **Purpose**: Constructor to initialize the parameters and components of the hybrid FHE scheme.
  * **Details**:
    * `N`: Polynomial degree. Operations are typically in a polynomial ring $R = \mathbb{Z}[x]/(x^N+1)$, where $x^N+1$ is a cyclotomic polynomial.
    * `q`: Ciphertext coefficient modulus. Coefficients of polynomials in ciphertexts are in $\mathbb{Z}_q$ (integers modulo $q$).
    * `p`: Plaintext coefficient modulus. Coefficients of plaintext polynomials are in $\mathbb{Z}_p$.
    * `sigma`: Standard deviation for sampling errors from a discrete Gaussian distribution. This is crucial for the security of LWE-based schemes.
It initializes `self.ntru` (an instance of `NTRU`) and `self.rlwe` (an instance of `RLWE`).
  * **Math Used**: Defines the fundamental algebraic setting: polynomial rings modulo $x^N+1$ and integer moduli $q$ and $p$. Gaussian distribution parameter $\sigma$.
* **`encrypt(self, m: np.ndarray) -> tuple`**
  * **Purpose**: Encrypts a plaintext message polynomial `m`.
  * **Details**: This is a two-layer encryption:

1. The plaintext polynomial `m` is first encrypted using the NTRU scheme: `ntru_ct = self.ntru.encrypt(m, self.ntru.h)`.
2. The resulting NTRU ciphertext `ntru_ct` is then treated as a new plaintext and encrypted using the RLWE scheme: `(a,b) = self.rlwe.encrypt(ntru_ct)`.
    * **Math Used**: Relies entirely on the mathematical operations of `self.ntru.encrypt` and `self.rlwe.encrypt` (detailed in `ntru.py` and `rlwe.py` sections).

* **`decrypt(self, ciphertext: tuple, partial: bool = False, share_num: int = 1) -> np.ndarray`**
  * **Purpose**: Decrypts a hybrid ciphertext.
  * **Details**: This is a two-layer decryption, reversing the encryption process:

1. The RLWE ciphertext `(a,b)` is decrypted using the RLWE secret key: `decrypted_rlwe = self.rlwe.decrypt(ciphertext, partial=partial)`. If `partial` is true, this is $m_{\text{NTRU_ct}} + e_{\text{RLWE}}$. Otherwise, it is $m_{\text{NTRU_ct}}$ (the NTRU ciphertext).
2. The result `decrypted_rlwe` (which is an NTRU ciphertext) is then decrypted using the NTRU private key: `final_plaintext = self.ntru.decrypt(decrypted_rlwe, partial=partial)`. If `partial` is true, this is an intermediate NTRU decryption value. Otherwise, it's the original message polynomial.
    * **Math Used**: Relies on `self.rlwe.decrypt` and `self.ntru.decrypt`. The `partial` flag affects whether final plaintext space reductions/cleanups are performed.

* **`add(self, ct1: tuple, ct2: tuple) -> tuple`**
  * **Purpose**: Performs homomorphic addition of two hybrid scheme ciphertexts. Since the outer layer is RLWE, this typically means RLWE addition.
  * **Details**: Given two RLWE ciphertexts $ct_1 = (a_1, b_1)$ and $ct_2 = (a_2, b_2)$, the addition is usually component-wise:
\$ ct_{add} = ( (a_1 + a_2) \pmod q, (b_1 + b_2) \pmod q ) \$
This relies on the property that $Dec(ct_1 + ct_2) = Dec(ct_1) + Dec(ct_2)$.
  * **Math Used**: Polynomial addition performed coefficient-wise, followed by coefficient-wise modular reduction modulo $q$.
    * `a_sum = poly_add(ct1, ct2, self.q)`
    * `b_sum = poly_add(ct1[1], ct2[1], self.q)`
These use `poly_add` from `utils.py`, which performs \$ (p_1[i] + p_2[i]) \pmod q \$ for each coefficient $i$.
* **`mul(self, ct1: tuple, ct2: tuple) -> tuple`**
  * **Purpose**: Performs homomorphic multiplication of two hybrid scheme ciphertexts.
  * **Details**: The implementation is described as a "simplified version." For standard RLWE ciphertexts $ct_1=(a_1,b_1)$ and $ct_2=(a_2,b_2)$, a common way to multiply (before relinearization) results in a 3-component ciphertext related to $m_1 m_2$. The provided code computes:
\$ a_{new} = (a_1 \cdot a_2 + a_1 \cdot b_2) \pmod{q, x^N+1} \$
\$ b_{new} = (b_1 \cdot a_2 + b_1 \cdot b_2) \pmod{q, x^N+1} \$
The new ciphertext is $(a_{\text{new}}, b_{\text{new}})$. This structure is non-standard for obtaining a ciphertext of $m_1 m_2$. It might be specific to this hybrid approach or a simplification.
  * **Math Used**: Polynomial addition (`poly_add`) and polynomial multiplication (`poly_mul`) in the ring $R_q = \mathbb{Z}_q[x]/(x^N+1)$.
  * **Simulated Aspect**: A production-level RLWE multiplication typically involves computing terms like $c_0 = b_1 b_2 \pmod q$, $c_1 = a_1 b_2 + a_2 b_1 \pmod q$, and $c_2 = a_1 a_2 \pmod q$. The resulting ciphertext, encrypting $m_1 m_2$, is of a larger form (quadratic in the secret key) and requires a "relinearization" step (using an evaluation key, also known as key switching) to reduce it back to a standard 2-component ciphertext. Modulus switching might also be used to manage noise growth. The current form seems to be an ad-hoc operation.
* **`bootstrap(self, ct: tuple) -> tuple`**
  * **Purpose**: Intended to reduce the noise in a ciphertext, allowing for more homomorphic operations.
  * **Details**: **This is a simulated/placeholder implementation.** It only performs `poly_mod(a, self.q)` and `poly_mod(b, self.q)`. This centers the coefficients modulo $q$ but does *not* perform cryptographic bootstrapping (noise reduction).
  * **Math Used (Simulated)**: `poly_mod` (coefficient centering).
  * **Actual Bootstrapping Math**: True FHE bootstrapping is a complex procedure. It involves homomorphically evaluating the decryption circuit using an encrypted version of the secret key (the "bootstrapping key"). For an RLWE ciphertext $ct=(a,b)$ encrypting $m$, the decryption is $m \approx (b - as) \pmod q$. Bootstrapping would compute $Dec_{sk_{boot}}( Enc_{pk_{boot}}(b - as) )$, where $Enc_{pk_{boot}}$ denotes encryption under the bootstrapping public key and $Dec_{sk_{boot}}$ is the homomorphic evaluation of this decryption using an encrypted secret key. This requires techniques like digit extraction, blind rotation (for GSW-like schemes), or homomorphic evaluation of polynomial multiplications and additions, often involving modulus switching (e.g., from $Q$ to $q$).
* **`estimate_noise(self, ciphertext: tuple) -> tuple`**
  * **Purpose**: Estimates the noise levels in both the RLWE and NTRU components of the hybrid ciphertext.
  * **Details**:

1. Estimates RLWE noise using `utils.estimate_noise_rlwe`.
2. Partially decrypts the RLWE layer: `decrypted_rlwe = self.rlwe.decrypt(ciphertext, partial=True)`. This result is the NTRU ciphertext with some RLWE noise.
3. Estimates NTRU noise in `decrypted_rlwe` using `utils.estimate_noise_ntru`.
    * **Math Used**: Relies on the noise estimation logic in `utils.py` (standard deviation of appropriately processed ciphertext components).

* **`noise_growth_benchmark(self, operations=5)`**
  * **Purpose**: Benchmarks how noise accumulates with sequential homomorphic additions and multiplications.
  * **Details**: Encrypts a message, then repeatedly applies `add` (with a fresh encryption of another message) or `mul` (with itself, i.e., squaring) and estimates noise at each step.
  * **Math Used**: Repeated application of `self.add`, `self.mul`, and `self.estimate_noise`.
* **`_check_decryptable(self, ciphertext)`**
  * **Purpose**: A utility to check if a ciphertext can be successfully decrypted.
  * **Details**: Attempts `self.decrypt(ciphertext)`. If it fails (e.g., due to excessive noise), it catches the exception.
  * **Math Used**: Relies on the decryption process.
