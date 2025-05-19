# `rlwe.py`

This file implements the Ring Learning With Errors (RLWE) cryptosystem. RLWE is a foundational problem for many modern FHE schemes.

**Class `RLWE`**

* **`__init__(self, N: int, q: int, sigma: float)`**
  * **Purpose**: Constructor for the RLWE scheme.
  * **Details**:
    * `N, q, sigma`: Parameters as defined previously.
    * Generates the secret key `s` as a polynomial with coefficients sampled from a discrete Gaussian distribution (`generate_gaussian_poly(self.N, self.sigma)`). The coefficients of `s` are usually small.
  * **Math Used**: Discrete Gaussian sampling for the secret key $s \in R_q$.
* **`encrypt(self, m: np.ndarray) -> tuple`**
  * **Purpose**: Encrypts a message polynomial `m`. This is a symmetric-key variant if `a` is fixed or public if derived from a public key like `(a, p_0 = -as+e')`. Here `a` is generated freshly.
  * **Details**:

1. A polynomial `a` is generated uniformly at random from $R_q$ (`generate_poly(self.N, self.q)`). This `a` is public.
2. An error polynomial `e` is sampled from the discrete Gaussian distribution (`generate_gaussian_poly(self.N, self.sigma)`).
3. The ciphertext component `b` is computed as:
$b = (a \cdot s + e + m) \pmod{q, x^N+1}$
The message `m` is embedded directly. For FHE, `m` is often scaled by $\Delta = q/p$ if the plaintext space is $\mathbb{Z}_p$.
The ciphertext is the tuple `(a, b)`.
    * **Math Used**:
        * Uniform random polynomial generation for $a$.
        * Discrete Gaussian sampling for error $e$.
        * Polynomial multiplication (`poly_mul`) for $a \cdot s$.
        * Polynomial addition (`poly_add`) for adding $e$ and $m$.
        * All operations are in the ring $R_q = \mathbb{Z}_q[x]/(x^N+1)$.

* **`decrypt(self, ciphertext: tuple, partial: bool = False) -> np.ndarray`**
  * **Purpose**: Decrypts an RLWE ciphertext `(a,b)`.
  * **Details**:

1. The core decryption computes:
\$ m' = (b - a \cdot s) \pmod{q, x^N+1} \$
Substituting $b = a \cdot s + e + m$, we get $m' = (a \cdot s + e + m - a \cdot s) = (m + e) \pmod{q, x^N+1}$.
1. If `partial` is true, this $m' = m+e$ is returned (coefficients centered by `poly_mod`).
2. Otherwise, `poly_mod(m', self.q)` is returned. This centers the coefficients of $m+e$. To recover the actual message `m` (especially if it was in $\mathbb{Z}_p$), further steps like rounding and modulo `p` reduction are typically needed, e.g., \$ round((p/q) \cdot (m+e)) \pmod p \$. The current implementation doesn't show this final plaintext recovery step, suggesting `m` might be an NTRU ciphertext with large coefficients, or the plaintext space is handled differently.
    * **Math Used**:
        * Polynomial multiplication (`poly_mul`) for $a \cdot s$.
        * Polynomial subtraction (via `poly_add` with a negative) for $b - (a \cdot s)$.
        * Modular reduction (`poly_mod`) to keep coefficients in a symmetric range around 0 modulo $q$.
