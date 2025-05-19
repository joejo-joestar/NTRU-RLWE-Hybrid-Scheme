# `ntru.py`

This file implements the NTRU (Nth-degree Truncated polynomial Ring Units) cryptosystem. NTRU is a lattice-based public-key cryptosystem.

**Class `NTRU`**

* **`__init__(self, N: int, q: int, p: int, sigma: float)`**
  * **Purpose**: Constructor for the NTRU scheme.
  * **Details**: Initializes parameters `N, q, p, sigma` (as in `hybrid.py`). Calls `_generate_key_pair` to create the private key `f` (and its inverses `fp`, `fq`) and public key `h`.
  * **Math Used**: Parameter setup.
* **`_generate_key_pair(self) -> tuple`**
  * **Purpose**: Generates the NTRU public and private key pair.
  * **Details**:

1. Polynomials `f` and `g` are generated, typically with small coefficients (e.g., from a discrete Gaussian distribution using `generate_gaussian_poly`).
2. A simplified check for `f`'s invertibility mod `p` is performed: `np.sum(f) % self.p != 0`. This is insufficient; true invertibility requires checking in the polynomial ring $R_p = \mathbb{Z}_p[x]/(x^N+1)$.
3. `fp = poly_inverse_ntru(f, self.p)` and `fq = poly_inverse_ntru(f, self.q)` are computed. **Crucially, `poly_inverse_ntru` in `utils.py` is a placeholder that returns a polynomial of ones.** So, `fp` and `fq` are not actual inverses.
4. The public key `h` is computed as $h = (p \cdot f_q \cdot g) \pmod{q, x^N+1}$. Due to the placeholder `fq`, this simplifies to $h = (p \cdot g) \pmod{q, x^N+1}$.
    * **Math Used (Partially Simulated)**:
        * Gaussian polynomial generation (`generate_gaussian_poly`).
        * Polynomial multiplication (`poly_mul`) and modular arithmetic.
    * **Actual NTRU Key Generation Math**:
        * Generate `f` from a specific distribution (e.g., ternary $\{-1,0,1\}$ or Gaussian).
        * Ensure `f` is invertible modulo $p$ and modulo $q$ in the rings $R_p$ and $R_q$. This means $\gcd(f(x), x^N+1)$ should be 1 (or an invertible constant) modulo $p$ and $q$.
        * Compute true inverses: $f_p = f^{-1} \pmod{p, x^N+1}$ and $f_q = f^{-1} \pmod{q, x^N+1}$ using the extended Euclidean algorithm for polynomials.
        * Generate `g` similarly.
        * The public key is $h = (f_q \cdot g) \pmod{q, x^N+1}$ (in some variants, it's $h = (p \cdot f_q \cdot g) \pmod{q, x^N+1}$).

* **`encrypt(self, m: np.ndarray, h: np.ndarray) -> np.ndarray`**
  * **Purpose**: Encrypts a message polynomial `m` using the public key `h`.
  * **Details**:

1. A random "blinding" polynomial `r` is generated (`generate_gaussian_poly`).
2. The ciphertext `e` (often denoted `c` or `ct`) is computed as:
\$ e = (r \cdot h + m) \pmod{q, x^N+1} \$
    * **Math Used**:
        * Gaussian polynomial generation.
        * Polynomial multiplication (`poly_mul`) and addition (`poly_add`).
        * All operations are in the ring $R_q = \mathbb{Z}_q[x]/(x^N+1)$.
    * **Effect of Simplified Key**: With $h = (p \cdot g) \pmod{q, x^N+1}$, encryption becomes $e = (r \cdot p \cdot g + m) \pmod{q, x^N+1}$.

* **`decrypt(self, e: np.ndarray, partial: bool = False) -> np.ndarray`**
  * **Purpose**: Decrypts an NTRU ciphertext `e`.
  * **Details**:

1. Compute an intermediate value: $a = (f \cdot e) \pmod{q, x^N+1}$. This polynomial's coefficients are centered around 0 modulo $q$.
2. If `partial` is true, `a` is returned.
3. Otherwise, recover the message: $m' = (f_p \cdot a) \pmod{p, x^N+1}$.
    * **Math Used (Affected by Simulated Key)**:
        * Polynomial multiplication (`poly_mul`).
        * Modular reduction (`poly_mod` for centering, and then modulo $p$).
    * **Effect of Simplified Key**:
        * With $e = (r \cdot p \cdot g + m_{\text{orig}}) \pmod{q, x^N+1}$, then $a = (f \cdot (r \cdot p \cdot g + m_{\text{orig}})) \pmod{q, x^N+1}$.
        * When reducing $a$ modulo $p$, the term $f \cdot r \cdot p \cdot g$ becomes zero. So, $a \equiv (f \cdot m_{\text{orig}}) \pmod p$.
        * Since `fp` (the supposed inverse of `f` mod `p`) is simulated as a polynomial of ones, the final step $m' = (f_p \cdot a) \pmod{p, x^N+1}$ becomes $m' = a \pmod{p, x^N+1}$.
        * Thus, decryption yields $m' = (f \cdot m_{\text{orig}}) \pmod{p, x^N+1}$.
        * For correct decryption (i.e., $m' = m_{\text{orig}}$), this implies that $f$ must act as an identity element modulo $p$ (e.g., $f \equiv \mathbf{1} \pmod p$), or that the message space/structure is chosen to accommodate this. This is a significant deviation from standard NTRU.
