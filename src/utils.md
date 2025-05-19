# `utils.py`

This file provides various utility functions for polynomial arithmetic, random polynomial generation, noise estimation, and other common operations needed by the FHE schemes.

* **`generate_poly(N: int, q: int) -> np.ndarray`**
  * **Purpose**: Generates a random polynomial of degree up to `N-1` with coefficients uniformly sampled from the symmetric range `[-q/2, q/2]`.
  * **Math Used**: Uniform random integer generation: `np.random.randint(-q // 2, q // 2 + 1, size=N)`. The interval is $[- \lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.
* **`generate_gaussian_poly(N: int, sigma: float) -> np.ndarray`**
  * **Purpose**: Generates a polynomial of degree `N-1` by sampling its coefficients from a discrete Gaussian (normal) distribution with mean 0 and standard deviation `sigma`.
  * **Math Used**:
    * Sampling from a continuous normal distribution: `scipy.stats.norm.rvs(loc=0, scale=sigma, size=N)`.
    * Rounding to the nearest integer: `np.round(...)`. This creates the discrete Gaussian sample.
* **`poly_mod(poly: np.ndarray, q: int) -> np.ndarray`**
  * **Purpose**: Reduces each coefficient of a polynomial modulo `q`, mapping them to the symmetric range `[-q/2, q/2]`.
  * **Math Used**: The formula effectively used is:
$coeff_{mod_q} = (coeff + \lfloor q/2 \rfloor) \pmod q - \lfloor q/2 \rfloor$
If `q` is odd, `q//2` is `(q-1)/2`. If `q` is even, `q//2` is `q/2`.
The implementation is `(poly + q // 2) % q - q // 2`.
* **`poly_add(poly1: np.ndarray, poly2: np.ndarray, q: int) -> np.ndarray`**
  * **Purpose**: Adds two polynomials `poly1` and `poly2` coefficient-wise, and then reduces each coefficient of the result modulo `q` using `poly_mod`.
  * **Math Used**:
    * Polynomial addition: `poly1 + poly2` (element-wise).
    * Coefficient-wise modular reduction via `poly_mod(sum_poly, q)`. Represents $(p_1(x) + p_2(x)) \pmod q$.
* **`poly_mul(poly1: np.ndarray, poly2: np.ndarray, q: int, N: int) -> np.ndarray`**
  * **Purpose**: Multiplies two polynomials `poly1` and `poly2` in the polynomial ring $R_{q,N} = \mathbb{Z}_q[x]/(x^N+1)$.
  * **Details**:

1. Standard polynomial multiplication (convolution): `result_full = numpy.polynomial.polynomial.polymul(poly1, poly2)`. This results in a polynomial of degree up to `2N-2`.
2. Reduction modulo $x^N+1$: This is done by substituting $x^N \equiv -1$, $x^{N+1} \equiv -x$, ..., $x^{N+k} \equiv -x^k$, etc. The loop implements this:
`remainder[i % N] += (-1)**(i // N) * result_full[i]`
For a term `c_i x^i` in `result_full`:
If $i < N$, it contributes `c_i` to `remainder[i]`. ($i // N = 0$)
If $N \le i < 2N$, let $i = N+k$ where $0 \le k < N$. Then $x^i = x^{N+k} = x^N x^k \equiv -x^k$. It contributes `-c_i` to `remainder[k]`. ($i // N = 1$)
And so on for higher degrees if `poly1`, `poly2` were not degree $<N$.
1. Coefficient-wise modular reduction of `remainder` modulo `q` using `poly_mod`.
    * **Math Used**: Polynomial multiplication (convolution). Ring arithmetic in $R_{q,N}$, specifically reduction modulo the cyclotomic polynomial $x^N+1$. Coefficient-wise modular reduction modulo $q$.

* **`poly_inverse_ntru(poly: np.ndarray, q: int) -> np.ndarray`**
  * **Purpose**: **Simulated/Placeholder.** Intended to compute the multiplicative inverse of a polynomial `poly` in the ring $R_{q,N} = \mathbb{Z}_q[x]/(x^N+1)$.
  * **Details**: The current implementation `return np.ones_like(poly)` simply returns a polynomial of all ones. This is not the correct inverse.
  * **Actual Math for Inverse**: To find $p(x)^{-1} \pmod{q, x^N+1}$:

1. The polynomial $p(x)$ must be invertible. This means its resultant with $x^N+1$ must be coprime to $q$ (i.e., $\gcd(\text{Res}(p(x), x^N+1), q) = 1$).
2. The extended Euclidean algorithm for polynomials is used. Given $p(x)$ and $m(x) = x^N+1$, it finds polynomials $u(x)$ and $v(x)$ such that: $p(x)u(x) + m(x)v(x) = \gcd(p(x), m(x))$
If this gcd is 1 (or an invertible constant modulo $q$), then $u(x)$ (after scaling by the inverse of the gcd if it's not 1) is the inverse of $p(x)$ modulo $m(x)$. All polynomial coefficient arithmetic is performed modulo $q$.

* **`estimate_noise_ntru(ciphertext: np.ndarray, secret_key: np.ndarray, q: int, N: int) -> float`**
  * **Purpose**: Estimates noise in what is presumed to be an intermediate NTRU value or ciphertext.
  * **Details**: It calculates `centered = poly_mod(ciphertext, q)` and then `np.std(centered)`. This measures the spread of the coefficients of `ciphertext` after centering them modulo `q`. The `secret_key` parameter is unused.
  * **Math Used**: `poly_mod` for centering. Standard deviation (`np.std`) of coefficients. This is a proxy for noise; true NTRU noise is typically defined as the difference between the value $a = f \cdot e \pmod q$ (after centering) and the closest multiple of $p$ times the message $m$, or related to the size of $r \cdot h$ vs $m$ in encryption.
* **`estimate_noise_rlwe(ciphertext: tuple, secret_key: np.ndarray, q: int, N: int) -> float`**
  * **Purpose**: Estimates noise in an RLWE ciphertext `(a,b)`.
  * **Details**: It computes $m+e = (b - a \cdot s) \pmod{q, x^N+1}$, where `s` is the `secret_key`. Then it calculates the standard deviation of the coefficients of `poly_mod(m+e, q)`.
  * **Math Used**: The core RLWE decryption step: $b - a \cdot s$. `poly_mod` for centering. Standard deviation (`np.std`) of the coefficients of the resulting $m+e$. If $m$ has small coefficients (or is 0 for noise measurement of a 0-encryption), this effectively gives $\text{std}(e)$ after ring operations.
* **`visualize_noise_distribution(ciphertext, secret_key=None, q=None, N=None, scheme="rlwe")`**
  * **Purpose**: Plots a histogram of the coefficient distribution of a ciphertext component, intended to visualize noise.
  * **Details**: For RLWE, it plots `poly_mod(ciphertext[^1], q)` (the `b` component). For NTRU, it plots `poly_mod(ciphertext, q)`. The title includes a noise estimate.
  * **Math Used**: Relies on the respective noise estimation functions. Histogram plotting.
* **`calculate_operational_depth(hybrid, max_operations=10)`**
  * **Purpose**: Estimates the maximum number of homomorphic multiplications (specifically, squarings) that can be performed on a ciphertext before decryption fails due to excessive noise.
  * **Details**: Encrypts a message. Then, in a loop, it squares the ciphertext (`ct = hybrid.mul(ct, ct)`), attempts decryption, and calculates the error between the decrypted message and the expected message (which also undergoes squaring: `expected_message = (expected_message ** 2) % hybrid.p` in a simplified sense, or more accurately `poly_mul(expected_message, expected_message, ...)`). The loop stops if decryption fails or error is too high.
  * **Math Used**:
    * `hybrid.encrypt`, `hybrid.mul`, `hybrid.decrypt`.
    * Calculation of expected message after squarings (conceptually $m \to m^2 \to m^4 \to \dots$).
    * Mean Absolute Error: $\text{MAE} = \frac{1}{N} \sum_{i=0}^{N-1} |\text{decrypted}_i - \text{expected}_i|$.
    * Comparison against a threshold.
