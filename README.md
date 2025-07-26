<h1 align="center">🔐 NTRU-RLWE Hybrid Scheme Demonstration</h1>

<p align = "center">
  <img src=".assets/pixmagicar.png" alt="Coding Car" title="Coding Car" width="128">
</p>

---

<p align="center">

| S. No |    ID No.     | Name                                                   |
| ----: | :-----------: | :------------------------------------------------------|
|    1. | 2022A7PS0004U | [Yusra Hakim](https://github.com/Sapphire-Starflower)  |
|    2. | 2022A7PS0019U | [Joseph Cijo](https://github.com/joejo-joestar)        |

</p>

This project demonstrates the NTRU-RLWE hybrid scheme for homomorphic encryption, focusing on its performance in terms of encryption, decryption, and bootstrapping operations. The demonstration includes a comparison with existing schemes like CKKS and BVK using the TenSEAL library.

## Requirements

OS: Linux (WSL2)

Python version: 3.10

Dependencies:

- tenseal: library for homomorphic encryption
- numpy: for numerical operations
- pandas: for data manipulation
- matplotlib: for plotting graphs
- seaborn: for statistical data visualization

## Demonstration

To run the project, follow these steps:

1. Install the required Python packages:

    ```bash
    pip install -r requirements.txt

    ```

2. Run the main script:

    ```bash
    python3 src/demo.py
    ```

3. The results will be saved in the `results` directory.
