# CUDA_prep
Ramping up on parallel programming basics and gpu architecture experiments.

## Summary of Benchmarks

### 🔴 `vec_opN` Kernel

Approach: Execute a synthetic vector operation that performs a fixed number of floating-point operations (FLOPs) per memory element accessed. For a given number of FLOPs per element (`FLOP_COUNT`), each thread executes:

$$
C[i] = \sum_{j=1}^{\mathrm{FLOP\_COUNT}/4} \left( A[i] \cdot B[i] + D[i] \cdot E[i] \right)
$$

#### Total FLOPs:
$$
\mathrm{FLOPs} = N \cdot \mathrm{FLOP\_COUNT}
$$

#### Memory Bytes Moved:
We read from `A[i]`, `B[i]`, `D[i]`, `E[i]` and write to `C[i]`:

$$
\text{Bytes} = 5 \cdot N \cdot \text{sizeof(float)}
$$

#### Arithmetic Intensity:
$$
\text{Intensity} = \frac{N \cdot \mathrm{FLOP\_COUNT}}{5 \cdot N \cdot 4} = \frac{\mathrm{FLOP\_COUNT}}{20}
$$

> The `vec_opN` kernel is used to **sweep arithmetic intensity** by increasing `FLOP_COUNT` and analyzing how performance shifts from memory-bound to compute-bound behavior.

---

### ⚫ `matmul` Kernel

We investigated standard dense matrix multiplication:

$$
C_{ij} = \sum_{k=1}^{N} A_{ik} \cdot B_{kj}
$$

#### Total FLOPs:
$$
\mathrm{FLOPs} = 2 \cdot N^3
$$

#### Memory Bytes Moved:
We read matrices `A`, `B` and write `C`, each of size $N \times N$:

$$
\text{Bytes} = 3 \cdot N^2 \cdot \text{sizeof(float)}
$$

#### Arithmetic Intensity:
$$
\text{Intensity} = \frac{2 \cdot N^3}{3 \cdot N^2 \cdot 4} = \frac{N}{6}
$$

> The `matmul` kernel evaluates a more realistic compute-bound workload where intensity increases with matrix size.

---

These two benchmarks allow us to **sweep across a range of arithmetic intensities**, visualizing memory- and compute-bound regions on the roofline plot of the RTX 3090. Red dots represent `vec_opN`, and black/gray triangles represent `matmul`.



![Screenshot 2025-05-11 223540](https://github.com/user-attachments/assets/50f46948-5c92-433d-bfc6-b1dbd46eb54a)
