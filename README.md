# Lagrange, Newton & Hermite Interpolation of eˣ

Numerical Analysis homework: three classical polynomial interpolation methods applied to
**f(x) = eˣ** at the nodes **x = −1, 0, 1, 2**, computed symbolically with SymPy,
compared graphically, and checked against the theoretical error bounds.

| Method   | Data used     | Degree | Max error on [−1, 2] |
|----------|---------------|:------:|:--------------------:|
| Lagrange | f(xᵢ)         |   3    | 9.47 × 10⁻²          |
| Newton   | f(xᵢ)         |   3    | 9.47 × 10⁻²          |
| Hermite  | f(xᵢ), f′(xᵢ) |   7    | 4.92 × 10⁻⁵          |

Key takeaways:

- **Lagrange and Newton give exactly the same polynomial** (uniqueness of the interpolating
  polynomial); they only differ in how it is written and computed.
- **Hermite** also matches the derivative at every node, giving a degree-7 polynomial that is
  about **2000× more accurate** on [−1, 2].

The full write-up (theory, hand computation, figures, error analysis) is in
[`report/report.pdf`](report/report.pdf).
