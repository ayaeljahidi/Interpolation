"""
Polynomial interpolation of f(x) = e^x
======================================

Compares three classical interpolation methods on the nodes
x = -1, 0, 1, 2 :

    1. Lagrange interpolation   (degree <= 3)
    2. Newton interpolation     (degree <= 3, divided differences)
    3. Hermite interpolation    (degree <= 7, uses f and f' at each node)

Everything is computed symbolically with SymPy (exact results), then
converted to NumPy functions for plotting and error analysis.

Usage
-----
    python interpolation.py            # saves figures in ./figures/
    python interpolation.py --show     # also opens the plot windows
"""

import argparse
import os

import matplotlib

import numpy as np
import sympy as sp

# ------------------------------------------------------------
# Parse options BEFORE importing pyplot so the backend can be
# chosen: without --show we use a non-interactive backend, which
# makes the script work on servers / CI without a display.
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Lagrange / Newton / Hermite interpolation of e^x")
parser.add_argument("--show", action="store_true", help="display the figures in addition to saving them")
ARGS = parser.parse_args()

if not ARGS.show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must come after backend selection)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def finish_figure(filename):
    """Save the current figure, and show it only if --show was given."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    if ARGS.show:
        plt.show()
    plt.close()


# ============================================================
# 1. FUNCTION AND INTERPOLATION POINTS
# ============================================================

x = sp.symbols("x")

f = sp.exp(x)                                  # original function
x_points = [-1, 0, 1, 2]                       # interpolation nodes
y_points = [f.subs(x, xi) for xi in x_points]  # f(x_i)
dy_points = [sp.diff(f, x).subs(x, xi) for xi in x_points]  # f'(x_i), used by Hermite

header("FUNCTION AND INTERPOLATION POINTS")
print("Function: f(x) = e^x")
print("\nPoints:")
for xi, yi in zip(x_points, y_points):
    print(f"x = {xi:2d}  -->  f(x) = {yi}  ~ {float(yi):.6f}")


# ============================================================
# 2. LAGRANGE INTERPOLATION
# ============================================================

def lagrange_interpolation(x_points, y_points, x):
    """Return P(x) = sum_i y_i * L_i(x) as an expanded SymPy expression."""
    polynomial = 0
    for i in range(len(x_points)):
        L_i = 1
        for j in range(len(x_points)):
            if i != j:
                L_i *= (x - x_points[j]) / sp.Integer(x_points[i] - x_points[j])
        polynomial += y_points[i] * L_i
    return sp.expand(polynomial)


P_lagrange = lagrange_interpolation(x_points, y_points, x)

header("LAGRANGE INTERPOLATION")
print("\nLagrange polynomial (collected in powers of x):")
print(sp.collect(P_lagrange, x))


# ============================================================
# 3. NEWTON INTERPOLATION
# ============================================================

def divided_difference_table(x_points, y_points):
    """Divided-difference table; table[i][j] = f[x_i, ..., x_{i+j}]."""
    n = len(x_points)
    table = [[sp.S(0) for _ in range(n)] for _ in range(n)]

    for i in range(n):
        table[i][0] = y_points[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = sp.expand(
                (table[i + 1][j - 1] - table[i][j - 1])
                / sp.Integer(x_points[i + j] - x_points[i])
            )
    return table


def newton_interpolation(x_points, y_points, x):
    """Return the Newton polynomial (expanded) and its divided-difference table."""
    table = divided_difference_table(x_points, y_points)
    n = len(x_points)

    polynomial = table[0][0]
    product = 1
    for i in range(1, n):
        product *= (x - x_points[i - 1])
        polynomial += table[0][i] * product

    return sp.expand(polynomial), table


P_newton, newton_table = newton_interpolation(x_points, y_points, x)

header("NEWTON INTERPOLATION")
print("\nDivided-difference table (one row per node):")
for i in range(len(x_points)):
    print([str(newton_table[i][j]) for j in range(len(x_points) - i)])

print("\nNewton polynomial:")
print(sp.collect(P_newton, x))


# ============================================================
# 4. HERMITE INTERPOLATION
# ============================================================

def hermite_interpolation(x_points, y_points, derivative_values, x):
    """
    Hermite interpolation with divided differences on doubled nodes
    z = [x0, x0, x1, x1, ...].  The resulting polynomial has degree <= 2n-1
    and satisfies P(x_i) = y_i and P'(x_i) = f'(x_i).
    """
    n = len(x_points)

    z = []
    for xi in x_points:
        z.extend([xi, xi])

    size = 2 * n
    # Upper-triangular convention (same as the Newton table):
    #   Q[i][j] = f[z_i, z_{i+1}, ..., z_{i+j}]
    Q = [[sp.S(0) for _ in range(size)] for _ in range(size)]

    # Column 0: function values (each node appears twice)
    for i in range(n):
        Q[2 * i][0] = y_points[i]
        Q[2 * i + 1][0] = y_points[i]

    # Column 1: f[z_k, z_{k+1}] -- equals f'(x_i) when the two nodes coincide
    for k in range(size - 1):
        if z[k + 1] == z[k]:
            Q[k][1] = derivative_values[k // 2]
        else:
            Q[k][1] = sp.expand((Q[k + 1][0] - Q[k][0]) / sp.Integer(z[k + 1] - z[k]))

    # Remaining columns: standard divided differences (z[i+j] != z[i] for j >= 2)
    for j in range(2, size):
        for i in range(size - j):
            Q[i][j] = sp.expand(
                (Q[i + 1][j - 1] - Q[i][j - 1]) / sp.Integer(z[i + j] - z[i])
            )

    polynomial = Q[0][0]
    product = 1
    for j in range(1, size):
        product *= (x - z[j - 1])
        polynomial += Q[0][j] * product

    return sp.expand(polynomial), Q, z


P_hermite, hermite_table, z = hermite_interpolation(x_points, y_points, dy_points, x)

header("HERMITE INTERPOLATION")
print("\nDerivative values (f'(x) = e^x):")
for xi, dyi in zip(x_points, dy_points):
    print(f"f'({xi}) = {dyi}")

print("\nHermite divided-difference table (one row per doubled node):")
for i in range(len(z)):
    print([str(hermite_table[i][j]) for j in range(len(z) - i)])

print("\nHermite polynomial:")
print(sp.collect(P_hermite, x))


# ============================================================
# 5. SANITY CHECKS (new)
# ============================================================

header("SANITY CHECKS")

# (a) Interpolation conditions
for name, P in (("Lagrange", P_lagrange), ("Newton", P_newton), ("Hermite", P_hermite)):
    ok = all(sp.expand(P.subs(x, xi) - yi) == 0 for xi, yi in zip(x_points, y_points))
    print(f"{name:9s}: P(x_i) = f(x_i) at all nodes ........ {ok}")

dP = sp.diff(P_hermite, x)
ok = all(sp.expand(dP.subs(x, xi) - dyi) == 0 for xi, dyi in zip(x_points, dy_points))
print(f"{'Hermite':9s}: P'(x_i) = f'(x_i) at all nodes ...... {ok}")

# (b) Uniqueness: Lagrange and Newton must be the SAME polynomial
same = sp.expand(P_lagrange - P_newton) == 0
print(f"Lagrange == Newton (uniqueness theorem) ........ {same}")

print(f"\ndegree(Lagrange) = {sp.degree(P_lagrange, x)}")
print(f"degree(Newton)   = {sp.degree(P_newton, x)}")
print(f"degree(Hermite)  = {sp.degree(P_hermite, x)}")


# ============================================================
# 6. NUMERICAL FUNCTIONS FOR PLOTTING
# ============================================================

f_numeric = sp.lambdify(x, f, "numpy")
lagrange_numeric = sp.lambdify(x, P_lagrange, "numpy")
newton_numeric = sp.lambdify(x, P_newton, "numpy")
hermite_numeric = sp.lambdify(x, P_hermite, "numpy")

x_plot = np.linspace(-1.2, 2.2, 500)
y_original = f_numeric(x_plot)
y_lagrange = lagrange_numeric(x_plot)
y_newton = newton_numeric(x_plot)
y_hermite = hermite_numeric(x_plot)

nodes_float = [float(xi) for xi in x_points]
values_float = [float(yi) for yi in y_points]


# ============================================================
# 7. INDIVIDUAL PLOTS
# ============================================================

def single_plot(y_interp, label, title, filename, style="--"):
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_original, label="f(x) = $e^x$", linewidth=2)
    plt.plot(x_plot, y_interp, label=label, linestyle=style, linewidth=2)
    plt.scatter(nodes_float, values_float, label="Interpolation points", zorder=5, color="black")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    finish_figure(filename)


single_plot(y_lagrange, "Lagrange interpolation", "Lagrange Interpolation", "lagrange.png")
single_plot(y_newton, "Newton interpolation", "Newton Interpolation", "newton.png")
single_plot(y_hermite, "Hermite interpolation", "Hermite Interpolation", "hermite.png")


# ============================================================
# 8. COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(12, 7))
plt.plot(x_plot, y_original, label="Original f(x) = $e^x$", linewidth=3)
plt.plot(x_plot, y_lagrange, label="Lagrange", linestyle="--")
plt.plot(x_plot, y_newton, label="Newton", linestyle=":", linewidth=2)
plt.plot(x_plot, y_hermite, label="Hermite", linestyle="-.")
plt.scatter(nodes_float, values_float, label="Interpolation points", zorder=5, color="black")
plt.title("Comparison of Interpolation Methods")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
finish_figure("comparison.png")


# ============================================================
# 9. ERROR ANALYSIS
# ============================================================

error_lagrange = np.abs(y_original - y_lagrange)
error_newton = np.abs(y_original - y_newton)
error_hermite = np.abs(y_original - y_hermite)

inside = (x_plot >= -1.0) & (x_plot <= 2.0)   # interpolation interval [-1, 2]

header("MAXIMUM ABSOLUTE ERROR")
print(f"{'Method':10s}{'on [-1, 2]':>18s}{'on [-1.2, 2.2]':>20s}")
for name, err in (("Lagrange", error_lagrange), ("Newton", error_newton), ("Hermite", error_hermite)):
    print(f"{name:10s}{np.max(err[inside]):18.6e}{np.max(err):20.6e}")

# Theoretical bounds on [-1, 2]  (|f^(k)| <= e^2 there)
M = float(sp.exp(2))
node_poly_L = np.prod([x_plot - xi for xi in nodes_float], axis=0)
bound_L = M / sp.factorial(4) * np.max(np.abs(node_poly_L[inside]))
bound_H = M / sp.factorial(8) * np.max(np.abs(node_poly_L[inside]) ** 2)

print("\nTheoretical error bounds on [-1, 2]:")
print(f"  Lagrange / Newton : e^2/4! * max|w(x)|      = {float(bound_L):.6e}")
print(f"  Hermite           : e^2/8! * max|w(x)|^2    = {float(bound_H):.6e}")


# ============================================================
# 10. ERROR GRAPHS
# ============================================================

# Linear scale (Lagrange and Newton curves coincide)
plt.figure(figsize=(12, 7))
plt.plot(x_plot, error_lagrange, label="Lagrange error", linewidth=3)
plt.plot(x_plot, error_newton, label="Newton error", linestyle="--", linewidth=2)
plt.plot(x_plot, error_hermite, label="Hermite error")
plt.title("Interpolation Error Comparison")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.legend()
finish_figure("error.png")

# Logarithmic scale (reveals the Hermite error, which is tiny)
plt.figure(figsize=(12, 7))
plt.semilogy(x_plot, error_lagrange + 1e-18, label="Lagrange error", linewidth=3)
plt.semilogy(x_plot, error_newton + 1e-18, label="Newton error", linestyle="--", linewidth=2)
plt.semilogy(x_plot, error_hermite + 1e-18, label="Hermite error")
plt.title("Interpolation Error Comparison (log scale)")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.grid(True, which="both", alpha=0.4)
plt.legend()
finish_figure("error_log.png")

print(f"\nFigures saved in ./{FIG_DIR}/")
