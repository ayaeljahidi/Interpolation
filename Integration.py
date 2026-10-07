import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Define the function
# ============================================================
def f(x):
    return np.exp(-x**2)

a, b = 0.0, 1.0
true_value = 0.7468241328124271  # scipy.integrate.quad result

# ============================================================
# 2. Composite Trapezoidal Rule
# ============================================================
def composite_trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    I = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return I, x, y

# ============================================================
# 3. Composite Simpson's Rule
# ============================================================
def composite_simpson(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of subintervals.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    # weights: 1, 4, 2, 4, ..., 4, 1
    weights = np.ones(n + 1)
    weights[1:-1:2] = 4   # odd indices
    weights[2:-1:2] = 2   # even middle indices
    I = (h / 3) * np.sum(weights * y)
    return I, x, y

# ============================================================
# 4. Compute approximations
# ============================================================
I_trap4, x_t4, y_t4 = composite_trapezoidal(f, a, b, 4)
I_trap6, x_t6, y_t6 = composite_trapezoidal(f, a, b, 6)
I_simp4, x_s4, y_s4 = composite_simpson(f, a, b, 4)
I_simp6, x_s6, y_s6 = composite_simpson(f, a, b, 6)

print("=" * 55)
print(f"{'Method':<25}{'n':<5}{'Approx':<15}{'Error'}")
print("=" * 55)
print(f"{'Trapezoidal':<25}{4:<5}{I_trap4:<15.8f}{abs(I_trap4 - true_value):.2e}")
print(f"{'Trapezoidal':<25}{6:<5}{I_trap6:<15.8f}{abs(I_trap6 - true_value):.2e}")
print(f"{'Simpson':<25}{4:<5}{I_simp4:<15.8f}{abs(I_simp4 - true_value):.2e}")
print(f"{'Simpson':<25}{6:<5}{I_simp6:<15.8f}{abs(I_simp6 - true_value):.2e}")
print("=" * 55)
print(f"{'True value':<25}{'':<5}{true_value:.8f}")

# ============================================================
# 5. PLOT 1 — Original Function
# ============================================================
x_dense = np.linspace(a, b, 500)
plt.figure(figsize=(8, 5))
plt.plot(x_dense, f(x_dense), 'b-', linewidth=2, label=r'$f(x) = e^{-x^2}$')
plt.fill_between(x_dense, f(x_dense), alpha=0.25, color='blue',
                 label=f'Area = {true_value:.6f}')
plt.title(r'Original Function $f(x) = e^{-x^2}$ on $[0, 1]$')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 6. PLOT 2 — Trapezoidal Approximation (n = 4 and n = 6)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (n, x_n, y_n, I_n) in zip(
        axes,
        [(4, x_t4, y_t4, I_trap4), (6, x_t6, y_t6, I_trap6)]):

    # Draw trapezoids
    for i in range(n):
        xs = [x_n[i], x_n[i], x_n[i+1], x_n[i+1]]
        ys = [0, y_n[i], y_n[i+1], 0]
        ax.fill(xs, ys, alpha=0.35, edgecolor='darkorange', linewidth=1.2,
                facecolor='orange')

    # True curve
    ax.plot(x_dense, f(x_dense), 'b-', linewidth=2, label='True f(x)')
    ax.plot(x_n, y_n, 'o-', color='darkorange', markersize=6,
            label=f'Trapezoid nodes (n={n})')

    ax.set_title(f'Composite Trapezoidal, n={n}\n'
                 f'Approx = {I_n:.6f}, Error = {abs(I_n - true_value):.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 7. PLOT 3 — Simpson Approximation (n = 4 and n = 6)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (n, x_n, y_n, I_n) in zip(
        axes,
        [(4, x_s4, y_s4, I_simp4), (6, x_s6, y_s6, I_simp6)]):

    # Draw parabolas piece by piece (each spans 2 subintervals)
    for i in range(0, n, 2):
        # Fit a parabola through the 3 points x_i, x_{i+1}, x_{i+2}
        xs_fit = x_n[i:i+3]
        ys_fit = y_n[i:i+3]
        coeffs = np.polyfit(xs_fit, ys_fit, 2)
        xs_par = np.linspace(xs_fit[0], xs_fit[-1], 50)
        ys_par = np.polyval(coeffs, xs_par)
        ax.fill_between(xs_par, ys_par, alpha=0.35,
                        facecolor='green', edgecolor='darkgreen',
                        linewidth=1.2)

    ax.plot(x_dense, f(x_dense), 'b-', linewidth=2, label='True f(x)')
    ax.plot(x_n, y_n, 'o-', color='darkgreen', markersize=6,
            label=f'Simpson nodes (n={n})')

    ax.set_title(f"Composite Simpson, n={n}\n"
                 f"Approx = {I_n:.6f}, Error = {abs(I_n - true_value):.2e}")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# 8. PLOT 4 — Comparison Bar Chart
# ============================================================
methods = [
    'Trapezoid\nn=4', 'Trapezoid\nn=6',
    'Simpson\nn=4',   'Simpson\nn=6',
    'True'
]
values = [I_trap4, I_trap6, I_simp4, I_simp6, true_value]
errors = [abs(v - true_value) for v in values[:-1]] + [0]

colors = ['#f4a261', '#e76f51', '#2a9d8f', '#264653', 'blue']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: approximation values
bars = axes[0].bar(methods, values, color=colors, alpha=0.85, edgecolor='black')
axes[0].axhline(true_value, color='blue', linestyle='--', linewidth=2,
                label=f'True = {true_value:.6f}')
axes[0].set_ylim(0.74, 0.75)
axes[0].set_ylabel('Approximated integral')
axes[0].set_title('Approximation Values')
axes[0].legend()
axes[0].grid(True, axis='y', alpha=0.3)

for bar, v in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.00005,
                 f'{v:.6f}', ha='center', va='bottom', fontsize=8)

# Right: errors
err_methods = methods[:-1]
err_bars = axes[1].bar(err_methods, errors,
                       color=colors[:-1], alpha=0.85, edgecolor='black')
axes[1].set_yscale('log')
axes[1].set_ylabel('Absolute error (log scale)')
axes[1].set_title('Errors vs True Value')
axes[1].grid(True, axis='y', alpha=0.3, which='both')

for bar, e in zip(err_bars, errors):
    axes[1].text(bar.get_x() + bar.get_width()/2, e * 1.3,
                 f'{e:.2e}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()