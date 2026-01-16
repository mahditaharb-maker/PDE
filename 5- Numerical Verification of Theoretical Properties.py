import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

class VariableExponentVerifier:
    """Verify theoretical properties numerically."""

    def __init__(self, domain=(0, 1)):
        self.domain = domain

    def verify_holder(self, p_func, q_func=None, num_tests=20):
        """Verify Holder inequality for random functions."""
        if q_func is None:
            # Use conjugate exponent
            def q_func(x):
                p_val = p_func(x)
                return p_val / (p_val - 1) if p_val != 1 else np.inf

        def s_func(x):
            p_val = p_func(x)
            q_val = q_func(x)
            if p_val == np.inf or q_val == np.inf:
                return max(p_val, q_val)
            return 1 / (1 / p_val + 1 / q_val)

        results = []

        for test in range(num_tests):
            # Generate random trigonometric functions
            np.random.seed(test)
            coeffs_f = np.random.randn(5)
            coeffs_g = np.random.randn(5)

            def random_trig(coeffs):
                def func(x):
                    result = 0
                    for k, c in enumerate(coeffs, 1):
                        result += c * np.sin(k * np.pi * x)
                    return result
                return func

            f = random_trig(coeffs_f)
            g = random_trig(coeffs_g)

            # Compute norms
            norm_f = self._compute_norm(f, p_func)
            norm_g = self._compute_norm(g, q_func)
            norm_fg = self._compute_norm(lambda x: f(x) * g(x), s_func)

            left = norm_fg
            right = 2 * norm_f * norm_g
            holds = left <= right + 1e-10  # Small tolerance

            results.append({
                'test': test,
                'norm_f_p': norm_f,
                'norm_g_q': norm_g,
                'norm_fg_s': left,
                'two_norms_product': right,
                'holds': holds,
                'ratio': left / right
            })

        return results

    def verify_poincare(self, p_func, num_tests=10):
        """Verify Poincare inequality: ||u||_p <= C||grad u||_p."""
        results = []

        for test in range(num_tests):
            # Generate random function with zero boundary
            np.random.seed(test)
            coeffs = np.random.randn(5)

            def u_func(x):
                result = 0
                for k, c in enumerate(coeffs, 1):
                    result += c * np.sin(k * np.pi * x)
                return result

            def du_func(x):
                result = 0
                for k, c in enumerate(coeffs, 1):
                    result += c * k * np.pi * np.cos(k * np.pi * x)
                return result

            # Compute norms
            norm_u = self._compute_norm(u_func, p_func)
            norm_du = self._compute_norm(du_func, p_func)

            if norm_du > 0:
                constant = norm_u / norm_du
                results.append({
                    'test': test,
                    'norm_u_p': norm_u,
                    'norm_du_p': norm_du,
                    'constant': constant
                })

        return results

    def _compute_norm(self, f_func, p_func, tol=1e-10):
        """Compute Luxemburg norm."""
        a, b = self.domain

        def modular(lam):
            def integrand(x):
                return np.abs(f_func(x) / lam) ** p_func(x)
            result, _ = quad(integrand, a, b, limit=200)
            return result - 1

        # Find root using bisection
        lam_low, lam_high = 1e-6, 100

        # Expand bounds if needed
        while modular(lam_low) > 0:
            lam_low /= 2

        while modular(lam_high) < 0:
            lam_high *= 2

        # Binary search
        for _ in range(50):
            lam_mid = (lam_low + lam_high) / 2
            val = modular(lam_mid)
            if val <= 0:
                lam_high = lam_mid
            else:
                lam_low = lam_mid

        return (lam_low + lam_high) / 2

    def analyze_reflexivity(self, p_func, num_dimensions=5):
        """
        Analyze uniform convexity as evidence of reflexivity.
        Compute modulus of convexity.
        """
        # Generate random unit vectors
        np.random.seed(42)
        vectors = []

        for _ in range(num_dimensions):
            coeffs = np.random.randn(5)
            def vec_func(x):
                result = 0
                for k, c in enumerate(coeffs, 1):
                    result += c * np.sin(k * np.pi * x)
                return result
            # Normalize
            norm = self._compute_norm(vec_func, p_func)
            if norm > 0:
                vectors.append((vec_func, norm))

        # Compute modulus of convexity for different epsilon
        epsilons = np.linspace(0.1, 0.9, 9)
        moduli = []

        for eps in epsilons:
            min_value = float('inf')
            # Sample pairs of vectors
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    f_func, norm_f = vectors[i]
                    g_func, norm_g = vectors[j]

                    # Normalize
                    def f_norm(x): return f_func(x) / norm_f
                    def g_norm(x): return g_func(x) / norm_g

                    # Compute distance
                    def diff_func(x): return f_norm(x) - g_norm(x)
                    distance = self._compute_norm(diff_func, p_func)

                    if distance >= eps:
                        # Compute average norm
                        def avg_func(x): return (f_norm(x) + g_norm(x)) / 2
                        avg_norm = self._compute_norm(avg_func, p_func)
                        min_value = min(min_value, 1 - avg_norm)

            if min_value < float('inf'):
                moduli.append((eps, min_value))

        return moduli

# Run verifications
verifier = VariableExponentVerifier()

# Test Holder inequality
print("Testing Holder inequality...")
p_func = lambda x: 2.0 + np.sin(2 * np.pi * x)
holder_results = verifier.verify_holder(p_func, num_tests=10)

print("\nHolder inequality results:")
print("Test   ||fg||_s     2||f||_p||g||_q   Holds    Ratio")
print("-" * 60)
for r in holder_results:
    print(f"{r['test']:4d}   {r['norm_fg_s']:10.6f}   {r['two_norms_product']:10.6f}   "
          f"{str(r['holds']):6s}   {r['ratio']:8.6f}")

# Test Poincare inequality
print("\n\nTesting Poincare inequality...")
poincare_results = verifier.verify_poincare(p_func, num_tests=8)

print("\nPoincare constant estimates:")
for r in poincare_results:
    print(f"Test {r['test']}: C <= {r['constant']:.6f}")

# Analyze uniform convexity
print("\n\nAnalyzing uniform convexity...")
moduli = verifier.analyze_reflexivity(p_func)

print("\nModulus of convexity delta(epsilon):")
print("epsilon   delta")
print("-" * 20)
for eps, delta in moduli:
    print(f"{eps:8.3f} {delta:8.6f}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Holder inequality ratios
ratios = [r['ratio'] for r in holder_results]
axes[0].bar(range(len(ratios)), ratios)
axes[0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Test number')
axes[0].set_ylabel('Ratio ||fg||_s / (2||f||_p||g||_q)')
axes[0].set_title('Holder inequality verification')
axes[0].grid(True, alpha=0.3)

# Poincare constants
constants = [r['constant'] for r in poincare_results]
axes[1].bar(range(len(constants)), constants)
axes[1].set_xlabel('Test number')
axes[1].set_ylabel('Poincare constant C')
axes[1].set_title('Poincare constant estimates')
axes[1].grid(True, alpha=0.3)

# Modulus of convexity
if moduli:
    eps_vals, delta_vals = zip(*moduli)
    axes[2].plot(eps_vals, delta_vals, 'bo-', linewidth=2)
    axes[2].set_xlabel('epsilon')
    axes[2].set_ylabel('delta(epsilon)')
    axes[2].set_title('Modulus of convexity')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()