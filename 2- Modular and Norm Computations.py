import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

class VariableExponentSpace:
    """Class for computations in L^{p(x)} spaces."""

    def __init__(self, p_func, domain=(0, 1)):
        self.p_func = p_func
        self.domain = domain

    def modular(self, f_func):
        """Compute modular rho(f) = integral |f(x)|^{p(x)} dx."""
        a, b = self.domain

        def integrand(x):
            return np.abs(f_func(x)) ** self.p_func(x)

        result, error = quad(integrand, a, b, limit=200, epsabs=1e-12)
        return result, error

    def norm(self, f_func, tol=1e-10):
        """Compute Luxemburg norm using root-finding."""
        a, b = self.domain

        def modular_at_lambda(lam):
            def integrand(x):
                return np.abs(f_func(x) / lam) ** self.p_func(x)
            result, _ = quad(integrand, a, b, limit=200)
            return result - 1

        # Find lambda such that rho(f/lambda) = 1
        lam_low, lam_high = 1e-6, 1000

        # Ensure we bracket the root
        while modular_at_lambda(lam_low) > 0:
            lam_low /= 2

        while modular_at_lambda(lam_high) < 0:
            lam_high *= 2

        # Find root
        lam_norm = brentq(modular_at_lambda, lam_low, lam_high, xtol=tol)
        return lam_norm

    def holder_product(self, f_func, g_func, q_func=None):
        """Verify Holder inequality: ||fg||_s <= 2||f||_p ||g||_q."""
        if q_func is None:
            # Default: q is conjugate of p
            def q_func(x):
                p_val = self.p_func(x)
                return p_val / (p_val - 1) if p_val != 1 else np.inf

        # Compute s(x) = 1/(1/p(x) + 1/q(x))
        def s_func(x):
            p_val = self.p_func(x)
            q_val = q_func(x)
            if p_val == np.inf or q_val == np.inf:
                return max(p_val, q_val)
            return 1 / (1 / p_val + 1 / q_val)

        # Define product function
        def fg_func(x):
            return f_func(x) * g_func(x)

        # Create spaces for p, q, s
        space_p = VariableExponentSpace(self.p_func, self.domain)
        space_q = VariableExponentSpace(q_func, self.domain)
        space_s = VariableExponentSpace(s_func, self.domain)

        # Compute norms
        norm_f = space_p.norm(f_func)
        norm_g = space_q.norm(g_func)
        norm_fg = space_s.norm(fg_func)

        left_side = norm_fg
        right_side = 2 * norm_f * norm_g

        return {
            'norm_fg_s': left_side,
            'two_norm_f_norm_g': right_side,
            'inequality_holds': left_side <= right_side,
            'ratio': left_side / right_side
        }

# Example usage
def p_var(x):
    return 2.0 + np.sin(2 * np.pi * x)

def f_test(x):
    return np.sin(3 * np.pi * x)

def g_test(x):
    return np.cos(4 * np.pi * x)

# Create space and compute
space = VariableExponentSpace(p_var)
modular_val, error = space.modular(f_test)
norm_val = space.norm(f_test)

print(f"Modular rho(f) = {modular_val:.6f} +/- {error:.2e}")
print(f"Norm ||f||_p(x) = {norm_val:.6f}")

# Test Holder inequality
result = space.holder_product(f_test, g_test)
print("\nHolder inequality test:")
print(f"||fg||_s = {result['norm_fg_s']:.6f}")
print(f"2||f||_p||g||_q = {result['two_norm_f_norm_g']:.6f}")
print(f"Inequality holds: {result['inequality_holds']}")
print(f"Ratio: {result['ratio']:.6f}")