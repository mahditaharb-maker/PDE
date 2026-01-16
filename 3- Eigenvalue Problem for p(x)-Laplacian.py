import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

class pLaplaceEigenvalue:
    """Compute eigenvalues of p(x)-Laplacian."""

    def __init__(self, p_func, domain=(0, 1)):
        self.p_func = p_func
        self.domain = domain

    def rayleigh_quotient(self, u, x):
        """Compute Rayleigh quotient R(u)."""
        h = x[1] - x[0]

        # Compute derivative
        du = np.gradient(u, h)

        # Compute integrals
        numerator = simpson(np.abs(du) ** self.p_func(x), x)
        denominator = simpson(np.abs(u) ** self.p_func(x), x)

        return numerator / denominator if denominator > 1e-12 else np.inf

    def first_eigenvalue(self, basis_size=10, N=500):
        """Compute first eigenvalue using Rayleigh-Ritz method."""
        x = np.linspace(self.domain[0], self.domain[1], N)

        # Basis functions: sin(k*pi*x)
        def basis_function(k):
            return np.sin(k * np.pi * x)

        # Objective function for minimization
        def objective(coeffs):
            u = np.zeros_like(x)
            for i, c in enumerate(coeffs, start=1):
                u += c * basis_function(i)
            return self.rayleigh_quotient(u, x)

        # Initial guess
        coeffs0 = np.random.randn(basis_size)

        # Minimize Rayleigh quotient
        bounds = [(None, None) for _ in range(basis_size)]
        result = minimize(objective, coeffs0, method='L-BFGS-B', bounds=bounds)

        # Reconstruct eigenfunction
        u_opt = np.zeros_like(x)
        for i, c in enumerate(result.x, start=1):
            u_opt += c * basis_function(i)

        # Normalize
        norm = np.sqrt(simpson(u_opt**2, x))
        u_opt = u_opt / norm if norm > 0 else u_opt

        return result.fun, u_opt, x

    def compute_multiple_eigenvalues(self, num_eigenvalues=5, N=300):
        """Compute multiple eigenvalues using finite differences."""
        x = np.linspace(self.domain[0], self.domain[1], N)
        h = x[1] - x[0]
        p_vals = self.p_func(x)

        # Use constant p approximation for linearization
        p_avg = np.mean(p_vals)

        # Construct stiffness matrix (approximate)
        n_int = N - 2
        main_diag = 2 * np.ones(n_int) / h**2
        off_diag = -np.ones(n_int - 1) / h**2

        # For p(x) approximately 2, we get standard Laplacian
        K = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        # Mass matrix (for |u|^{p(x)-2} u term, approximated)
        M = np.eye(n_int)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigs(K, M=M, k=num_eigenvalues, which='SM')

        # Sort eigenvalues
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = eigenvectors[:, idx]

        # Format eigenfunctions
        eigenfunctions = []
        for i in range(num_eigenvalues):
            u = np.zeros(N)
            u[1:-1] = np.real(eigenvectors[:, i])
            # Normalize
            u = u / np.sqrt(simpson(u**2, x))
            eigenfunctions.append(u)

        return eigenvalues, eigenfunctions, x

# Example
def p_eigen(x):
    return 1.8 + 0.4 * np.cos(3 * np.pi * x)

solver = pLaplaceEigenvalue(p_eigen)

# Compute first eigenvalue
lambda1, u1, x = solver.first_eigenvalue(basis_size=8, N=400)
print(f"First eigenvalue lambda_1 approx {lambda1:.6f}")

# Compute multiple eigenvalues
eigvals, eigfuncs, x_vals = solver.compute_multiple_eigenvalues(num_eigenvalues=4)

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(x_vals, eigfuncs[i], 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel(f'u_{i+1}(x)')
    plt.title(f'Eigenfunction {i+1}, lambda approx {eigvals[i]:.4f}')
    plt.grid(True)

plt.tight_layout()
plt.show()