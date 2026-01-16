import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def p_laplace_1d_fd(p_func, f_func, N=100, max_iter=50, tol=1e-8):
    """
    Solve 1D p(x)-Laplace equation using finite differences.
    """
    # Create grid
    x = np.linspace(0, 1, N+1)
    h = x[1] - x[0]

    # Evaluate p and f
    p_vals = p_func(x)
    f_vals = f_func(x)

    # Initial guess (sinusoidal)
    u = np.sin(np.pi * x)

    # Newton iteration
    for iteration in range(max_iter):
        # Compute gradient approximations
        du = np.diff(u) / h

        # Nonlinear coefficients
        eps = 1e-12  # Regularization
        a = (np.abs(du) + eps) ** (p_vals[:-1] - 2)

        # Construct tridiagonal matrix
        main_diag = np.zeros(N+1)
        lower_diag = np.zeros(N)
        upper_diag = np.zeros(N)

        # Interior points
        for i in range(1, N):
            main_diag[i] = (a[i-1] + a[i]) / h**2
            lower_diag[i-1] = -a[i-1] / h**2
            upper_diag[i] = -a[i] / h**2

        # Boundary conditions (Dirichlet)
        main_diag[0] = 1
        main_diag[N] = 1
        f_vals[0] = 0
        f_vals[N] = 0

        # Create sparse matrix
        A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')

        # Solve linear system
        u_new = spsolve(A, f_vals)

        # Check convergence
        residual = np.linalg.norm(u_new - u) / np.linalg.norm(u_new)
        u = u_new

        if residual < tol:
            print(f"Converged in {iteration+1} iterations")
            break

    return x, u

# Example usage
def p_example(x):
    return 1.5 + 0.5 * np.sin(2 * np.pi * x)

def f_example(x):
    return 10 * np.sin(np.pi * x)

# Solve and plot
x, u = p_laplace_1d_fd(p_example, f_example, N=200)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(x, u, 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, p_example(x), 'r-', linewidth=2)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Variable exponent')
plt.grid(True)

plt.subplot(1, 3, 3)
du = np.gradient(u, x)
energy = np.abs(du) ** p_example(x) / p_example(x)
plt.plot(x, energy, 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Energy density')
plt.title('Local energy')
plt.grid(True)

plt.tight_layout()
plt.show()