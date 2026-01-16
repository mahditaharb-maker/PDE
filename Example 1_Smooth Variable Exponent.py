import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import root

def example_smooth_exponent():
    """Example with smooth p(x) = 2 + sin(pi*x)."""
    # Define functions
    def p_smooth(x):
        return 2 + np.sin(np.pi * x)
    
    def f_smooth(x):
        return 10 * np.exp(-10 * (x - 0.5)**2)  # Gaussian source
    
    # Grid for visualization
    x = np.linspace(0, 1, 1000)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot p(x)
    axes[0, 0].plot(x, p_smooth(x), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('p(x)')
    axes[0, 0].set_title('Variable exponent: p(x) = 2 + sin(pi*x)')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='p=2 (Laplacian)')
    axes[0, 0].legend()
    
    # Plot f(x)
    axes[0, 1].plot(x, f_smooth(x), 'r-', linewidth=2)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')
    axes[0, 1].set_title('Source term: Gaussian')
    axes[0, 1].grid(True)
    
    # Compute and plot exact solution for constant p=2 case
    N = 200
    x_fd = np.linspace(0, 1, N+1)
    h = 1/N
    
    # Solve for p=2 (Laplacian) as reference
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(N-1, N-1)) / h**2
    f_vals = f_smooth(x_fd[1:-1])
    u_const = spsolve(A, f_vals)
    u_const_full = np.zeros(N+1)
    u_const_full[1:-1] = u_const
    
    axes[0, 2].plot(x_fd, u_const_full, 'g-', linewidth=2)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('u(x)')
    axes[0, 2].set_title('Solution for p=2 (Laplacian)')
    axes[0, 2].grid(True)
    
    # Now solve for variable p(x) using finite differences
    def p_laplace_residual(u):
        """Residual for p(x)-Laplace equation."""
        residual = np.zeros_like(u)
        h = 1/(len(u)-1)
        
        for i in range(1, len(u)-1):
            p_left = p_smooth(x_fd[i] - h/2)
            p_right = p_smooth(x_fd[i] + h/2)
            
            du_left = (u[i] - u[i-1]) / h
            du_right = (u[i+1] - u[i]) / h
            
            # Nonlinear fluxes
            flux_left = np.abs(du_left)**(p_left-2) * du_left
            flux_right = np.abs(du_right)**(p_right-2) * du_right
            
            residual[i] = (flux_right - flux_left)/h - f_smooth(x_fd[i])
        
        # Boundary conditions
        residual[0] = u[0]
        residual[-1] = u[-1]
        
        return residual
    
    # Initial guess (solution for p=2)
    u_guess = u_const_full.copy()
    
    # Solve using Newton's method
    solution = root(p_laplace_residual, u_guess, method='krylov', tol=1e-8)
    u_var = solution.x
    
    axes[1, 0].plot(x_fd, u_var, 'b-', linewidth=2, label='Variable p(x)')
    axes[1, 0].plot(x_fd, u_const_full, 'r--', linewidth=2, label='Constant p=2')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u(x)')
    axes[1, 0].set_title('Comparison: Variable vs Constant p')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot gradient comparison
    grad_const = np.gradient(u_const_full, x_fd)
    grad_var = np.gradient(u_var, x_fd)
    
    axes[1, 1].plot(x_fd, grad_var, 'b-', linewidth=2, label='Variable p(x)')
    axes[1, 1].plot(x_fd, grad_const, 'r--', linewidth=2, label='Constant p=2')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('du/dx')
    axes[1, 1].set_title('Gradient comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot energy density
    energy_const = 0.5 * grad_const**2  # For p=2
    energy_var = np.abs(grad_var)**p_smooth(x_fd) / p_smooth(x_fd)
    
    axes[1, 2].plot(x_fd, energy_var, 'b-', linewidth=2, label='Variable p(x)')
    axes[1, 2].plot(x_fd, energy_const, 'r--', linewidth=2, label='Constant p=2')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Energy density')
    axes[1, 2].set_title('Energy density comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Quantitative analysis
    print("=" * 60)
    print("QUANTITATIVE ANALYSIS: Smooth Exponent Example")
    print("=" * 60)
    
    # Compute norms
    def compute_norm(u, p_vals):
        """Compute approximate Luxemburg norm."""
        h = 1/(len(u)-1)
        integral = np.sum(np.abs(u)**p_vals) * h
        
        # Find lambda such that integral = 1
        lam_low, lam_high = 0.01, 10
        for _ in range(50):
            lam_mid = (lam_low + lam_high) / 2
            if np.sum(np.abs(u/lam_mid)**p_vals) * h > 1:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
        
        return (lam_low + lam_high) / 2
    
    norm_const = np.linalg.norm(u_const_full)
    norm_var = compute_norm(u_var, p_smooth(x_fd))
    
    print(f"\nSolution norms:")
    print(f"L2 norm (constant p=2): {norm_const:.6f}")
    print(f"L^p(x) norm (variable p): {norm_var:.6f}")
    
    # Maximum values
    print(f"\nMaximum values:")
    print(f"max|u| (constant p=2): {np.max(np.abs(u_const_full)):.6f}")
    print(f"max|u| (variable p): {np.max(np.abs(u_var)):.6f}")
    
    # Gradient statistics
    print(f"\nGradient statistics:")
    print(f"max|du/dx| (constant p=2): {np.max(np.abs(grad_const)):.6f}")
    print(f"max|du/dx| (variable p): {np.max(np.abs(grad_var)):.6f}")
    
    # Energy comparison
    total_energy_const = np.sum(energy_const) * h
    total_energy_var = np.sum(energy_var) * h
    
    print(f"\nTotal energy:")
    print(f"Constant p=2: {total_energy_const:.6f}")
    print(f"Variable p(x): {total_energy_var:.6f}")
    print(f"Ratio (variable/constant): {total_energy_var/total_energy_const:.6f}")
    
    return u_const_full, u_var, x_fd

# Run example
u_const, u_var, x = example_smooth_exponent()