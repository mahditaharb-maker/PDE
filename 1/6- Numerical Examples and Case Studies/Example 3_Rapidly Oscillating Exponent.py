import numpy as np
import matplotlib.pyplot as plt

def example_oscillating_exponent():
    """Example with rapidly oscillating exponent."""
    
    def p_oscillating(x):
        return 2 + np.sin(10 * np.pi * x)
    
    def f_oscillating(x):
        return 5 * np.cos(2 * np.pi * x)
    
    # High resolution grid
    N = 1000
    x = np.linspace(0, 1, N+1)
    h = 1/N
    
    # Solve using finite differences
    def solve_p_laplace_fd(p_func, f_func, u_init=None, max_iter=200, tol=1e-8):
        """Finite difference solver for p(x)-Laplace."""
        N = len(x) - 1
        h = 1/N
        
        if u_init is None:
            u = np.sin(np.pi * x)
        else:
            u = u_init.copy()
        
        for iteration in range(max_iter):
            u_old = u.copy()
            
            # Evaluate p at midpoints
            p_mid = (p_func(x[:-1]) + p_func(x[1:])) / 2
            
            residual = np.zeros(N+1)
            
            for i in range(1, N):
                du_left = (u[i] - u[i-1]) / h
                du_right = (u[i+1] - u[i]) / h
                
                coeff_left = (np.abs(du_left) + 1e-12)**(p_mid[i-1] - 2)
                coeff_right = (np.abs(du_right) + 1e-12)**(p_mid[i] - 2)
                
                residual[i] = (coeff_right * du_right - coeff_left * du_left) / h - f_func(x[i])
            
            # Boundary conditions
            residual[0] = u[0]
            residual[-1] = u[-1]
            
            # Simple relaxation
            u = u - 0.1 * residual
            
            if np.linalg.norm(u - u_old) / np.linalg.norm(u) < tol:
                print(f"Converged in {iteration+1} iterations")
                break
        
        return u
    
    # Solve
    u = solve_p_laplace_fd(p_oscillating, f_oscillating)
    
    # Compute homogenized p (average)
    p_avg = np.mean(p_oscillating(x))
    
    # Solve homogenized problem (constant p = average)
    def solve_constant_p(p_const, f_func):
        """Solve constant p Laplacian."""
        N = len(x) - 1
        h = 1/N
        
        if abs(p_const - 2) < 1e-10:
            # Linear case
            A = np.zeros((N-1, N-1))
            for i in range(N-1):
                A[i, i] = 2/h**2
                if i > 0:
                    A[i, i-1] = -1/h**2
                if i < N-2:
                    A[i, i+1] = -1/h**2
            
            f_vals = f_func(x[1:-1])
            u_int = np.linalg.solve(A, f_vals)
            u_full = np.zeros(N+1)
            u_full[1:-1] = u_int
            return u_full
        else:
            # Nonlinear constant p
            u_guess = np.sin(np.pi * x)
            return solve_p_laplace_fd(lambda x: p_const * np.ones_like(x), f_func, u_guess)
    
    u_homog = solve_constant_p(p_avg, f_oscillating)
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot p(x)
    axes[0, 0].plot(x, p_oscillating(x), 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(y=p_avg, color='r', linestyle='--', linewidth=2, 
                      label=f'Average p = {p_avg:.3f}')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('p(x)')
    axes[0, 0].set_title('Rapidly oscillating exponent: p(x) = 2 + sin(10*pi*x)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Zoom in on p(x)
    axes[0, 1].plot(x, p_oscillating(x), 'b-', linewidth=1.5)
    axes[0, 1].set_xlim(0.4, 0.6)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('p(x)')
    axes[0, 1].set_title('Zoom: Oscillations in p(x)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot solutions
    axes[1, 0].plot(x, u, 'b-', linewidth=2, label='Variable p(x)')
    axes[1, 0].plot(x, u_homog, 'r--', linewidth=2, label=f'Homogenized p={p_avg:.3f}')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u(x)')
    axes[1, 0].set_title('Solution comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot difference
    axes[1, 1].plot(x, u - u_homog, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Difference')
    axes[1, 1].set_title('Difference: u(x) - u_homog(x)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot gradients
    grad_u = np.gradient(u, x)
    grad_homog = np.gradient(u_homog, x)
    
    axes[2, 0].plot(x, grad_u, 'b-', linewidth=2, label='Variable p(x)', alpha=0.7)
    axes[2, 0].plot(x, grad_homog, 'r--', linewidth=2, label='Homogenized', alpha=0.7)
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('du/dx')
    axes[2, 0].set_title('Gradient comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot energy densities
    energy_u = np.abs(grad_u)**p_oscillating(x) / p_oscillating(x)
    energy_homog = np.abs(grad_homog)**p_avg / p_avg
    
    axes[2, 1].plot(x, energy_u, 'b-', linewidth=2, label='Variable p(x)', alpha=0.7)
    axes[2, 1].plot(x, energy_homog, 'r--', linewidth=2, label='Homogenized', alpha=0.7)
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('Energy density')
    axes[2, 1].set_title('Energy density comparison')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis
    print("=" * 60)
    print("STATISTICAL ANALYSIS: Rapidly Oscillating Exponent")
    print("=" * 60)
    
    print(f"\nExponent statistics:")
    print(f"Minimum p(x): {np.min(p_oscillating(x)):.6f}")
    print(f"Maximum p(x): {np.max(p_oscillating(x)):.6f}")
    print(f"Average p(x): {p_avg:.6f}")
    print(f"Standard deviation: {np.std(p_oscillating(x)):.6f}")
    
    print(f"\nSolution statistics:")
    print(f"Mean |u(x)|: {np.mean(np.abs(u)):.6f}")
    print(f"Std |u(x)|: {np.std(u):.6f}")
    print(f"Mean |u_homog(x)|: {np.mean(np.abs(u_homog)):.6f}")
    print(f"Std |u_homog(x)|: {np.std(u_homog):.6f}")
    
    # Error analysis
    L2_error = np.sqrt(np.sum((u - u_homog)**2) * h)
    Linf_error = np.max(np.abs(u - u_homog))
    
    print(f"\nError analysis:")
    print(f"L2 error: {L2_error:.6e}")
    print(f"L_inf error: {Linf_error:.6e}")
    print(f"Relative L2 error: {L2_error/np.linalg.norm(u):.6e}")
    
    # Energy comparison
    total_energy_u = np.sum(energy_u) * h
    total_energy_homog = np.sum(energy_homog) * h
    
    print(f"\nTotal energy:")
    print(f"Variable p(x): {total_energy_u:.6f}")
    print(f"Homogenized: {total_energy_homog:.6f}")
    print(f"Relative difference: {abs(total_energy_u-total_energy_homog)/total_energy_u:.6e}")
    
    # Effective properties
    print(f"\nEffective properties analysis:")
    print(f"The rapid oscillations suggest homogenization theory applies.")
    print(f"Effective exponent p_eff approx {p_avg:.3f}")
    print(f"For high-frequency oscillations, solution approaches homogenized limit.")
    
    return u, u_homog, x

# Run example
u_osc, u_homog, x_osc = example_oscillating_exponent()