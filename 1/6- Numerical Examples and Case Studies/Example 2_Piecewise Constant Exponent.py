import numpy as np
import matplotlib.pyplot as plt

def example_piecewise_exponent():
    """Example with piecewise constant exponent."""
    
    def p_piecewise(x):
        return np.where(x < 0.5, 1.5, 2.5)
    
    def f_piecewise(x):
        return 1.0  # Constant source
    
    # Create grid
    N = 400
    x = np.linspace(0, 1, N+1)
    h = 1/N
    
    # Solve using iterative method
    u = np.zeros(N+1)
    
    # Iterative solver
    max_iter = 1000
    tol = 1e-8
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Solve tridiagonal system
        a = np.zeros(N)  # Lower diagonal
        b = np.zeros(N+1)  # Main diagonal
        c = np.zeros(N)  # Upper diagonal
        rhs = np.zeros(N+1)
        
        # Evaluate p at midpoints
        p_mid = (p_piecewise(x[:-1]) + p_piecewise(x[1:])) / 2
        
        for i in range(1, N):
            # Gradients at interfaces
            du_left = (u[i] - u[i-1]) / h
            du_right = (u[i+1] - u[i]) / h
            
            # Nonlinear coefficients
            coeff_left = (np.abs(du_left) + 1e-12)**(p_mid[i-1] - 2)
            coeff_right = (np.abs(du_right) + 1e-12)**(p_mid[i] - 2)
            
            a[i-1] = -coeff_left / h**2
            b[i] = (coeff_left + coeff_right) / h**2
            c[i] = -coeff_right / h**2
            rhs[i] = f_piecewise(x[i])
        
        # Boundary conditions
        b[0] = 1
        rhs[0] = 0
        b[N] = 1
        rhs[N] = 0
        
        # Solve tridiagonal system (Thomas algorithm)
        w = np.zeros(N)
        g = np.zeros(N+1)
        
        w[0] = c[0] / b[0]
        g[0] = rhs[0] / b[0]
        
        for i in range(1, N):
            denom = b[i] - a[i-1] * w[i-1]
            w[i] = c[i] / denom if i < N else 0
            g[i] = (rhs[i] - a[i-1] * g[i-1]) / denom
        
        g[N] = (rhs[N] - a[N-1] * g[N-1]) / (b[N] - a[N-1] * w[N-1])
        
        # Back substitution
        u[N] = g[N]
        for i in range(N-1, -1, -1):
            u[i] = g[i] - w[i] * u[i+1]
        
        # Check convergence
        if np.linalg.norm(u - u_old) / np.linalg.norm(u) < tol:
            print(f"Converged in {iteration+1} iterations")
            break
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot p(x)
    axes[0, 0].plot(x, p_piecewise(x), 'b-', linewidth=2)
    axes[0, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('p(x)')
    axes[0, 0].set_title('Piecewise constant exponent')
    axes[0, 0].grid(True)
    axes[0, 0].text(0.25, 1.6, 'p=1.5', ha='center', fontsize=12)
    axes[0, 0].text(0.75, 2.6, 'p=2.5', ha='center', fontsize=12)
    
    # Plot solution
    axes[0, 1].plot(x, u, 'b-', linewidth=2)
    axes[0, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u(x)')
    axes[0, 1].set_title('Solution u(x)')
    axes[0, 1].grid(True)
    
    # Plot gradient
    grad_u = np.gradient(u, x)
    axes[1, 0].plot(x, grad_u, 'b-', linewidth=2)
    axes[1, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('du/dx')
    axes[1, 0].set_title('Gradient du/dx')
    axes[1, 0].grid(True)
    
    # Plot energy density
    energy = np.abs(grad_u)**p_piecewise(x) / p_piecewise(x)
    axes[1, 1].plot(x, energy, 'b-', linewidth=2)
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Energy density')
    axes[1, 1].set_title('Energy density |grad u|^{p(x)}/p(x)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis at interface
    print("=" * 60)
    print("INTERFACE ANALYSIS: Piecewise Constant Exponent")
    print("=" * 60)
    
    # Find index closest to interface
    idx_interface = np.argmin(np.abs(x - 0.5))
    
    print(f"\nAt interface x = 0.5:")
    print(f"Left limit p(0.5-) = 1.5")
    print(f"Right limit p(0.5+) = 2.5")
    print(f"\nSolution values:")
    print(f"u(0.5) = {u[idx_interface]:.6f}")
    print(f"u'(0.5-) approx {grad_u[idx_interface-1]:.6f}")
    print(f"u'(0.5+) approx {grad_u[idx_interface+1]:.6f}")
    
    # Flux continuity check
    flux_left = np.abs(grad_u[idx_interface-1])**(1.5-2) * grad_u[idx_interface-1]
    flux_right = np.abs(grad_u[idx_interface+1])**(2.5-2) * grad_u[idx_interface+1]
    
    print(f"\nFlux continuity check:")
    print(f"Flux at x=0.5-: {flux_left:.6f}")
    print(f"Flux at x=0.5+: {flux_right:.6f}")
    print(f"Difference: {abs(flux_left - flux_right):.6e}")
    print(f"Continuous? {abs(flux_left - flux_right) < 1e-5}")
    
    # Energy in each region
    mask_left = x < 0.5
    mask_right = x >= 0.5
    
    energy_left = np.sum(energy[mask_left]) * h
    energy_right = np.sum(energy[mask_right]) * h
    
    print(f"\nEnergy distribution:")
    print(f"Energy in [0, 0.5): {energy_left:.6f} ({100*energy_left/(energy_left+energy_right):.1f}%)")
    print(f"Energy in [0.5, 1]: {energy_right:.6f} ({100*energy_right/(energy_left+energy_right):.1f}%)")
    
    return u, x

# Run example
u_piecewise, x_piecewise = example_piecewise_exponent()