import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def example_boundary_layer():
    """Example showing boundary layer formation."""
    
    def p_boundary(x, epsilon):
        # p(x) approaches 1 near boundaries
        return 1 + epsilon + (2 - 2*epsilon) * (x - 0.5)**2
    
    def f_boundary(x):
        return 1.0  # Constant source
    
    # Solve for different epsilon values
    epsilon_vals = [0.5, 0.1, 0.05, 0.01]
    solutions = []
    
    for eps in epsilon_vals:
        def p_current(x):
            return p_boundary(x, eps)
        
        # Solve using finite differences
        N = 500
        x = np.linspace(0, 1, N+1)
        h = 1/N
        
        # Initial guess
        u = 4 * x * (1 - x)
        
        # Iterative solver
        max_iter = 1000
        tol = 1e-10
        
        for iteration in range(max_iter):
            u_old = u.copy()
            
            # Evaluate p at midpoints
            p_mid = (p_current(x[:-1]) + p_current(x[1:])) / 2
            
            # Build and solve linearized system
            A = np.zeros((N+1, N+1))
            rhs = np.zeros(N+1)
            
            for i in range(1, N):
                du_left = (u[i] - u[i-1]) / h
                du_right = (u[i+1] - u[i]) / h
                
                # Linearized coefficients
                coeff_left = (np.abs(du_left) + 1e-12)**(p_mid[i-1] - 2)
                coeff_right = (np.abs(du_right) + 1e-12)**(p_mid[i] - 2)
                
                A[i, i-1] = -coeff_left / h**2
                A[i, i] = (coeff_left + coeff_right) / h**2
                A[i, i+1] = -coeff_right / h**2
                rhs[i] = f_boundary(x[i])
            
            # Boundary conditions
            A[0, 0] = 1
            rhs[0] = 0
            A[N, N] = 1
            rhs[N] = 0
            
            # Solve
            u = np.linalg.solve(A, rhs)
            
            if np.linalg.norm(u - u_old) / np.linalg.norm(u) < tol:
                if eps == epsilon_vals[0]:
                    print(f"epsilon = {eps}: converged in {iteration+1} iterations")
                break
        
        solutions.append((eps, x.copy(), u.copy()))
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot p(x) for different epsilon
    x_plot = np.linspace(0, 1, 1000)
    for eps, _, _ in solutions:
        p_vals = p_boundary(x_plot, eps)
        axes[0, 0].plot(x_plot, p_vals, label=f'eps = {eps}')
    
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('p(x)')
    axes[0, 0].set_title('Exponent p(x) = 1 + eps + (2-2*eps)*(x-0.5)^2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot solutions
    for eps, x_vals, u_vals in solutions:
        axes[0, 1].plot(x_vals, u_vals, label=f'eps = {eps}')
    
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u(x)')
    axes[0, 1].set_title('Solutions for different eps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot gradients
    for eps, x_vals, u_vals in solutions:
        grad_u = np.gradient(u_vals, x_vals)
        axes[0, 2].plot(x_vals, grad_u, label=f'eps = {eps}')
    
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('du/dx')
    axes[0, 2].set_title('Gradients')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Zoom near boundary x=0
    for eps, x_vals, u_vals in solutions:
        mask = x_vals <= 0.1
        axes[1, 0].plot(x_vals[mask], u_vals[mask], label=f'eps = {eps}')
    
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u(x)')
    axes[1, 0].set_title('Zoom near x=0 (boundary layer)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot boundary layer thickness
    boundary_thickness = []
    eps_vals_plot = []
    
    for eps, x_vals, u_vals in solutions:
        # Estimate boundary layer thickness
        grad_u = np.gradient(u_vals, x_vals)
        max_grad = np.max(np.abs(grad_u))
        
        # Find where gradient drops to 10% of maximum
        threshold = 0.1 * max_grad
        idx = np.where(np.abs(grad_u) > threshold)[0][0]
        thickness = x_vals[idx]
        
        boundary_thickness.append(thickness)
        eps_vals_plot.append(eps)
    
    axes[1, 1].loglog(eps_vals_plot, boundary_thickness, 'bo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('epsilon (log scale)')
    axes[1, 1].set_ylabel('Boundary layer thickness (log scale)')
    axes[1, 1].set_title('Boundary layer thickness vs epsilon')
    axes[1, 1].grid(True, alpha=0.3, which='both')
    
    # Plot energy near boundary
    for eps, x_vals, u_vals in solutions:
        p_vals = p_boundary(x_vals, eps)
        grad_u = np.gradient(u_vals, x_vals)
        energy = np.abs(grad_u)**p_vals / p_vals
        
        mask = x_vals <= 0.2
        axes[1, 2].plot(x_vals[mask], energy[mask], label=f'eps = {eps}')
    
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('Energy density')
    axes[1, 2].set_title('Energy density near boundary')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("=" * 60)
    print("BOUNDARY LAYER ANALYSIS")
    print("=" * 60)
    
    print(f"\nAs epsilon -> 0+, p(x) -> 1 near boundaries:")
    print("This creates singular behavior and boundary layers.")
    
    print(f"\nBoundary layer thickness estimates:")
    for eps, thickness in zip(eps_vals_plot, boundary_thickness):
        print(f"epsilon = {eps:.4f}: thickness approx {thickness:.6f}")
    
    # Theoretical prediction: boundary layer thickness ~ epsilon^alpha
    def power_law(x, a, b):
        return a * x**b
    
    if len(eps_vals_plot) >= 3:
        popt, pcov = curve_fit(power_law, eps_vals_plot, boundary_thickness)
        print(f"\nPower law fit: thickness approx {popt[0]:.3f} * epsilon^{popt[1]:.3f}")
        print("For p -> 1, theory predicts boundary layers of width O(epsilon).")
    
    # Maximum gradient analysis
    print(f"\nMaximum gradient analysis:")
    for eps, x_vals, u_vals in solutions:
        grad_u = np.gradient(u_vals, x_vals)
        max_grad = np.max(np.abs(grad_u))
        print(f"epsilon = {eps:.4f}: max|du/dx| = {max_grad:.6f}")
    
    print(f"\nObservation: As epsilon decreases (p -> 1),")
    print("1. Boundary layer becomes thinner")
    print("2. Gradient near boundary becomes larger")
    print("3. Solution develops sharper transitions")
    print("4. This illustrates the singular limit p -> 1")
    
    return solutions

# Run example
boundary_solutions = example_boundary_layer()