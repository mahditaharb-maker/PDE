import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata

class pLaplace2DFEM:
    """2D FEM solver for p(x)-Laplace equation on unit square."""

    def __init__(self, p_func, f_func, boundary_condition='dirichlet'):
        self.p_func = p_func
        self.f_func = f_func
        self.bc_type = boundary_condition

    def create_mesh(self, num_points=50):
        """Create triangular mesh for unit square."""
        # Simple structured mesh for unit square
        n = int(np.sqrt(num_points))
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y)

        vertices = np.column_stack([X.ravel(), Y.ravel()])

        # Create triangle connectivity
        triangles = []
        for i in range(n-1):
            for j in range(n-1):
                # Two triangles per cell
                v0 = i * n + j
                v1 = i * n + j + 1
                v2 = (i + 1) * n + j
                v3 = (i + 1) * n + j + 1

                triangles.append([v0, v1, v2])  # Lower triangle
                triangles.append([v1, v3, v2])  # Upper triangle

        triangles = np.array(triangles)
        return vertices, triangles

    def solve(self, vertices, triangles, max_iter=30, tol=1e-6):
        """Solve using FEM with fixed-point iteration."""
        n_vertices = len(vertices)

        # Identify boundary vertices
        boundary_vertices = []
        for i, (x, y) in enumerate(vertices):
            if abs(x) < 1e-6 or abs(x - 1) < 1e-6 or abs(y) < 1e-6 or abs(y - 1) < 1e-6:
                boundary_vertices.append(i)

        # Initial guess
        u = np.zeros(n_vertices)

        for iteration in range(max_iter):
            # Assemble stiffness matrix and load vector
            K = lil_matrix((n_vertices, n_vertices))
            F = np.zeros(n_vertices)

            # Element assembly
            for tri in triangles:
                v0, v1, v2 = vertices[tri]

                # Triangle area
                area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))

                # Gradient of basis functions (constant per triangle)
                grad_phi = np.array([
                    [v1[1] - v2[1], v2[0] - v1[0]],
                    [v2[1] - v0[1], v0[0] - v2[0]],
                    [v0[1] - v1[1], v1[0] - v0[0]]
                ]) / (2 * area)

                # Current solution in triangle
                u_tri = u[tri]
                grad_u = grad_phi.T.dot(u_tri)

                # Centroid for p(x,y) evaluation
                centroid = np.mean(vertices[tri], axis=0)
                p_val = self.p_func(centroid[0], centroid[1])

                # Nonlinear coefficient
                grad_norm = np.linalg.norm(grad_u)
                eps = 1e-10
                coeff = (grad_norm**2 + eps) ** ((p_val - 2) / 2)

                # Element stiffness matrix
                Ke = area * coeff * (grad_phi.dot(grad_phi.T))

                # Assemble
                for i, idx_i in enumerate(tri):
                    for j, idx_j in enumerate(tri):
                        K[idx_i, idx_j] += Ke[i, j]

                    # Load vector (f at centroid)
                    F[idx_i] += area / 3 * self.f_func(centroid[0], centroid[1])

            # Apply boundary conditions
            K = K.tocsr()
            for idx in boundary_vertices:
                K[idx, :] = 0
                K[idx, idx] = 1
                F[idx] = 0

            # Solve linear system
            u_new = spsolve(K, F)

            # Check convergence
            residual = np.linalg.norm(u_new - u) / np.linalg.norm(u_new)
            u = u_new

            if residual < tol:
                print(f"Converged in {iteration+1} iterations")
                break

        return u

    def postprocess(self, vertices, triangles, solution):
        """Post-process solution."""
        # Create interpolation grid
        xi = yi = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate solution
        points = vertices
        values = solution
        Z = griddata(points, values, (X, Y), method='cubic')

        # Compute energy density
        grad_x, grad_y = np.gradient(Z, xi, yi)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Evaluate p(x,y) on grid
        P = self.p_func(X, Y)
        energy = grad_mag**P

        return X, Y, Z, energy

# Example usage
def p_2d(x, y):
    return 2.0 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

def f_2d(x, y):
    return 10 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Create solver
solver = pLaplace2DFEM(p_2d, f_2d)

# Create mesh
vertices, triangles = solver.create_mesh(num_points=400)

# Solve
solution = solver.solve(vertices, triangles)

# Post-process
X, Y, Z, energy = solver.postprocess(vertices, triangles, solution)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot mesh
ax = axes[0, 0]
ax.triplot(vertices[:, 0], vertices[:, 1], triangles, 'k-', alpha=0.5)
ax.set_aspect('equal')
ax.set_title('Finite element mesh')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot solution
ax = axes[0, 1]
contour = ax.contourf(X, Y, Z, 20, cmap='viridis')
plt.colorbar(contour, ax=ax)
ax.set_aspect('equal')
ax.set_title('Solution u(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot p(x,y)
ax = axes[1, 0]
P = p_2d(X, Y)
contour = ax.contourf(X, Y, P, 20, cmap='hot')
plt.colorbar(contour, ax=ax)
ax.set_aspect('equal')
ax.set_title('Variable exponent p(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Plot energy density
ax = axes[1, 1]
contour = ax.contourf(X, Y, energy, 20, cmap='plasma')
plt.colorbar(contour, ax=ax)
ax.set_aspect('equal')
ax.set_title('Energy density |grad u|^{p(x,y)}')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
plt.show()