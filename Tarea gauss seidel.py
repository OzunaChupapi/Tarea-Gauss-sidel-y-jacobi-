import numpy as np

# Definir la matriz A y el vector b
A = np.array([[8, 4, -1],
              [-2, 5, 1],
              [2, -1, 6]], dtype=float)

b = np.array([11, 4, 7], dtype=float)


# Función para la descomposición LU
def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        # U es la parte superior
        U[i, i:] = A[i, i:] - np.dot(L[i, :i], U[:i, i:])

        # L es la parte inferior
        if i < n - 1:
            L[i + 1:, i] = (A[i + 1:, i] - np.dot(L[i + 1:, :i], U[:i, i])) / U[i, i]

    np.fill_diagonal(L, 1)  # L tiene 1s en la diagonal
    return L, U


# Descomposición LU
L, U = lu_decomposition(A)

# Resolver Ly = b
y = np.linalg.solve(L, b)

# Resolver Ux = y
x = np.linalg.solve(U, y)

# Resultados
print("Matriz L:")
print(L)
print("\nMatriz U:")
print(U)
print("\nSolución del sistema:")
print(x)