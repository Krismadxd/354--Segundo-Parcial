import numpy as np

# Función para generar vecinos utilizando el operador 2-opt
def generar_vecinos_2opt(perm):
    """
    Genera vecinos de una permutación dada utilizando el operador 2-opt.

    Args:
    perm (ndarray): Permutación de ciudades.

    Returns:
    ndarray: Matriz de vecinos generados.
    """
    n = perm.shape[0]
    n_vecinos = n * (n - 1) // 2 - n  # Número de vecinos posibles
    vecinos = np.zeros((n_vecinos, n), dtype=int)  # Matriz para almacenar vecinos
    ind = 0
    
    # Generar vecinos aplicando el operador 2-opt
    for i in range(n - 1):
        for j in range(i + 2, n):
            if not (i == 0 and j == n - 1):
                vecinos[ind, :] = perm
                segmento_reverso = perm[i:j + 1].copy()
                vecinos[ind, i:j + 1] = segmento_reverso[::-1]  # Aplicar la inversión 2-opt
                ind += 1
    
    return vecinos

# Ejemplo de uso
distancias = np.array([
    [0, 1, 2, 3, 4],
    [1, 0, 5, 6, 7],
    [2, 5, 0, 8, 9],
    [3, 6, 8, 0, 10],
    [4, 7, 9, 10, 0]
])

permutacion_inicial = np.random.permutation(distancias.shape[0])
print(f"Permutación inicial: {permutacion_inicial}")

vecinos = generar_vecinos_2opt(permutacion_inicial)

# Imprimir vecinos de manera diferente
print("Vecinos generados:")
for idx, vecino in enumerate(vecinos):
    distancia_total = sum([distancias[vecino[i], vecino[(i + 1) % vecino.shape[0]]] for i in range(vecino.shape[0])])
    print(f" {idx + 1}: {vecino}  \n Distancia total: {distancia_total}")
