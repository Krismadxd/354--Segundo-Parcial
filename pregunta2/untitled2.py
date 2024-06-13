import numpy as np

# Función de activación escalón
def escalon(x, derivada=False):
    """
    Función de activación escalón.

    Args:
    x (ndarray): Entrada a la función de activación.
    derivada (bool): Si es True, calcula la derivada de la función escalón. Por defecto False.

    Returns:
    ndarray: Salida de la función de activación (o su derivada si se especifica).
    """
    if derivada:
        return np.where(x <= 0, 0, 1)  # Derivada de la función escalón
    else:
        return np.where(x <= 0, 0, 1)  # Función escalón

# Inicialización de los pesos de manera aleatoria
def inicializar_pesos(dim_entrada, dim_oculta, dim_salida):
    """
    Inicializa los pesos de la red neuronal de manera aleatoria.

    Args:
    dim_entrada (int): Número de características de entrada.
    dim_oculta (int): Número de neuronas en la capa oculta.
    dim_salida (int): Número de neuronas en la capa de salida.

    Returns:
    pesos_oculta (ndarray): Matriz de pesos para la capa oculta.
    pesos_salida (ndarray): Matriz de pesos para la capa de salida.
    """
    np.random.seed(1)
    pesos_oculta = np.random.randn(dim_entrada, dim_oculta)
    pesos_salida = np.random.randn(dim_oculta, dim_salida)
    return pesos_oculta, pesos_salida

# Propagación hacia adelante
def forward_propagation(X, pesos_oculta, pesos_salida):
    """
    Realiza la propagación hacia adelante en la red neuronal.

    Args:
    X (ndarray): Datos de entrada.
    pesos_oculta (ndarray): Pesos de la capa oculta.
    pesos_salida (ndarray): Pesos de la capa de salida.

    Returns:
    capa_oculta_salida (ndarray): Salida de la capa oculta después de la activación.
    capa_salida (ndarray): Salida final de la red neuronal después de la activación.
    """
    # Capa oculta: activación y salida
    capa_oculta_activacion = np.dot(X, pesos_oculta)
    capa_oculta_salida = escalon(capa_oculta_activacion)
    
    # Capa de salida: activación y salida
    capa_salida_activacion = np.dot(capa_oculta_salida, pesos_salida)
    capa_salida = escalon(capa_salida_activacion)
    
    return capa_oculta_salida, capa_salida

# Propagación hacia atrás y actualización de pesos
def back_propagation(X, y, capa_oculta_salida, capa_salida, pesos_oculta, pesos_salida, tasa_aprendizaje):
    """
    Realiza la propagación hacia atrás (backpropagation) y actualiza los pesos de la red neuronal.

    Args:
    X (ndarray): Datos de entrada.
    y (ndarray): Etiquetas verdaderas.
    capa_oculta_salida (ndarray): Salida de la capa oculta después de la activación.
    capa_salida (ndarray): Salida final de la red neuronal después de la activación.
    pesos_oculta (ndarray): Pesos de la capa oculta.
    pesos_salida (ndarray): Pesos de la capa de salida.
    tasa_aprendizaje (float): Tasa de aprendizaje para controlar la magnitud de la actualización de pesos.

    Returns:
    pesos_oculta (ndarray): Pesos de la capa oculta actualizados.
    pesos_salida (ndarray): Pesos de la capa de salida actualizados.
    """
    # Error en la capa de salida
    error_salida = y - capa_salida
    
    # Cálculo de delta en la capa de salida y ajuste de pesos de la capa de salida
    delta_salida = error_salida * escalon(capa_salida, derivada=True)
    pesos_salida += capa_oculta_salida.T.dot(delta_salida) * tasa_aprendizaje
    
    # Error en la capa oculta
    error_oculta = delta_salida.dot(pesos_salida.T)
    
    # Cálculo de delta en la capa oculta y ajuste de pesos de la capa oculta
    delta_oculta = error_oculta * escalon(capa_oculta_salida, derivada=True)
    pesos_oculta += X.T.dot(delta_oculta) * tasa_aprendizaje
    
    return pesos_oculta, pesos_salida

# Función principal para entrenar la red neuronal
def entrenar_red_neuronal(X, y, dim_oculta, tasa_aprendizaje, epochs):
    """
    Entrena una red neuronal de tres capas con función de activación escalón.

    Args:
    X (ndarray): Datos de entrada.
    y (ndarray): Etiquetas verdaderas.
    dim_oculta (int): Número de neuronas en la capa oculta.
    tasa_aprendizaje (float): Tasa de aprendizaje para el descenso de gradiente.
    epochs (int): Número de épocas (iteraciones completas sobre el conjunto de datos).

    Returns:
    pesos_oculta (ndarray): Pesos de la capa oculta entrenados.
    pesos_salida (ndarray): Pesos de la capa de salida entrenados.
    """
    # Inicialización de pesos
    pesos_oculta, pesos_salida = inicializar_pesos(X.shape[1], dim_oculta, y.shape[1])
    
    # Entrenamiento sobre las épocas
    for epoch in range(epochs):
        # Propagación hacia adelante
        capa_oculta_salida, capa_salida = forward_propagation(X, pesos_oculta, pesos_salida)
        
        # Propagación hacia atrás y actualización de pesos
        pesos_oculta, pesos_salida = back_propagation(X, y, capa_oculta_salida, capa_salida, pesos_oculta, pesos_salida, tasa_aprendizaje)
        
        # Impresión del error cada 1000 épocas
        if epoch % 1000 == 0:
            error = np.mean(np.abs(y - capa_salida))
            print(f"Epoch {epoch}: Error = {error}")
    
    return pesos_oculta, pesos_salida

# Datos de entrenamiento (ejemplo con datos aleatorios)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Definir hiperparámetros
dim_oculta = 3
tasa_aprendizaje = 0.2
epochs = 10000

# Entrenar la red neuronal
pesos_oculta, pesos_salida = entrenar_red_neuronal(X, y, dim_oculta, tasa_aprendizaje, epochs)

# Prueba con nuevos datos (propagación hacia adelante)
capa_oculta_salida, prediccion = forward_propagation(X, pesos_oculta, pesos_salida)
print("Predicciones finales:")
print(prediccion)
