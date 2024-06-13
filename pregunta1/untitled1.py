import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Funciones auxiliares para la red neuronal

def inicializar_pesos(dim_entrada, dim_oculta, dim_salida):
    """
    Inicializa los pesos de la red neuronal de manera aleatoria.

    Args:
    dim_entrada (int): Número de características de entrada.
    dim_oculta (int): Número de neuronas en la capa oculta.
    dim_salida (int): Número de neuronas en la capa de salida.

    Returns:
    pesos_ocultos (ndarray): Matriz de pesos para la capa oculta.
    pesos_salida (ndarray): Matriz de pesos para la capa de salida.
    """
    np.random.seed(1)
    pesos_ocultos = np.random.randn(dim_entrada, dim_oculta) * 0.1
    pesos_salida = np.random.randn(dim_oculta, dim_salida) * 0.1
    return pesos_ocultos, pesos_salida

def sigmoid(x):
    """
    Función de activación sigmoidal.

    Args:
    x (ndarray): Vector de entrada.

    Returns:
    ndarray: Salida después de aplicar la función sigmoidal.
    """
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """
    Función softmax para la capa de salida.

    Args:
    x (ndarray): Vector de entrada.

    Returns:
    ndarray: Salida después de aplicar la función softmax.
    """
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def calcular_gradiente_oculta(X, delta_salida, pesos_salida):
    """
    Calcula el gradiente para la capa oculta utilizando backpropagation.

    Args:
    X (ndarray): Salida de la capa oculta.
    delta_salida (ndarray): Error en la capa de salida.
    pesos_salida (ndarray): Pesos de la capa de salida.

    Returns:
    ndarray: Gradiente para la capa oculta.
    """
    return np.dot(delta_salida, pesos_salida.T) * X * (1 - X)

def calcular_gradiente_salida(delta_salida, salida_capa_oculta):
    """
    Calcula el gradiente para la capa de salida utilizando backpropagation.

    Args:
    delta_salida (ndarray): Error en la capa de salida.
    salida_capa_oculta (ndarray): Salida de la capa oculta.

    Returns:
    ndarray: Gradiente para la capa de salida.
    """
    return np.dot(salida_capa_oculta.T, delta_salida)

def actualizar_pesos(pesos, gradiente, tasa_aprendizaje):
    """
    Actualiza los pesos de la red neuronal utilizando el descenso de gradiente.

    Args:
    pesos (ndarray): Pesos actuales de la red neuronal.
    gradiente (ndarray): Gradiente calculado para los pesos.
    tasa_aprendizaje (float): Tasa de aprendizaje para controlar la magnitud de la actualización.

    Returns:
    ndarray: Pesos actualizados.
    """
    return pesos - tasa_aprendizaje * gradiente

def calcular_perdida(y_true, y_pred):
    """
    Calcula la pérdida utilizando la entropía cruzada como medida de error.

    Args:
    y_true (ndarray): Etiquetas verdaderas.
    y_pred (ndarray): Predicciones del modelo.

    Returns:
    float: Pérdida calculada.
    """
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

# Función principal para entrenar la red neuronal

def entrenar_red_neuronal(X, y, dim_oculta, tasa_aprendizaje, epochs):
    """
    Entrena una red neuronal con una capa oculta para el dataset de iris.

    Args:
    X (ndarray): Características de entrada.
    y (ndarray): Etiquetas de salida (en formato one-hot).
    dim_oculta (int): Número de neuronas en la capa oculta.
    tasa_aprendizaje (float): Tasa de aprendizaje para el descenso de gradiente.
    epochs (int): Número de épocas (iteraciones completas sobre el conjunto de datos).

    Returns:
    pesos_ocultos (ndarray): Pesos aprendidos para la capa oculta.
    pesos_salida (ndarray): Pesos aprendidos para la capa de salida.
    """
    dim_entrada = X.shape[1]  # Número de características de entrada
    dim_salida = y.shape[1]   # Número de clases de salida
    
    # Inicialización de pesos aleatorios
    pesos_ocultos, pesos_salida = inicializar_pesos(dim_entrada, dim_oculta, dim_salida)
    
    # Ciclo de entrenamiento sobre las épocas
    for epoch in range(epochs):
        # Propagación hacia adelante (forward propagation)
        salida_capa_oculta = sigmoid(np.dot(X, pesos_ocultos))
        salida_capa_salida = softmax(np.dot(salida_capa_oculta, pesos_salida))
        
        # Cálculo de la pérdida
        error = calcular_perdida(y, salida_capa_salida)
        
        # Impresión de la pérdida en cada época
        print(f"Epoch {epoch + 1}/{epochs} - Pérdida: {error}")
        
        # Cálculo del gradiente utilizando backpropagation
        delta_salida = (salida_capa_salida - y) / len(X)
        gradiente_salida = calcular_gradiente_salida(delta_salida, salida_capa_oculta)
        delta_oculta = calcular_gradiente_oculta(salida_capa_oculta, delta_salida, pesos_salida)
        gradiente_oculta = np.dot(X.T, delta_oculta)
        
        # Actualización de pesos usando el descenso de gradiente
        pesos_ocultos = actualizar_pesos(pesos_ocultos, gradiente_oculta, tasa_aprendizaje)
        pesos_salida = actualizar_pesos(pesos_salida, gradiente_salida, tasa_aprendizaje)
    
    return pesos_ocultos, pesos_salida

# Cargar dataset iris
iris = load_iris()
X = iris.data
y = iris.target

# Convertir y en un formato de matriz one-hot
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Definir hiperparámetros
dim_oculta = 5
tasa_aprendizaje = 0.4
epochs = 100

# Entrenar la red neuronal
pesos_ocultos, pesos_salida = entrenar_red_neuronal(X_train, y_train, dim_oculta, tasa_aprendizaje, epochs)
