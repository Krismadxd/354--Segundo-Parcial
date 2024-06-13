# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 08:22:50 2024

@author: krismad10
"""

import numpy as np
import random

# Matriz de distancias
distancias = np.array([
    [0, 7, 9, 8, 20],
    [7, 0, 10, 4, 11],
    [9, 10, 0, 15, 5],
    [8, 4, 15, 0, 17],
    [20, 11, 5, 17, 0],
])

# Número de ciudades
num_ciudades = distancias.shape[0]

# Función para crear una ruta aleatoria
def crear_ruta():
    ruta = list(np.random.permutation(num_ciudades))
    return ruta

# Función para crear una población inicial
def crear_poblacion(tamaño_poblacion):
    poblacion = [crear_ruta() for _ in range(tamaño_poblacion)]
    return poblacion

# Función para calcular la distancia total de una ruta
def calcular_distancia(ruta):
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += distancias[ruta[i], ruta[i + 1]]
    distancia_total += distancias[ruta[-1], ruta[0]]  # Regresar al punto de origen
    return distancia_total

# Función para evaluar la aptitud de una ruta
def evaluar_aptitud(ruta):
    return 1 / calcular_distancia(ruta)

# Función para seleccionar padres (torneo)
def seleccion_torneo(poblacion, tamaño_torneo):
    torneo = random.sample(poblacion, tamaño_torneo)
    torneo.sort(key=lambda ruta: calcular_distancia(ruta))
    return torneo[0]

# Función para realizar el cruzamiento (OX)
def cruzamiento_ox(padre1, padre2):
    start, end = sorted(random.sample(range(len(padre1)), 2))
    hijo = [None] * len(padre1)
    hijo[start:end] = padre1[start:end]
    
    puntero = 0
    for gen in padre2:
        if gen not in hijo:
            while hijo[puntero] is not None:
                puntero += 1
            hijo[puntero] = gen
    
    return hijo

# Función para realizar la mutación (swap)
def mutacion_swap(ruta, tasa_mutacion):
    for i in range(len(ruta)):
        if random.random() < tasa_mutacion:
            j = random.randint(0, len(ruta) - 1)
            ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

# Función principal del algoritmo genético
def algoritmo_genetico(tamaño_poblacion, tamaño_torneo, tasa_mutacion, num_generaciones):
    poblacion = crear_poblacion(tamaño_poblacion)
    print("Población inicial:")
    for ruta in poblacion:
        print(ruta, "Distancia:", calcular_distancia(ruta))
    
    for generacion in range(num_generaciones):
        nueva_poblacion = []
        for _ in range(tamaño_poblacion):
            padre1 = seleccion_torneo(poblacion, tamaño_torneo)
            padre2 = seleccion_torneo(poblacion, tamaño_torneo)
            hijo = cruzamiento_ox(padre1, padre2)
            hijo = mutacion_swap(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion
        
        # Imprimir detalles de la generación
        print(f"\nGeneración {generacion + 1}:")
        for ruta in poblacion:
            print(ruta, "Distancia:", calcular_distancia(ruta))
    
    mejor_ruta = min(poblacion, key=lambda ruta: calcular_distancia(ruta))
    mejor_distancia = calcular_distancia(mejor_ruta)
    
    return mejor_ruta, mejor_distancia

# Parámetros del algoritmo genético
tamaño_poblacion = 10
tamaño_torneo = 3
tasa_mutacion = 0.01
num_generaciones = 5

# Ejecutar el algoritmo genético
mejor_ruta, mejor_distancia = algoritmo_genetico(tamaño_poblacion, tamaño_torneo, tasa_mutacion, num_generaciones)

# Mostrar los resultados
print("\nMejor ruta encontrada:", mejor_ruta)
print("Distancia de la mejor ruta:", mejor_distancia)

