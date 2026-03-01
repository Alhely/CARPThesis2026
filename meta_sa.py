import math
import copy
import random
import carp_core as carp  # Importamos tu motor principal

def optimizar(solucion_inicial, data, matriz_distancias, t_inicial, alfa, iter_por_t, t_final, operador):
    """
    Ejecuta el Algoritmo de Recocido Simulado (Simulated Annealing).
    Registra exactamente qué hizo el operador seleccionado.
    Retorna: mejor_solucion, mejor_costo, historial_costos, estadisticas_transparencia
    """
    # 1. Inicialización usando las funciones rápidas del core
    solucion_actual = copy.deepcopy(solucion_inicial)
    costo_actual = carp.calcular_costo_rapido(solucion_actual, data, matriz_distancias)
    
    mejor_solucion = copy.deepcopy(solucion_actual)
    mejor_costo = costo_actual
    
    T = t_inicial
    historial_costos = []
    
    # Diccionario de Transparencia
    stats = {
        "operador_usado": operador,
        "iteraciones_totales": 0,
        "vecinos_factibles": 0,
        "movimientos_aceptados": 0,
        "mejoras_globales": 0
    }
    
    # 2. Bucle de Enfriamiento
    while T > t_final:
        for _ in range(iter_por_t):
            stats["iteraciones_totales"] += 1
            
            # Llamamos al generador de vecinos del core
            vecino, exito = carp.generar_vecino_rapido(solucion_actual, data, matriz_distancias, operador=operador)
            if not exito: 
                continue # El operador generó algo infactible y se descartó
                
            stats["vecinos_factibles"] += 1
            costo_vecino = carp.calcular_costo_rapido(vecino, data, matriz_distancias)
            delta = costo_vecino - costo_actual
            
            # 3. Criterio de Aceptación (Metrópolis)
            # Si mejora (delta < 0) o si la probabilidad de temperatura lo permite
            if delta < 0 or random.random() < math.exp(-delta / T):
                solucion_actual = copy.deepcopy(vecino)
                costo_actual = costo_vecino
                stats["movimientos_aceptados"] += 1
                
                # Actualizar el óptimo global
                if costo_actual < mejor_costo:
                    mejor_solucion = copy.deepcopy(solucion_actual)
                    mejor_costo = costo_actual
                    stats["mejoras_globales"] += 1
                    
        historial_costos.append(mejor_costo)
        T *= alfa # Reducir temperatura
        
    return mejor_solucion, mejor_costo, historial_costos, stats