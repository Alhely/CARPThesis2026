import copy
import random
from collections import deque
import carp_core as carp

def optimizar(solucion_inicial, costo_inicial, data, grafo, distancias, 
              tenencia_tabu=15, max_iteraciones=100, tam_vecindario=20, 
              operador="mixto", p_inter=0.5, max_intentos_vecino=50):
    """
    Ejecuta el algoritmo de metaheurística Búsqueda Tabú (Tabu Search).
    """
    # 1. Inicialización
    mejor_solucion_global = copy.deepcopy(solucion_inicial)
    mejor_costo_global = costo_inicial
    
    solucion_actual = copy.deepcopy(solucion_inicial)
    costo_actual = costo_inicial
    
    # La Lista Tabú será una cola de tamaño fijo (FIFO). 
    # Guardamos el string del arreglo para detectar ciclos exactos.
    lista_tabu = deque(maxlen=tenencia_tabu)
    lista_tabu.append(str(solucion_actual))
    
    historial = [mejor_costo_global]
    stats = {
        'iteraciones_totales': 0,
        'mejoras_globales': 0,
        'movimientos_tabu_aspirados': 0,
        'candidatos_evaluados': 0
    }
    
    operadores_disponibles = ["swap", "insertion", "inversion"]
    
    # 2. Bucle Principal
    for iteracion in range(max_iteraciones):
        candidatos = []
        
        # 2.1 Generar el vecindario (Muestreo aleatorio)
        for _ in range(tam_vecindario):
            op_usar = random.choice(operadores_disponibles) if operador == "mixto" else operador
            
            # Usamos el motor base para generar un vecino
            vecino, c_vec, _ = carp.aplicar_y_evaluar_vecindario(
                solucion_actual, data, grafo, distancias, 
                operador=op_usar, p_inter=p_inter, max_intentos=max_intentos_vecino
            )
            
            # Si el motor logró generar un vecino diferente a la base, lo consideramos
            if vecino != solucion_actual:
                candidatos.append((vecino, c_vec))
                stats['candidatos_evaluados'] += 1
        
        if not candidatos:
            # Si es imposible generar vecinos válidos (restricciones muy duras), abortamos
            break
            
        # 2.2 Ordenar candidatos de mejor (menor costo) a peor
        candidatos.sort(key=lambda x: x[1])
        
        mejor_candidato = None
        mejor_costo_candidato = float('inf')
        
        # 2.3 Evaluar restricciones Tabú y Criterios de Aspiración
        for cand_sol, cand_costo in candidatos:
            sol_str = str(cand_sol)
            es_tabu = sol_str in lista_tabu
            
            # Criterio de Aspiración: Está prohibido, pero es el nuevo récord global
            if es_tabu and cand_costo < mejor_costo_global:
                mejor_candidato = cand_sol
                mejor_costo_candidato = cand_costo
                stats['movimientos_tabu_aspirados'] += 1
                break # Encontramos el santo grial, paramos de buscar
            
            # Movimiento válido normal
            if not es_tabu:
                mejor_candidato = cand_sol
                mejor_costo_candidato = cand_costo
                break
        
        # Si por alguna razón TODOS los candidatos eran tabú y no aspiraron, 
        # tomamos el "menos malo" para forzar al algoritmo a moverse.
        if mejor_candidato is None:
            mejor_candidato, mejor_costo_candidato = candidatos[0]
        
        # 3. Ejecutar el Salto (Actualizar estados)
        solucion_actual = mejor_candidato
        costo_actual = mejor_costo_candidato
        
        # Actualizamos la memoria Tabú
        lista_tabu.append(str(solucion_actual))
        
        # 4. Actualizar el Óptimo Global Histórico
        if costo_actual < mejor_costo_global:
            mejor_solucion_global = copy.deepcopy(solucion_actual)
            mejor_costo_global = costo_actual
            stats['mejoras_globales'] += 1
            
        historial.append(mejor_costo_global)
        stats['iteraciones_totales'] += 1

    return mejor_solucion_global, mejor_costo_global, historial, stats