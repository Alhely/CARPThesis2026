import copy
import random
import carp_core as carp

def vuelo_levy_discreto(solucion, data, grafo, distancias, operador, p_inter, max_intentos):
    """
    Simula un Vuelo de Lévy en un espacio discreto.
    La mayoría de las veces da 1 paso (búsqueda local), pero ocasionalmente
    da 2, 3 o 4 pasos (salto largo / exploración).
    """
    # Simulamos una distribución de cola pesada simple
    r = random.random()
    if r > 0.90: pasos = random.randint(3, 4) # Salto largo (10% prob)
    elif r > 0.70: pasos = 2                  # Salto medio (20% prob)
    else: pasos = 1                           # Paso corto  (70% prob)
        
    vecino_actual = copy.deepcopy(solucion)
    costo_actual = float('inf')
    
    for _ in range(pasos):
        op_usar = random.choice(["swap", "insertion", "inversion"]) if operador == "mixto" else operador
        v, c, _ = carp.aplicar_y_evaluar_vecindario(
            vecino_actual, data, grafo, distancias, 
            operador=op_usar, p_inter=p_inter, max_intentos=max_intentos
        )
        vecino_actual = v
        costo_actual = c
        
    return vecino_actual, costo_actual

def optimizar(solucion_inicial, costo_inicial, data, grafo, distancias, 
              num_nidos=15, p_a=0.25, max_iteraciones=100, 
              operador="mixto", p_inter=0.5, max_intentos_vecino=50):
    """
    Ejecuta el algoritmo Cuckoo Search adaptado para CARP.
    """
    mejor_solucion_global = copy.deepcopy(solucion_inicial)
    mejor_costo_global = costo_inicial
    
    # 1. Inicializar Población (Nidos)
    nidos = [(copy.deepcopy(solucion_inicial), costo_inicial)]
    
    # Rellenar el resto de los nidos con mutaciones de la inicial para tener diversidad
    for _ in range(num_nidos - 1):
        v_rand, c_rand = vuelo_levy_discreto(solucion_inicial, data, grafo, distancias, operador, p_inter, max_intentos_vecino)
        nidos.append((v_rand, c_rand))
        
    historial = [mejor_costo_global]
    stats = {
        'iteraciones_totales': 0,
        'mejoras_globales': 0,
        'huevos_cuckoo_exitosos': 0,
        'nidos_abandonados': 0
    }
    
    # 2. Bucle Principal
    for iteracion in range(max_iteraciones):
        
        # A. Un Cuco genera un nuevo huevo (solución) vía Vuelo de Lévy desde un nido aleatorio
        nido_origen, _ = random.choice(nidos)
        cuckoo_sol, cuckoo_costo = vuelo_levy_discreto(
            nido_origen, data, grafo, distancias, 
            operador, p_inter, max_intentos_vecino
        )
        
        # B. Elegir un nido al azar para intentar reemplazarlo
        idx_azar = random.randint(0, num_nidos - 1)
        if cuckoo_costo < nidos[idx_azar][1]:
            nidos[idx_azar] = (cuckoo_sol, cuckoo_costo)
            stats['huevos_cuckoo_exitosos'] += 1
            
        # C. Ordenar nidos por costo (el mejor en el índice 0)
        nidos.sort(key=lambda x: x[1])
        
        # Actualizar el mejor global si el mejor nido lo superó
        if nidos[0][1] < mejor_costo_global:
            mejor_solucion_global = copy.deepcopy(nidos[0][0])
            mejor_costo_global = nidos[0][1]
            stats['mejoras_globales'] += 1
            
        # D. Descubrimiento y Abandono (Fracción p_a de los peores nidos)
        num_abandonar = int(num_nidos * p_a)
        if num_abandonar > 0:
            stats['nidos_abandonados'] += num_abandonar
            # Reemplazar los peores (los últimos de la lista) con nuevas soluciones
            for i in range(num_nidos - num_abandonar, num_nidos):
                # Generamos una nueva semilla aleatoria (Exploración profunda)
                nueva_sol = carp.generar_solucion_inicial_aleatoria(data, distancias)
                # Evaluamos su costo
                _, n_costo, _ = carp.calcular_y_mostrar_rutas_compacto(nueva_sol, data, grafo)
                nidos[i] = (nueva_sol, n_costo)
                
        historial.append(mejor_costo_global)
        stats['iteraciones_totales'] += 1

    return mejor_solucion_global, mejor_costo_global, historial, stats