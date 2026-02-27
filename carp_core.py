import re
import os
import copy
import json
import random
import pickle
import logging
from pathlib import Path
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. LECTURA Y VALIDACIÓN
# ==========================================

def leer_carplib_dat(filepath):
    parsed_data = {}
    current_list_key = None
    contador_tarea = 1
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(('=', '-', '|')):
                continue
            
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                
                if key.startswith('LISTA_ARISTAS'):
                    current_list_key = key
                    parsed_data[key] = []
                    contador_tarea = 1
                else:
                    current_list_key = None
                    if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                        parsed_data[key] = int(val)
                    else:
                        parsed_data[key] = val
                        
            elif current_list_key:
                numeros = [int(x) for x in re.findall(r'-?\d+', line)]
                if len(numeros) >= 3:
                    parsed_data[current_list_key].append({
                        'tarea': f"T{contador_tarea}",
                        'nodos': (numeros[0], numeros[1]),
                        'costo': numeros[2],
                        'demanda': numeros[3] if len(numeros) >= 4 else 0
                    })
                    contador_tarea += 1
    return parsed_data

def validate_instance(created_dict, dat_file_path):
    n_req_esperadas = created_dict.get("ARISTAS_REQ", 0)
    n_noreq_esperadas = created_dict.get("ARISTAS_NOREQ", 0)
    lista_req = created_dict.get("LISTA_ARISTAS_REQ", [])
    lista_noreq = created_dict.get("LISTA_ARISTAS_NOREQ", [])
    
    todas_las_aristas = lista_req + lista_noreq
    nodos_encontrados = []
    for item in todas_las_aristas:
        nodos_encontrados.extend(item['nodos'])
    
    max_nodo = max(nodos_encontrados) if nodos_encontrados else 0
    total_vertices = created_dict.get("VERTICES", 0)

    lineas_utiles_archivo = 0
    with open(dat_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(('=', '-', '|')):
                continue
            lineas_utiles_archivo += 1

    elementos_en_dict = len(created_dict) + len(lista_req) + len(lista_noreq)
    is_valid = (n_req_esperadas == len(lista_req)) and \
               (n_noreq_esperadas == len(lista_noreq)) and \
               (max_nodo <= total_vertices) and \
               (lineas_utiles_archivo == elementos_en_dict)

    return {
        "is_valid": is_valid,
        "status": "VALIDACIÓN EXITOSA" if is_valid else "ERROR DE INTEGRIDAD",
        "detalles": {
            "aristas_req": f"{len(lista_req)}/{n_req_esperadas}",
            "aristas_noreq": f"{len(lista_noreq)}/{n_noreq_esperadas}",
            "nodos_ok": f"Max Nodo {max_nodo} <= {total_vertices}",
            "lineas_vs_dict": f"Archivo: {lineas_utiles_archivo} == Dict: {elementos_en_dict}"
        }
    }

# ==========================================
# 2. GESTIÓN DE EJECUCIÓN Y GUARDADO
# ==========================================

def iniciar_ejecucion(metadata_dict, base_dir="instance_runs"):
    project_name = metadata_dict.get("NOMBRE", "Instancia")
    run_id = datetime.now().strftime("%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        directorio_base = Path(__file__).resolve().parent
    except NameError:
        directorio_base = Path.cwd()
        
    run_path = directorio_base / base_dir / f"{date_str}_{project_name}_ID-{run_id}"
    run_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{project_name}_{run_id}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(run_path / "execution.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

    logger.info("Run initialized.")
    return run_path, logger

def guardar_objeto_automatico(carpeta, nombre_base, objeto):
    if hasattr(objeto, 'savefig'):
        ruta = os.path.join(carpeta, f"{nombre_base}.jpg")
        objeto.savefig(ruta, format='jpg', dpi=300, bbox_inches='tight')
        return ruta
    elif isinstance(objeto, str):
        ruta = os.path.join(carpeta, f"{nombre_base}.txt")
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write(objeto)
        return ruta
    elif isinstance(objeto, (dict, list)):
        ruta_json = os.path.join(carpeta, f"{nombre_base}.json")
        try:
            with open(ruta_json, 'w', encoding='utf-8') as f:
                json.dump(objeto, f, indent=4)
            return ruta_json
        except TypeError:
            pass 
            
    ruta_pkl = os.path.join(carpeta, f"{nombre_base}.pkl")
    with open(ruta_pkl, 'wb') as f:
        pickle.dump(objeto, f)
    return ruta_pkl

def finalizar_ejecucion(run_path, logger, success=True, reason=""):
    status_str = "SUCCESS" if success else "FAILED"
    (run_path / f"STATUS_{status_str}").touch()
    
    mensaje = f"Run finished with status: {status_str}"
    if reason: 
        mensaje += f" | Reason: {reason}"
        
    logger.info(mensaje)

# ==========================================
# 3. REDES Y DISTANCIAS
# ==========================================

def generar_grafo_limpio(data, carpeta_salida=None, figsize=(16, 12), k_layout=1.5):
    G = nx.Graph()
    for i in range(1, data.get('VERTICES', 0) + 1):
        G.add_node(i)
        
    for item in data.get('LISTA_ARISTAS_REQ', []):
        u, v = item['nodos']
        G.add_edge(u, v, cost=item['costo'], demand=item['demanda'], tipo='req')

    for item in data.get('LISTA_ARISTAS_NOREQ', []):
        u, v = item['nodos']
        G.add_edge(u, v, cost=item['costo'], demand=item.get('demanda', 0), tipo='noreq')

    pos = nx.spring_layout(G, k=k_layout, iterations=200, seed=42)
    plt.figure(figsize=figsize) 
    
    deposito = data.get('DEPOSITO', 1)
    node_colors = ['#ff4d4d' if node == deposito else '#74b9ff' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors, edgecolors='#74b9ff', linewidths=1)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    
    arcos_req = [(u, v) for u, v, d in G.edges(data=True) if d['tipo'] == 'req']
    arcos_noreq = [(u, v) for u, v, d in G.edges(data=True) if d['tipo'] == 'noreq']
    
    nx.draw_networkx_edges(G, pos, edgelist=arcos_req, width=2.0, edge_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=arcos_noreq, width=1.5, edge_color='gray', style='dashed')
    
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d['tipo'] == 'req':
            edge_labels[(u, v)] = f"C:{d['cost']}\nD:{d['demand']}"
        else:
            edge_labels[(u, v)] = f"C:{d['cost']}"
            
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='#2d3436', label_pos=0.5, bbox=dict(alpha=0))
    
    texto_resumen = (f"Nombre: {data.get('NOMBRE', 'N/A')}\nVehículos: {data.get('VEHICULOS', 0)}\nCapacidad: {data.get('CAPACIDAD', 0)}")
    plt.text(0.02, 0.98, texto_resumen, transform=plt.gca().transAxes, fontsize=16, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f2f6', alpha=0.9))

    nombre_instancia = data.get('NOMBRE', 'instancia')
    plt.title(f"Grafo de Instancia: {nombre_instancia}", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Si se proporciona la carpeta, lo guarda directamente ahí
    if carpeta_salida:
        guardar_objeto_automatico(carpeta_salida, f"{nombre_instancia}_grafo", plt.gcf())
        
    plt.close() # Cierra la figura para liberar memoria
    return G

def calcular_matriz_distancias(G, algoritmo='dijkstra'):
    algoritmo = algoritmo.strip().lower()
    nodos = list(G.nodes())
    matriz_distancias = {}
    
    if algoritmo == 'dijkstra':
        dijkstra_raw = dict(nx.all_pairs_dijkstra_path_length(G, weight='cost'))
        for u in nodos:
            matriz_distancias[u] = {}
            for v in nodos:
                matriz_distancias[u][v] = dijkstra_raw.get(u, {}).get(v, float('inf'))
    elif algoritmo in ['floyd-warshall', 'floyd_warshall', 'floyd']:
        fw_raw = nx.floyd_warshall(G, weight='cost')
        for u in nodos:
            matriz_distancias[u] = dict(fw_raw[u])
    else:
        raise ValueError("Algoritmo no reconocido.")
    return matriz_distancias

# ==========================================
# 4. SOLUCIÓN Y VECINDARIOS
# ==========================================

def es_ruta_factible(ruta, data, matriz_distancias):
    if not ruta: return True
    capacidad_max = data.get('CAPACIDAD', 0)
    deposito = data.get('DEPOSITO', 1)
    info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}
    
    demanda_total = 0
    nodo_actual = deposito
    
    for id_tarea in ruta:
        tarea = info_tareas[id_tarea]
        u, v = tarea['nodos']
        demanda_total += tarea['demanda']
        if demanda_total > capacidad_max: return False
        if matriz_distancias[nodo_actual][u] == float('inf'): return False
        nodo_actual = v
        
    if matriz_distancias[nodo_actual][deposito] == float('inf'): return False
    return True

def generar_solucion_inicial_aleatoria(data, matriz_distancias, max_intentos=10000):
    num_vehiculos = data.get('VEHICULOS', 0)
    capacidad_max = data.get('CAPACIDAD', 0)
    deposito = data.get('DEPOSITO', 1)
    tareas_requeridas = data.get('LISTA_ARISTAS_REQ', [])
    
    for t in tareas_requeridas:
        if t['demanda'] > capacidad_max: raise ValueError("Instancia inviable por capacidad.")

    intentos = 0
    while intentos < max_intentos:
        intentos += 1
        tareas_mezcladas = tareas_requeridas.copy()
        random.shuffle(tareas_mezcladas)
        
        solucion = [[] for _ in range(num_vehiculos)]
        demanda_vehiculos = [0] * num_vehiculos
        v_idx = 0
        es_factible = True
        nodos_anteriores = (deposito, deposito) 
        
        for tarea in tareas_mezcladas:
            id_tarea, demanda_tarea = tarea['tarea'], tarea['demanda']
            u_act, v_act = tarea['nodos']
            u_ant, v_ant = nodos_anteriores
            
            hay_camino = (matriz_distancias[u_ant][u_act] != float('inf') or matriz_distancias[u_ant][v_act] != float('inf') or
                          matriz_distancias[v_ant][u_act] != float('inf') or matriz_distancias[v_ant][v_act] != float('inf'))
            if not hay_camino:
                es_factible = False; break
            
            if demanda_vehiculos[v_idx] + demanda_tarea <= capacidad_max:
                solucion[v_idx].append(id_tarea)
                demanda_vehiculos[v_idx] += demanda_tarea
                nodos_anteriores = (u_act, v_act)
            else:
                v_idx += 1
                if v_idx >= num_vehiculos:
                    es_factible = False; break 
                if matriz_distancias[deposito][u_act] == float('inf') and matriz_distancias[deposito][v_act] == float('inf'):
                    es_factible = False; break
                
                solucion[v_idx].append(id_tarea)
                demanda_vehiculos[v_idx] += demanda_tarea
                nodos_anteriores = (u_act, v_act)
                
        if es_factible: return solucion
    raise RuntimeError("No se halló solución factible.")

def calcular_y_mostrar_rutas_compacto(solucion, data, G):
    deposito = data.get('DEPOSITO', 1)
    capacidad_max = data.get('CAPACIDAD', 0)
    info_tareas = {t['tarea']: {'u': t['nodos'][0], 'v': t['nodos'][1], 'costo': t['costo'], 'demanda': t['demanda']}
                   for t in data.get('LISTA_ARISTAS_REQ', [])}
        
    costos_rutas = []
    costo_total_solucion = 0
    reporte = [] # Guardaremos el reporte en una lista para retornarlo como string
    
    reporte.append("="*80)
    reporte.append("EVALUACIÓN COMPACTA DE RUTAS")
    reporte.append("="*80)
    
    for i, ruta in enumerate(solucion):
        reporte.append(f"RUTA {i + 1} {ruta}")
        if not ruta:
            reporte.append(f"  -> Vehículo vacío | Costo Total: 0 | Demanda: 0 / {capacidad_max}\n")
            costos_rutas.append(0)
            continue
            
        costo_vehiculo, demanda_vehiculo, nodo_actual = 0, 0, deposito
        for id_tarea in ruta:
            tarea = info_tareas[id_tarea]
            u, v, costo_serv, dem_serv = tarea['u'], tarea['v'], tarea['costo'], tarea['demanda']
            
            if nodo_actual != u:
                camino_dh = nx.shortest_path(G, source=nodo_actual, target=u, weight='cost')
                costo_dh = nx.shortest_path_length(G, source=nodo_actual, target=u, weight='cost')
                str_dh = " -> ".join(map(str, camino_dh))
            else:
                costo_dh, str_dh = 0, f"Ninguno (ya en {u})"
                
            costo_total_paso = costo_dh + costo_serv
            costo_vehiculo += costo_total_paso
            demanda_vehiculo += dem_serv
            
            reporte.append(f"  -> {id_tarea} ({u},{v}) -> DH: [{str_dh}] | Demanda: {dem_serv} | Costo (DH + Serv): {costo_dh} + {costo_serv} = {costo_total_paso}")
            nodo_actual = v
            
        if nodo_actual != deposito:
            camino_ret = nx.shortest_path(G, source=nodo_actual, target=deposito, weight='cost')
            costo_ret = nx.shortest_path_length(G, source=nodo_actual, target=deposito, weight='cost')
            str_ret = " -> ".join(map(str, camino_ret))
        else:
            costo_ret, str_ret = 0, f"Ninguno (ya en {deposito})"
            
        costo_vehiculo += costo_ret
        reporte.append(f"  -> REGRESO A DEPÓSITO ({deposito}) -> DH: [{str_ret}] | Costo Regreso: {costo_ret}")
        
        estado_cap = "OK" if demanda_vehiculo <= capacidad_max else "EXCEDIDA"
        reporte.append(f"  => TOTAL RUTA {i + 1}: Costo Total = {costo_vehiculo} | Demanda Total = {demanda_vehiculo} / {capacidad_max} [{estado_cap}]\n")
        costos_rutas.append(costo_vehiculo)
        costo_total_solucion += costo_vehiculo
        
    reporte.append("="*80)
    reporte.append(f"COSTO TOTAL DE LA SOLUCIÓN: {costo_total_solucion}")
    reporte.append("="*80 + "\n")
    
    # Imprimimos y también retornamos el texto para poder guardarlo
    texto_final = "\n".join(reporte)
    print(texto_final)
    return costos_rutas, costo_total_solucion, texto_final

def aplicar_y_evaluar_vecindario(solucion, data, G, matriz_distancias, operador="swap", p_inter=0.5, max_intentos=100):
    intentos, num_vehiculos, vecino_encontrado, nueva, detalles_cambio = 0, len(solucion), False, [], ""
    
    while intentos < max_intentos:
        intentos += 1
        nueva = copy.deepcopy(solucion)
        activas = [i for i, r in enumerate(nueva) if len(r) > 0]
        if not activas:
            detalles_cambio = "No hay tareas para mover."; break
            
        es_inter = (random.random() < p_inter) and (len(activas) >= 2)
        tipo, rutas_modificadas = "Intra", set()
        
        if operador == "swap":
            if es_inter: r1, r2 = random.sample(activas, 2); tipo = "Inter"
            else:
                r1 = r2 = random.choice(activas)
                if len(nueva[r1]) < 2: continue
            i1, i2 = random.randrange(len(nueva[r1])), random.randrange(len(nueva[r2]))
            t1, t2 = nueva[r1][i1], nueva[r2][i2]
            nueva[r1][i1], nueva[r2][i2] = t2, t1
            rutas_modificadas.update([r1, r2])
            detalles_cambio = f"SWAP ({tipo}-ruta):\n  -> Tarea '{t1}' (Ruta {r1+1}, Índice {i1}) <--> Tarea '{t2}' (Ruta {r2+1}, Índice {i2})"
            
        elif operador == "insertion":
            r_orig = random.choice(activas)
            if es_inter:
                rutas_posibles = [i for i in range(num_vehiculos) if i != r_orig]
                if not rutas_posibles: continue
                r_dest, tipo = random.choice(rutas_posibles), "Inter"
            else:
                r_dest = r_orig
                if len(nueva[r_orig]) < 2: continue
            i_orig = random.randrange(len(nueva[r_orig]))
            t_movida = nueva[r_orig].pop(i_orig)
            i_dest = random.randint(0, len(nueva[r_dest]))
            nueva[r_dest].insert(i_dest, t_movida)
            rutas_modificadas.update([r_orig, r_dest])
            detalles_cambio = f"INSERTION ({tipo}-ruta):\n  -> Tarea '{t_movida}' extraída de Ruta {r_orig+1} (Índice {i_orig})\n  -> Insertada en Ruta {r_dest+1} (Índice {i_dest})"
            
        elif operador == "inversion":
            rutas_validas = [r for r in activas if len(nueva[r]) >= 2]
            if not rutas_validas: continue
            r_idx = random.choice(rutas_validas)
            a, b = random.sample(range(len(nueva[r_idx])), 2)
            i, j = min(a, b), max(a, b)
            segmento_orig = nueva[r_idx][i:j+1]
            nueva[r_idx][i:j+1] = segmento_orig[::-1]
            rutas_modificadas.add(r_idx)
            detalles_cambio = f"INVERSION (Intra-ruta):\n  -> Ruta {r_idx+1}, segmento invertido desde índice {i} hasta {j}\n  -> ({segmento_orig} pasa a ser {segmento_orig[::-1]})"
            
        movimiento_factible = True
        for r_idx in rutas_modificadas:
            if not es_ruta_factible(nueva[r_idx], data, matriz_distancias):
                movimiento_factible = False; break
                
        if movimiento_factible:
            vecino_encontrado = True; break

    reporte_debug = "\n" + "*"*80 + "\n🔍 DEBUG: APLICACIÓN DE OPERADOR DE VECINDARIO\n" + "*"*80 + "\nSOLUCIÓN ORIGINAL:\n"
    for i, r in enumerate(solucion): reporte_debug += f"  Ruta {i+1}: {r}\n"
    
    if not vecino_encontrado:
        reporte_debug += f"\nOPERACIÓN FALLIDA:\n  -> Tras {max_intentos} intentos, no se halló un vecino factible.\n" + "*"*80 + "\n"
        print(reporte_debug)
        return solucion, 0, reporte_debug
        
    reporte_debug += f"\nDETALLE DEL CAMBIO:\n{detalles_cambio}\n\nVECINO RESULTANTE:\n"
    for i, r in enumerate(nueva): reporte_debug += f"  Ruta {i+1}: {r}\n"
    reporte_debug += "*"*80 + "\n"
    
    print(reporte_debug)
    costos_rutas, costo_total_solucion, texto_eval = calcular_y_mostrar_rutas_compacto(nueva, data, G)
    
    return nueva, costo_total_solucion, reporte_debug + texto_eval