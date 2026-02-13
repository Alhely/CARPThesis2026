import re
import networkx as nx
import numpy as np
import os
import random
import copy
import math
import time
import csv
import pandas as pd
from datetime import datetime

# =============================================================================
# CLASE CarpLib: VERSIÓN FINAL INTEGRADA
# =============================================================================

class CarpLib:
    def __init__(self):
        self.datos = None
        self.G = None
        self.m_dist = None
        self.alcanzables = []
        self.id_instancia = ""

    # --- TAREA 1: CARGA E IMPRESIÓN DE DATOS ---
    def cargar_instancia(self, ruta_archivo, algoritmo_dist="dijkstra"):
        print(f"\n{'='*80}\n[1] CARGANDO INSTANCIA: {ruta_archivo}\n{'='*80}")
        self.id_instancia = os.path.splitext(os.path.basename(ruta_archivo))[0]
        self.datos = self._leer_dat(ruta_archivo)
        
        self.G = nx.Graph()
        self.G.add_nodes_from(range(1, self.datos['VERTICES'] + 1))
        for item in self.datos['LISTA_ARISTAS_REQ']:
            u, v = item['arco']
            self.G.add_edge(u, v, weight=item['coste'], demanda=item['demanda'])
        
        self.generar_matriz_distancias(algoritmo_dist)
        self.analizar_conectividad()
        
        print("\n--- OBJETO: DATOS CONFIGURADOS ---")
        for k, v in self.datos.items():
            if k != 'LISTA_ARISTAS_REQ': print(f"{k}: {v}")

    def _leer_dat(self, ruta):
        instancia = {}
        aristas = []
        dentro_de_lista = False
        with open(ruta, 'r', encoding='utf-8-sig') as f:
            for linea in f:
                linea = linea.strip()
                if not linea: continue
                if "LISTA_ARISTAS_REQ" in linea: dentro_de_lista = True; continue
                if "DEPOSITO" in linea:
                    dentro_de_lista = False
                    m = re.search(r'\d+', linea.split(":")[-1])
                    if m: instancia["DEPOSITO"] = int(m.group())
                    continue
                if dentro_de_lista:
                    n = re.findall(r'\d+', linea)
                    if len(n) >= 4:
                        aristas.append({'arco': (int(n[0]), int(n[1])), 'coste': int(n[2]), 'demanda': int(n[3])})
                elif ":" in linea:
                    k, v = linea.split(":", 1)
                    v_l = re.sub(r'\(.*?\)', '', v).strip()
                    try: instancia[k.strip()] = int(re.search(r'\d+', v_l).group())
                    except: instancia[k.strip()] = v_l
        instancia["LISTA_ARISTAS_REQ"] = aristas
        return instancia

    # --- TAREA 2: MATRICES E IMPRESIÓN ---
    def generar_matriz_distancias(self, algoritmo):
        n = self.datos['VERTICES']
        self.m_dist = np.full((n + 1, n + 1), np.inf)
        np.fill_diagonal(self.m_dist, 0)
        
        dist_dict = nx.floyd_warshall(self.G, weight='weight') if algoritmo == "floyd-warshall" else dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
            
        for u in dist_dict:
            for v in dist_dict[u]:
                self.m_dist[int(u)][int(v)] = dist_dict[u][v]
        
        print("\n--- OBJETO: MATRIZ DE DISTANCIAS (Muestra) ---")
        print(pd.DataFrame(self.m_dist).iloc[1:7, 1:7])

    def analizar_conectividad(self):
        dep = self.datos.get('DEPOSITO', 1)
        self.alcanzables = [i+1 for i, it in enumerate(self.datos['LISTA_ARISTAS_REQ']) 
                           if self.m_dist[dep][it['arco'][0]] != np.inf]
        print(f"\n--- OBJETO: CONECTIVIDAD ---\nTareas alcanzables: {len(self.alcanzables)}")

    # --- TAREA 3: SOLUCIÓN INICIAL ---
    def generar_solucion_inicial(self):
        vehiculos, cap_max = self.datos['VEHICULOS'], self.datos['CAPACIDAD']
        solucion = [[] for _ in range(vehiculos)]
        tareas = self.alcanzables.copy()
        random.shuffle(tareas)
        v_idx, carga = 0, 0
        for t_id in tareas:
            dem = self.datos['LISTA_ARISTAS_REQ'][t_id-1]['demanda']
            if carga + dem <= cap_max:
                solucion[v_idx].append(t_id); carga += dem
            elif v_idx + 1 < vehiculos:
                v_idx += 1; solucion[v_idx].append(t_id); carga = dem
        return solucion

    def calcular_costo_y_factibilidad(self, solucion):
        cap_max, dep = self.datos['CAPACIDAD'], self.datos.get('DEPOSITO', 1)
        costo_total = 0
        for ruta in solucion:
            if not ruta: continue
            carga, pos = 0, dep
            for t_id in ruta:
                arco_info = self.datos['LISTA_ARISTAS_REQ'][t_id-1]
                u, v = arco_info['arco']
                carga += arco_info['demanda']
                d_u, d_v = self.m_dist[pos][u], self.m_dist[pos][v]
                if carga > cap_max or (d_u == np.inf and d_v == np.inf): return float('inf')
                costo_total += min(d_u, d_v) + arco_info['coste']
                pos = v if d_u <= d_v else u
            if self.m_dist[pos][dep] == np.inf: return float('inf')
            costo_total += self.m_dist[pos][dep]
        return costo_total

    def calcular_detalle_por_ruta(self, solucion):
        """
        Retorna costo total, costos por ruta, capacidad usada por ruta y arcos intermedios
        (deadheading) por ruta. Los arcos intermedios son los tramos recorridos sin servicio,
        usando la matriz de costes mínimos (camino más corto entre nodos).
        """
        cap_max = self.datos['CAPACIDAD']
        dep = self.datos.get('DEPOSITO', 1)
        costos_rutas = []
        capacidad_rutas = []
        segmentos_por_ruta = []  # list of list of (desde, hasta, distancia) deadhead

        for ruta in solucion:
            costo_ruta = 0.0
            carga_ruta = 0
            segmentos = []
            if not ruta:
                costos_rutas.append(0.0)
                capacidad_rutas.append(0)
                segmentos_por_ruta.append(segmentos)
                continue

            pos = dep
            for t_id in ruta:
                arco_info = self.datos['LISTA_ARISTAS_REQ'][t_id - 1]
                u, v = arco_info['arco']
                d_u, d_v = self.m_dist[pos][u], self.m_dist[pos][v]
                if d_u == np.inf and d_v == np.inf:
                    costos_rutas.append(float('inf'))
                    capacidad_rutas.append(carga_ruta + arco_info['demanda'])
                    segmentos_por_ruta.append(segmentos)
                    return float('inf'), costos_rutas, capacidad_rutas, segmentos_por_ruta
                dist_dead = min(d_u, d_v)
                nodo_llegada = v if d_u <= d_v else u
                segmentos.append((int(pos), int(nodo_llegada), float(dist_dead)))
                costo_ruta += dist_dead + arco_info['coste']
                carga_ruta += arco_info['demanda']
                pos = nodo_llegada

            if self.m_dist[pos][dep] == np.inf:
                costos_rutas.append(float('inf'))
                capacidad_rutas.append(carga_ruta)
                segmentos_por_ruta.append(segmentos)
                return float('inf'), costos_rutas, capacidad_rutas, segmentos_por_ruta
            segmentos.append((int(pos), int(dep), float(self.m_dist[pos][dep])))
            costo_ruta += self.m_dist[pos][dep]
            costos_rutas.append(costo_ruta)
            capacidad_rutas.append(carga_ruta)
            segmentos_por_ruta.append(segmentos)

        costo_total = sum(costos_rutas)
        return costo_total, costos_rutas, capacidad_rutas, segmentos_por_ruta

    # --- TAREA 4: OPERADORES ---
    def mutar(self, solucion, operador="swap", p_inter=0.7):
        nueva = copy.deepcopy(solucion)
        activas = [i for i, r in enumerate(nueva) if r]
        if not activas: return nueva, "Ninguno"
        es_inter = (random.random() < p_inter) and (len(activas) >= 2)
        tipo = "Intra"
        if operador == "swap":
            r1, r2 = (random.sample(activas, 2) if es_inter else (random.choice(activas),)*2)
            if r1 != r2: tipo = "Inter"
            i1, i2 = random.randrange(len(nueva[r1])), random.randrange(len(nueva[r2]))
            nueva[r1][i1], nueva[r2][i2] = nueva[r2][i2], nueva[r1][i1]
        elif operador == "insertion":
            r_orig = random.choice(activas)
            t = nueva[r_orig].pop(random.randrange(len(nueva[r_orig])))
            r_dest = random.choice([i for i in range(len(nueva)) if i != r_orig]) if es_inter else r_orig
            if r_orig != r_dest: tipo = "Inter"
            nueva[r_dest].insert(random.randint(0, len(nueva[r_dest])), t)
        elif operador == "inversion":
            r_idx = random.choice(activas)
            if len(nueva[r_idx]) > 1:
                a, b = random.sample(range(len(nueva[r_idx])), 2); i, j = min(a,b), max(a,b)
                nueva[r_idx][i:j+1] = nueva[r_idx][i:j+1][::-1]
        return nueva, tipo