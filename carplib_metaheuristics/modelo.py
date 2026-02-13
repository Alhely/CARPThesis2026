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

  