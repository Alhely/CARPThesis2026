"""
Interfaz gráfica para visualizar:
 - Datos originales de la instancia
 - Grafo
 - Matriz de distancias
 - Matriz de demandas
 - Matriz de mínima distancia

Autor: Tesis CARP 2026
"""

import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import networkx as nx

from carplib_metaheuristics.modelo import CarpLib


class CarpGUI(tk.Tk):
    """
    Ventana principal de la aplicación CARP.
    """

    def __init__(self):
        super().__init__()

        self.title("CARPThesis2026 - Visualizador de Instancias CARP")
        self.geometry("1100x700")

        # Objeto de lógica
        self.carp = CarpLib()
        self.current_file = None

        # Algoritmo de caminos mínimos seleccionado
        self.alg_var = tk.StringVar(value="dijkstra")
        self.algoritmo_actual = None

        # Componentes principales
        self._crear_componentes()

    # ------------------------------------------------------------------
    # Creación de interfaz
    # ------------------------------------------------------------------
    def _crear_componentes(self):
        # Barra superior con botón de carga y etiqueta de archivo
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        btn_cargar = ttk.Button(
            top_frame, text="Cargar instancia (.dat)", command=self._cargar_instancia
        )
        btn_cargar.pack(side=tk.LEFT)

        self.lbl_archivo = ttk.Label(
            top_frame, text="Ninguna instancia cargada", foreground="gray"
        )
        self.lbl_archivo.pack(side=tk.LEFT, padx=10)

        # Selector de algoritmo de caminos mínimos
        ttk.Label(top_frame, text="Algoritmo de caminos mínimos:").pack(
            side=tk.LEFT, padx=(20, 5)
        )
        self.cbo_alg = ttk.Combobox(
            top_frame,
            textvariable=self.alg_var,
            state="readonly",
            values=["dijkstra", "floyd-warshall"],
            width=15,
        )
        self.cbo_alg.pack(side=tk.LEFT)

        # Notebook principal (pestañas)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Pestaña 1: Datos originales ---
        self.tab_datos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_datos, text="Datos de la instancia")
        self._crear_tab_datos()

        # --- Pestaña 2: Grafo ---
        self.tab_grafo = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_grafo, text="Grafo")
        self._crear_tab_grafo()

        # --- Pestaña 3: Matrices ---
        self.tab_matrices = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_matrices, text="Matrices")
        self._crear_tab_matrices()

    def _crear_tab_datos(self):
        # Panel principal con dos secciones: texto a la izquierda, grafo a la derecha
        paned = ttk.PanedWindow(self.tab_datos, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Panel izquierdo: Texto de la instancia ---
        frame_texto = ttk.Frame(paned)
        paned.add(frame_texto, weight=1)

        text_frame = ttk.Frame(frame_texto)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_datos = tk.Text(
            text_frame, wrap=tk.NONE, font=("Consolas", 10), state=tk.DISABLED
        )
        self.txt_datos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.txt_datos.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_datos.configure(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(
            frame_texto, orient=tk.HORIZONTAL, command=self.txt_datos.xview
        )
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_datos.configure(xscrollcommand=scroll_x.set)

        # --- Panel derecho: Grafo ---
        frame_grafo = ttk.Frame(paned)
        paned.add(frame_grafo, weight=1)

        # Figura de Matplotlib para el grafo en la pestaña de datos
        self.fig_datos = Figure(figsize=(5, 4), dpi=100)
        self.ax_datos = self.fig_datos.add_subplot(111)
        self.ax_datos.set_title("Grafo de la instancia")

        self.canvas_datos = FigureCanvasTkAgg(self.fig_datos, master=frame_grafo)
        self.canvas_datos.draw()
        self.canvas_datos.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _crear_tab_grafo(self):
        # Figura de Matplotlib embebida
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Grafo de la instancia")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_grafo)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _crear_tab_matrices(self):
        # Notebook interno para las matrices
        self.nb_matrices = ttk.Notebook(self.tab_matrices)
        self.nb_matrices.pack(fill=tk.BOTH, expand=True)

        # Distancias
        self.tab_dist = ttk.Frame(self.nb_matrices)
        self.nb_matrices.add(self.tab_dist, text="Matriz de distancias")
        self.txt_dist = self._crear_texto_matriz(self.tab_dist)

        # Demandas
        self.tab_dem = ttk.Frame(self.nb_matrices)
        self.nb_matrices.add(self.tab_dem, text="Matriz de demandas")
        self.txt_dem = self._crear_texto_matriz(self.tab_dem)

        # Mínima distancia (aquí se asume que es la misma matriz de distancias
        # calculada con caminos mínimos; se presenta por separado por claridad)
        self.tab_min = ttk.Frame(self.nb_matrices)
        self.nb_matrices.add(self.tab_min, text="Matriz de mínima distancia")
        self.txt_min = self._crear_texto_matriz(self.tab_min)


    @staticmethod
    def _crear_texto_matriz(parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 9), state=tk.DISABLED)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=txt.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        txt.configure(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=txt.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        txt.configure(xscrollcommand=scroll_x.set)

        return txt

    # ------------------------------------------------------------------
    # Lógica de carga y actualización de vistas
    # ------------------------------------------------------------------
    def _cargar_instancia(self):
        file_path = filedialog.askopenfilename(
            title="Selecciona un archivo de instancia (.dat)",
            filetypes=[("Archivos .dat", "*.dat"), ("Todos los archivos", "*.*")],
        )

        if not file_path:
            return

        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"El archivo no existe:\n{file_path}")
            return

        try:
            # Algoritmo elegido por el usuario
            algoritmo = self.alg_var.get() or "dijkstra"

            # Carga de datos y matrices con el algoritmo seleccionado
            self.carp.cargar_instancia(file_path, algoritmo_dist=algoritmo)
            self.current_file = file_path
            self.algoritmo_actual = algoritmo
            self.lbl_archivo.configure(text=os.path.basename(file_path), foreground="black")

            # Actualizar todas las vistas
            self._actualizar_datos()
            self._actualizar_grafo()
            self._actualizar_matrices()

            messagebox.showinfo("Instancia cargada", "La instancia se cargó correctamente.")
        except Exception as e:
            messagebox.showerror("Error al cargar instancia", str(e))

    def _actualizar_datos(self):
        # Mostrar el archivo de instancia con ID de tarea en cada arista requerida
        if not self.current_file or not os.path.exists(self.current_file):
            return

        self.txt_datos.configure(state=tk.NORMAL)
        self.txt_datos.delete("1.0", tk.END)

        try:
            with open(self.current_file, "r", encoding="utf-8-sig") as f:
                lineas = f.readlines()
        except UnicodeDecodeError:
            with open(self.current_file, "r", errors="replace") as f:
                lineas = f.readlines()

        # Añadir ID de tarea a cada arista requerida (dentro de LISTA_ARISTAS_REQ)
        dentro_lista = False
        id_tarea = 0
        for linea in lineas:
            linea_orig = linea.rstrip("\n\r")
            if "LISTA_ARISTAS_REQ" in linea:
                dentro_lista = True
                self.txt_datos.insert(tk.END, linea_orig + "\n")
                continue
            if "DEPOSITO" in linea:
                dentro_lista = False
            if dentro_lista:
                n = re.findall(r"\d+", linea)
                if len(n) >= 4:
                    id_tarea += 1
                    self.txt_datos.insert(tk.END, linea_orig + f"   # Tarea {id_tarea}\n")
                else:
                    self.txt_datos.insert(tk.END, linea_orig + "\n")
            else:
                self.txt_datos.insert(tk.END, linea_orig + "\n")

        self.txt_datos.configure(state=tk.DISABLED)

        # Actualizar el grafo en la pestaña de datos (sin análisis para ahorrar espacio)
        if self.carp.G is not None and len(self.carp.G.nodes) > 0:
            self._dibujar_grafo_en_ax(
                self.ax_datos, self.canvas_datos, mostrar_leyenda=True, mostrar_analisis=False
            )

    def _dibujar_grafo_en_ax(self, ax, canvas, mostrar_leyenda=True, mostrar_analisis=True):
        """
        Método auxiliar para dibujar el grafo en un ax y canvas dados.
        """
        if self.carp.G is None or len(self.carp.G.nodes) == 0:
            return

        ax.clear()
        ax.set_title("Grafo de la instancia y asignación de tareas")

        # Layout para el grafo
        try:
            pos = nx.spring_layout(self.carp.G, seed=42)
        except Exception:
            pos = nx.random_layout(self.carp.G)

        # -----------------------------
        # Colores de nodos:
        # - Azul   : nodo depósito
        # - Verde  : nodos que aparecen en aristas requeridas
        # - Gris   : otros nodos
        # -----------------------------
        lista_req = self.carp.datos.get("LISTA_ARISTAS_REQ", []) if self.carp.datos else []
        nodos_req = set()
        for item in lista_req:
            u, v = item["arco"]
            nodos_req.add(u)
            nodos_req.add(v)

        deposito = None
        if self.carp.datos:
            deposito = self.carp.datos.get("DEPOSITO", None)

        node_colors = []
        for n in self.carp.G.nodes():
            if deposito is not None and n == deposito:
                node_colors.append("blue")
            elif n in nodos_req:
                node_colors.append("green")
            else:
                node_colors.append("#999999")

        nx.draw_networkx_nodes(
            self.carp.G, pos, ax=ax, node_color=node_colors, node_size=320, edgecolors="black"
        )
        nx.draw_networkx_labels(self.carp.G, pos, ax=ax, font_size=8, font_color="white")

        # -----------------------------
        # Colores y etiquetas de aristas:
        # - Rojo para aristas requeridas
        # - Etiqueta incluye:
        #   T{k}  -> tarea k asociada a la arista requerida k
        #   dist: distancia (peso)
        #   dem : demanda
        # -----------------------------
        # Construir mapeo arista requerida -> índice de tarea
        tarea_por_arco = {}
        for idx, item in enumerate(lista_req, start=1):
            u, v = item["arco"]
            key = tuple(sorted((u, v)))
            tarea_por_arco[key] = idx

        edges = list(self.carp.G.edges())
        edge_colors = []
        edge_labels = {}

        for (u, v) in edges:
            key = tuple(sorted((u, v)))
            datos = self.carp.G.get_edge_data(u, v, default={})
            dist = datos.get("weight", 0)
            dem = datos.get("demanda", 0)

            if key in tarea_por_arco:
                k = tarea_por_arco[key]
                # Arista requerida -> roja y etiqueta con tarea
                edge_colors.append("red")
                edge_labels[(u, v)] = f"T{k} | dist:{dist}, dem:{dem}"
            else:
                # Arista no requerida (por si se añaden en el futuro)
                edge_colors.append("#555555")
                edge_labels[(u, v)] = f"dist:{dist}, dem:{dem}"

        nx.draw_networkx_edges(self.carp.G, pos, ax=ax, edgelist=edges, edge_color=edge_colors)
        nx.draw_networkx_edge_labels(
            self.carp.G, pos, edge_labels=edge_labels, ax=ax, font_size=7
        )

        # -----------------------------
        # Leyenda y análisis (opcionales)
        # -----------------------------
        if mostrar_leyenda:
            leyenda_texto = [
                "Leyenda:",
                "• dem: Demanda de la arista",
                "• dist: Distancia (peso) de la arista",
                "• T1, T2, ...: Tareas asignadas a aristas requeridas",
                "  (T1 = arista requerida 1, T2 = arista requerida 2, ...)",
            ]
            texto_leyenda = "\n".join(leyenda_texto)
            ax.text(
                0.02,
                0.98,
                texto_leyenda,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85, edgecolor="black"),
                family="monospace",
            )

        if mostrar_analisis:
            analisis_texto = self._generar_analisis_grafo()
            ax.text(
                0.98,
                0.02,
                analisis_texto,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.85, edgecolor="black"),
                family="monospace",
            )

        ax.axis("off")
        canvas.draw()

    def _actualizar_grafo(self):
        """Actualiza el grafo en la pestaña dedicada de grafo."""
        self._dibujar_grafo_en_ax(self.ax, self.canvas, mostrar_leyenda=True, mostrar_analisis=True)

    def _actualizar_matrices(self):
        if self.carp.m_dist is None or self.carp.datos is None:
            return

        n = self.carp.datos.get("VERTICES", 0)
        if n <= 0:
            return

        # --- Matriz de distancias (ya calculada en CarpLib como m_dist) ---
        dist_df = pd.DataFrame(self.carp.m_dist)
        # Para que las filas/columnas empiecen en 1
        dist_df = dist_df.iloc[1 : n + 1, 1 : n + 1]
        dist_df.index = range(1, n + 1)
        dist_df.columns = range(1, n + 1)

        self._mostrar_matriz_en_texto(self.txt_dist, dist_df, titulo="MATRIZ DE DISTANCIAS")

        # --- Matriz de demandas ---
        dem_mat = np.zeros((n + 1, n + 1), dtype=float)
        for item in self.carp.datos.get("LISTA_ARISTAS_REQ", []):
            u, v = item["arco"]
            dem = item["demanda"]
            dem_mat[u, v] = dem
            dem_mat[v, u] = dem  # grafo no dirigido

        dem_df = pd.DataFrame(dem_mat).iloc[1 : n + 1, 1 : n + 1]
        dem_df.index = range(1, n + 1)
        dem_df.columns = range(1, n + 1)

        self._mostrar_matriz_en_texto(self.txt_dem, dem_df, titulo="MATRIZ DE DEMANDAS")

        # --- Matriz de mínima distancia ---
        # Corresponde a las distancias mínimas entre pares de vértices.
        min_df = dist_df.copy()

        # Información sobre el algoritmo usado (basado en la documentación de NetworkX:
        # https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html)
        alg = self.algoritmo_actual or self.alg_var.get() or "dijkstra"
        if alg == "floyd-warshall":
            alg_nombre = "Floyd–Warshall"
            orden = "O(V^3) - recomendado para todos los pares en grafos densos."
        else:
            alg_nombre = "Dijkstra"
            orden = "O((V + E) log V) - elección general para grafos ponderados sin pesos negativos."

        descripcion_min = (
            "Esta matriz representa las distancias mínimas entre cada par de vértices del grafo.\n"
            f"Algoritmo utilizado: {alg_nombre} ({orden})\n"
            "Referencia: NetworkX Shortest Paths "
            "(https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html)\n"
        )

        self._mostrar_matriz_en_texto(
            self.txt_min,
            min_df,
            titulo="MATRIZ DE MÍNIMA DISTANCIA ENTRE VÉRTICES",
            descripcion=descripcion_min,
        )

    def _generar_analisis_grafo(self):
        """
        Genera un texto con el análisis del grafo usando NetworkX.
        Retorna un string formateado para mostrar en el gráfico.
        """
        if self.carp.G is None or self.carp.datos is None:
            return "Análisis no disponible"

        G = self.carp.G
        datos = self.carp.datos
        lineas = []

        # Tipo de grafo
        es_dirigido = G.is_directed()
        lineas.append(f"Tipo de grafo: {'Dirigido' if es_dirigido else 'No dirigido'}")

        # Número de nodos y aristas
        n_nodos = G.number_of_nodes()
        n_aristas = G.number_of_edges()
        lineas.append(f"Número de nodos: {n_nodos}")
        lineas.append(f"Número de aristas: {n_aristas}")

        # Conectividad global
        try:
            if not es_dirigido:
                conexo = nx.is_connected(G)
            else:
                conexo = nx.is_strongly_connected(G)
        except Exception:
            conexo = False

        lineas.append(f"¿Es conexo?: {'Sí' if conexo else 'No'}")
        if conexo:
            lineas.append("  Significa: existe un camino entre")
            lineas.append("  cualquier par de nodos")
        else:
            lineas.append("  Significa: hay nodos aislados")
            lineas.append("  o componentes desconectadas")

        # Depósito y nodos alcanzables
        deposito = datos.get("DEPOSITO", None)
        if deposito is not None and deposito in G.nodes:
            lineas.append(f"\nDepósito: nodo {deposito}")
            try:
                if not es_dirigido:
                    comp_dep = nx.node_connected_component(G, deposito)
                else:
                    comp_dep = nx.node_connected_component(G.to_undirected(), deposito)
            except Exception:
                comp_dep = set()

            nodos_no_alcanzables = sorted(set(G.nodes()) - set(comp_dep))
            if nodos_no_alcanzables:
                lineas.append(f"¿Nodos alcanzables?: No")
                if len(nodos_no_alcanzables) <= 10:
                    lineas.append(f"  Nodos no alcanzables: {nodos_no_alcanzables}")
                else:
                    lineas.append(f"  Nodos no alcanzables: {len(nodos_no_alcanzables)} nodos")
            else:
                lineas.append(f"¿Nodos alcanzables?: Sí")
                lineas.append("  Todos los nodos son alcanzables")

            # Arcos requeridos alcanzables desde el depósito
            lista_req = datos.get("LISTA_ARISTAS_REQ", [])
            arcos_no_alcanzables = []
            for item in lista_req:
                u, v = item["arco"]
                if u not in comp_dep and v not in comp_dep:
                    arcos_no_alcanzables.append((u, v))

            if arcos_no_alcanzables:
                lineas.append(f"\n¿Arcos requeridos alcanzables?: No")
                lineas.append(f"  {len(arcos_no_alcanzables)} arcos no pueden llegar al depósito")
            else:
                lineas.append(f"\n¿Arcos requeridos alcanzables?: Sí")
                lineas.append("  Todos los arcos requeridos pueden llegar al depósito")

        return "\n".join(lineas)

    @staticmethod
    def _mostrar_matriz_en_texto(widget_txt, df, titulo="MATRIZ", descripcion=None):
        widget_txt.configure(state=tk.NORMAL)
        widget_txt.delete("1.0", tk.END)
        widget_txt.insert(tk.END, f"=== {titulo} ===\n\n")
        if descripcion:
            widget_txt.insert(tk.END, descripcion + "\n")
        widget_txt.insert(tk.END, df.to_string(float_format=lambda x: f"{x:7.1f}"))
        widget_txt.configure(state=tk.DISABLED)


def main():
    app = CarpGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
