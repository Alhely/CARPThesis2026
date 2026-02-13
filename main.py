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
        # Caja de texto con scroll para mostrar los datos de la instancia
        text_frame = ttk.Frame(self.tab_datos)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_datos = tk.Text(
            text_frame, wrap=tk.NONE, font=("Consolas", 10), state=tk.DISABLED
        )
        self.txt_datos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.txt_datos.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_datos.configure(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(
            self.tab_datos, orient=tk.HORIZONTAL, command=self.txt_datos.xview
        )
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_datos.configure(xscrollcommand=scroll_x.set)

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
        # Mostrar el archivo de instancia EXACTAMENTE como está en el texto original
        if not self.current_file or not os.path.exists(self.current_file):
            return

        self.txt_datos.configure(state=tk.NORMAL)
        self.txt_datos.delete("1.0", tk.END)

        try:
            with open(self.current_file, "r", encoding="utf-8-sig") as f:
                contenido = f.read()
        except UnicodeDecodeError:
            # Fallback simple si la codificación es distinta
            with open(self.current_file, "r", errors="replace") as f:
                contenido = f.read()

        # Insertar el contenido tal cual está en el archivo
        self.txt_datos.insert(tk.END, contenido)

        self.txt_datos.configure(state=tk.DISABLED)

    def _actualizar_grafo(self):
        if self.carp.G is None or len(self.carp.G.nodes) == 0:
            return

        self.ax.clear()
        self.ax.set_title("Grafo de la instancia")

        # Layout para el grafo
        try:
            pos = nx.spring_layout(self.carp.G, seed=42)
        except Exception:
            pos = nx.random_layout(self.carp.G)

        # Dibujar nodos y aristas
        nx.draw_networkx_nodes(self.carp.G, pos, ax=self.ax, node_color="#1976D2", node_size=300)
        nx.draw_networkx_labels(self.carp.G, pos, ax=self.ax, font_size=8, font_color="white")

        # Etiquetas con coste y demanda
        edge_labels = {
            (u, v): f"c:{d['weight']}, q:{d.get('demanda', 0)}"
            for u, v, d in self.carp.G.edges(data=True)
        }
        nx.draw_networkx_edges(self.carp.G, pos, ax=self.ax, edge_color="#555555")
        nx.draw_networkx_edge_labels(self.carp.G, pos, edge_labels=edge_labels, ax=self.ax, font_size=7)

        self.ax.axis("off")
        self.canvas.draw()

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
