import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import math

def formatear_ruta_texto(ruta, info_tareas):
    """Crea el string detallado de la ruta: [T1(1→4), T9(9→16), ...]"""
    detalles = []
    for id_tarea in ruta:
        u, v = info_tareas[id_tarea]['nodos']
        detalles.append(f"{id_tarea}({u}→{v})")
    return "[" + ", ".join(detalles) + "]"

def calcular_coordenadas_offset(x0, y0, x1, y1, count, base_offset=0.04):
    """
    Geometría vectorial: Calcula coordenadas desplazadas lateralmente para 
    evitar sobreposición de flechas cuando se recorre un arco múltiples veces.
    """
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0: return x0, y0, x1, y1

    # Vector de dirección normalizado
    nx_dir, ny_dir = dx / length, dy / length
    # Vector ortogonal (perpendicular) para desplazar a los lados
    ox, oy = -ny_dir, nx_dir 

    # Lógica de "carriles": 0 al centro, 1 a la derecha, 2 a la izquierda, 3 más a la derecha...
    if count == 0:
        shift = 0
    else:
        sign = 1 if count % 2 != 0 else -1
        multiplier = (count + 1) // 2
        shift = sign * multiplier * base_offset

    # Acortamos ligeramente la flecha (inicio y fin) para que no tape el centro del nodo
    acortar_inicio = 0.05
    acortar_fin = 0.08
    
    x0_adj = x0 + nx_dir * acortar_inicio + ox * shift
    y0_adj = y0 + ny_dir * acortar_inicio + oy * shift
    x1_adj = x1 - nx_dir * acortar_fin + ox * shift
    y1_adj = y1 - ny_dir * acortar_fin + oy * shift

    return x0_adj, y0_adj, x1_adj, y1_adj

def dibujar_rutas(G, solucion, data, ruta_idx=None, k_layout=1.5):
    deposito = data.get('DEPOSITO', 1)
    info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}

    # Posiciones de los nodos limitadas entre -1 y 1
    pos = nx.spring_layout(G, k=k_layout, iterations=200, seed=42)
    fig = go.Figure()

    # 1. DIBUJAR ARISTAS BASE (Muy claras al fondo)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#e0e0e0'),
        hoverinfo='none', showlegend=False, name='Red Base'
    ))

    annotations = []
    texto_leyenda = ""
    # Diccionario para contar cuántas veces se recorre un arco y separar las flechas
    conteo_arcos = {} 

    # ==========================================
    # MODO DETALLE (Una sola ruta)
    # ==========================================
    if ruta_idx is not None:
        if ruta_idx >= len(solucion) or not solucion[ruta_idx]:
            return fig, "La ruta seleccionada está vacía."

        ruta = solucion[ruta_idx]
        nodo_actual = deposito
        texto_leyenda = f"**Secuencia de Tareas:**\n\n{formatear_ruta_texto(ruta, info_tareas)}"

        for id_tarea in ruta:
            tarea = info_tareas[id_tarea]
            u, v = tarea['nodos']

            # A) Deadheading (Naranja)
            if nodo_actual != u:
                camino_dh = nx.shortest_path(G, source=nodo_actual, target=u, weight='cost')
                for i in range(len(camino_dh) - 1):
                    n1, n2 = camino_dh[i], camino_dh[i+1]
                    arco = tuple(sorted((n1, n2)))
                    count = conteo_arcos.get(arco, 0)
                    
                    x0, y0, x1, y1 = calcular_coordenadas_offset(pos[n1][0], pos[n1][1], pos[n2][0], pos[n2][1], count)
                    conteo_arcos[arco] = count + 1
                    
                    annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor='#e67e22'))

            # B) Servicio (Verde)
            arco_serv = tuple(sorted((u, v)))
            count_serv = conteo_arcos.get(arco_serv, 0)
            x0_s, y0_s, x1_s, y1_s = calcular_coordenadas_offset(pos[u][0], pos[u][1], pos[v][0], pos[v][1], count_serv)
            conteo_arcos[arco_serv] = count_serv + 1
            
            annotations.append(dict(ax=x0_s, ay=y0_s, axref='x', ayref='y', x=x1_s, y=y1_s, xref='x', yref='y',
                                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor='#2ecc71'))
            nodo_actual = v

        # C) Regreso al Depósito (Naranja)
        if nodo_actual != deposito:
            camino_ret = nx.shortest_path(G, source=nodo_actual, target=deposito, weight='cost')
            for i in range(len(camino_ret) - 1):
                n1, n2 = camino_ret[i], camino_ret[i+1]
                arco_ret = tuple(sorted((n1, n2)))
                count_ret = conteo_arcos.get(arco_ret, 0)
                x0, y0, x1, y1 = calcular_coordenadas_offset(pos[n1][0], pos[n1][1], pos[n2][0], pos[n2][1], count_ret)
                conteo_arcos[arco_ret] = count_ret + 1
                
                annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor='#e67e22'))

        # Leyenda en la gráfica interactiva
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#2ecc71', width=3), name='Servicio (Tarea)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#e67e22', width=3), name='Deadheading (Viaje)'))

    # ==========================================
    # MODO GENERAL (Todas las rutas)
    # ==========================================
    else:
        colores = px.colors.qualitative.Plotly
        texto_leyenda = "**Secuencias Completas:**\n\n"

        for i, ruta in enumerate(solucion):
            if not ruta: continue
            color = colores[i % len(colores)]
            texto_leyenda += f"**Ruta {i+1}:** {formatear_ruta_texto(ruta, info_tareas)}\n\n"
            
            nodo_actual = deposito
            path_nodos = [deposito]

            for id_t in ruta:
                u, v = info_tareas[id_t]['nodos']
                if nodo_actual != u:
                    path_nodos.extend(nx.shortest_path(G, nodo_actual, u, weight='cost')[1:])
                path_nodos.append(v)
                nodo_actual = v
                
            if nodo_actual != deposito:
                path_nodos.extend(nx.shortest_path(G, nodo_actual, deposito, weight='cost')[1:])

            for k in range(len(path_nodos)-1):
                n1, n2 = path_nodos[k], path_nodos[k+1]
                arco = tuple(sorted((n1, n2)))
                count = conteo_arcos.get(arco, 0)
                
                x0, y0, x1, y1 = calcular_coordenadas_offset(pos[n1][0], pos[n1][1], pos[n2][0], pos[n2][1], count)
                conteo_arcos[arco] = count + 1
                
                annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor=color, opacity=0.9))
            
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color, width=3), name=f'Ruta {i+1}'))

    # 2. DIBUJAR LOS NODOS
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_text.append(f"<b>Nodo {node}</b>" + (" (Depósito)" if node == deposito else ""))
        if node == deposito:
            node_color.append('#ff4d4d'); node_size.append(22)
        else:
            node_color.append('#95a5a6'); node_size.append(14)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()], textposition="bottom center",
        hoverinfo="text", hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color='white')),
        showlegend=False
    ))

    # 3. LÓGICA DE MÁRGENES (Para que no se corte la imagen)
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_pad = (max(x_vals) - min(x_vals)) * 0.15 # 15% de margen extra
    y_pad = (max(y_vals) - min(y_vals)) * 0.15

    fig.update_layout(
        annotations=annotations,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(x_vals)-x_pad, max(x_vals)+x_pad]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(y_vals)-y_pad, max(y_vals)+y_pad]),
        margin=dict(l=20, r=20, t=30, b=20),
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    
    # Ahora devolvemos DOS cosas: La figura limpia, y el texto para ponerlo afuera
    return fig, texto_leyenda