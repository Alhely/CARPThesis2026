import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import math

def formatear_ruta_texto(ruta, info_tareas):
    detalles = []
    for id_tarea in ruta:
        u, v = info_tareas[id_tarea]['nodos']
        detalles.append(f"{id_tarea}({u}→{v})")
    return "[" + ", ".join(detalles) + "]"

def calcular_coordenadas_offset(x0, y0, x1, y1, count, base_offset=0.03):
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0: return x0, y0, x1, y1

    nx_dir, ny_dir = dx / length, dy / length
    ox, oy = -ny_dir, nx_dir 

    if count == 0:
        shift = 0
    else:
        sign = 1 if count % 2 != 0 else -1
        multiplier = (count + 1) // 2
        shift = sign * multiplier * base_offset

    acortar_inicio = 0.04
    acortar_fin = 0.06
    
    x0_adj = x0 + nx_dir * acortar_inicio + ox * shift
    y0_adj = y0 + ny_dir * acortar_inicio + oy * shift
    x1_adj = x1 - nx_dir * acortar_fin + ox * shift
    y1_adj = y1 - ny_dir * acortar_fin + oy * shift

    return x0_adj, y0_adj, x1_adj, y1_adj

def obtener_secuencia_movimientos(G, ruta, deposito, info_tareas):
    movimientos = []
    nodo_actual = deposito
    for id_tarea in ruta:
        tarea = info_tareas[id_tarea]
        u, v = tarea['nodos']
        if nodo_actual != u:
            camino_dh = nx.shortest_path(G, source=nodo_actual, target=u, weight='cost')
            for i in range(len(camino_dh) - 1):
                movimientos.append({'u': camino_dh[i], 'v': camino_dh[i+1], 'tipo': 'DH', 'tarea': None})
        movimientos.append({'u': u, 'v': v, 'tipo': 'Servicio', 'tarea': id_tarea})
        nodo_actual = v
    if nodo_actual != deposito:
        camino_ret = nx.shortest_path(G, source=nodo_actual, target=deposito, weight='cost')
        for i in range(len(camino_ret) - 1):
            movimientos.append({'u': camino_ret[i], 'v': camino_ret[i+1], 'tipo': 'DH', 'tarea': None})
    return movimientos

def dibujar_rutas(G, solucion, data, ruta_idx=None, k_layout=1.5, metaheuristica="Desconocida", paso_limite=None):
    deposito = data.get('DEPOSITO', 1)
    nombre_instancia = data.get('NOMBRE', 'Instancia_N/A')
    info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}

    pos = nx.spring_layout(G, k=k_layout, iterations=200, seed=42)
    fig = go.Figure()

    # ==========================================
    # 1. DIBUJAR ARISTAS BASE
    # ==========================================
    req_edges = set()
    for t in data.get('LISTA_ARISTAS_REQ', []):
        req_edges.add(tuple(sorted(t['nodos'])))

    if not solucion:
        # MODO: GRAFO ORIGINAL ESTÁTICO
        edge_x_req, edge_y_req = [], []
        edge_x_noreq, edge_y_noreq = [], []
        
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            if tuple(sorted((u, v))) in req_edges:
                edge_x_req.extend([x0, x1, None])
                edge_y_req.extend([y0, y1, None])
            else:
                edge_x_noreq.extend([x0, x1, None])
                edge_y_noreq.extend([y0, y1, None])

        # Se removieron los emojis del 'name' para una leyenda profesional
        fig.add_trace(go.Scatter(
            x=edge_x_req if edge_x_req else [None], 
            y=edge_y_req if edge_y_req else [None], 
            mode='lines', line=dict(width=3.5, color='#2980b9'),
            hoverinfo='none', showlegend=True, name='Arista Requerida (Con Demanda)'
        ))
        
        fig.add_trace(go.Scatter(
            x=edge_x_noreq if edge_x_noreq else [None], 
            y=edge_y_noreq if edge_y_noreq else [None], 
            mode='lines', line=dict(width=1.5, color='#bdc3c7', dash='dash'),
            hoverinfo='none', showlegend=True, name='Arista No Requerida (Solo Tránsito)'
        ))
    else:
        # MODO: RUTAS SUPERPUESTAS
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
    conteo_arcos = {} 
    titulo_grafica = ""

    ARR_SIZE = 1.0   
    ARR_WIDTH = 1.8  

    # ==========================================
    # MODO DETALLE (Una sola ruta)
    # ==========================================
    if ruta_idx is not None:
        if ruta_idx >= len(solucion) or not solucion[ruta_idx]:
            return fig, "La ruta seleccionada está vacía."

        movimientos = obtener_secuencia_movimientos(G, solucion[ruta_idx], deposito, info_tareas)
        total_pasos = len(movimientos)
        paso_actual = total_pasos if paso_limite is None else paso_limite

        texto_detallado = f"### 📍 Bitácora de Viaje\n\n**Ruta {ruta_idx + 1}**\n\n---\n\n🏁 **Inicio ({deposito})**\n\n"
        nodo_camion = deposito 

        for i in range(paso_actual):
            mov = movimientos[i]
            u, v, tipo, tarea = mov['u'], mov['v'], mov['tipo'], mov['tarea']
            nodo_camion = v 
            
            arco = tuple(sorted((u, v)))
            count = conteo_arcos.get(arco, 0)
            x0, y0, x1, y1 = calcular_coordenadas_offset(pos[u][0], pos[u][1], pos[v][0], pos[v][1], count)
            conteo_arcos[arco] = count + 1

            if tipo == 'DH':
                annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                        showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH, arrowcolor='#e67e22'))
            else:
                annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                        showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH+1.5, arrowcolor='#2ecc71'))

        for i, mov in enumerate(movimientos):
            marca = "👉 " if i == paso_actual - 1 else ""
            if mov['tipo'] == 'DH':
                texto_detallado += f"{marca}🔸 *DH:* `{mov['u']} → {mov['v']}`\n\n"
            else:
                texto_detallado += f"{marca}🟩 **Serv. {mov['tarea']}:** `{mov['u']} → {mov['v']}`\n\n"

        if paso_actual == total_pasos:
            texto_detallado += f"🏁 **Fin en ({deposito})**\n"
        
        texto_leyenda = texto_detallado

        if paso_actual >= 0:
            fig.add_trace(go.Scatter(
                x=[pos[nodo_camion][0]], y=[pos[nodo_camion][1] + 0.05], 
                mode='text', text=['🚚'], textfont=dict(size=35),
                hoverinfo='none', showlegend=False
            ))

        # Emojis removidos de la leyenda interactiva también
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#2ecc71', width=3), name='Servicio (Tarea)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#e67e22', width=3, dash='dash'), name='Deadheading (Viaje Vacío)'))
        
        titulo_grafica = f"Instancia: {nombre_instancia} | {metaheuristica} | Ruta {ruta_idx + 1}"

    # ==========================================
    # MODO GENERAL (Todas las rutas o Grafo Base)
    # ==========================================
    else:
        if not solucion:
            texto_leyenda = ""
            titulo_grafica = f"Instancia: {nombre_instancia} | {metaheuristica}"
        else:
            colores = px.colors.qualitative.Plotly
            texto_leyenda = "### 📋 Secuencias Generales\n\n"

            for i, ruta in enumerate(solucion):
                if not ruta: continue
                color = colores[i % len(colores)]
                texto_leyenda += f"**Ruta {i+1}:**\n`{formatear_ruta_texto(ruta, info_tareas)}`\n\n"
                
                movimientos = obtener_secuencia_movimientos(G, ruta, deposito, info_tareas)
                for mov in movimientos:
                    u, v = mov['u'], mov['v']
                    arco = tuple(sorted((u, v)))
                    count = conteo_arcos.get(arco, 0)
                    x0, y0, x1, y1 = calcular_coordenadas_offset(pos[u][0], pos[u][1], pos[v][0], pos[v][1], count)
                    conteo_arcos[arco] = count + 1
                    annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                            showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH, arrowcolor=color, opacity=0.85))
                
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color, width=3), name=f'Ruta {i+1}'))

            titulo_grafica = f"Instancia: {nombre_instancia} | {metaheuristica} | Rutas Completas"

    # ==========================================
    # 2. DIBUJAR NODOS
    # ==========================================
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_text.append(f"<b>Nodo {node}</b>" + (" (Depósito)" if node == deposito else ""))
        if node == deposito:
            node_color.append('#ff4d4d'); node_size.append(22)
        else:
            node_color.append('#bdc3c7'); node_size.append(12)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()], textposition="bottom center",
        hoverinfo="text", hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        showlegend=False
    ))

    # ==========================================
    # 3. MÁRGENES Y CONFIGURACIÓN FINAL
    # ==========================================
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_pad = (max(x_vals) - min(x_vals)) * 0.15 
    y_pad = (max(y_vals) - min(y_vals)) * 0.15

    fig.update_layout(
        title=dict(
            text=f"<b>{titulo_grafica}</b>", 
            x=0.5, 
            font=dict(size=18, color='black')
        ), 
        font=dict(color='black'), 
        annotations=annotations,
        hovermode='closest',
        plot_bgcolor='white', 
        paper_bgcolor='white', 
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(x_vals)-x_pad, max(x_vals)+x_pad]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(y_vals)-y_pad, max(y_vals)+y_pad]),
        margin=dict(l=20, r=20, t=60, b=20),
        height=700,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, 
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="lightgray", borderwidth=1,
            font=dict(color='black'), 
            itemclick=False, itemdoubleclick=False 
        )
    )
    
    return fig, texto_leyenda