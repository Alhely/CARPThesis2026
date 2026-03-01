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

def dibujar_rutas(G, solucion, data, ruta_idx=None, k_layout=1.5, metaheuristica="Desconocida"):
    deposito = data.get('DEPOSITO', 1)
    nombre_instancia = data.get('NOMBRE', 'Instancia_N/A')
    info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}

    pos = nx.spring_layout(G, k=k_layout, iterations=200, seed=42)
    fig = go.Figure()

    # 1. DIBUJAR ARISTAS BASE
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

        ruta = solucion[ruta_idx]
        nodo_actual = deposito
        
        texto_detallado = f"### 📍 Recorrido Completo\n\n**Ruta {ruta_idx + 1}**\n\n---\n\n"
        texto_detallado += f"🏁 **Inicio en Depósito ({deposito})**\n\n"

        for id_tarea in ruta:
            tarea = info_tareas[id_tarea]
            u, v = tarea['nodos']

            if nodo_actual != u:
                camino_dh = nx.shortest_path(G, source=nodo_actual, target=u, weight='cost')
                str_dh = " → ".join(map(str, camino_dh))
                texto_detallado += f"🔸 *Viaje vacío (DH):* `{str_dh}`\n\n"
                
                for i in range(len(camino_dh) - 1):
                    n1, n2 = camino_dh[i], camino_dh[i+1]
                    arco = tuple(sorted((n1, n2)))
                    count = conteo_arcos.get(arco, 0)
                    x0, y0, x1, y1 = calcular_coordenadas_offset(pos[n1][0], pos[n1][1], pos[n2][0], pos[n2][1], count)
                    conteo_arcos[arco] = count + 1
                    annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                            showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH, arrowcolor='#e67e22'))

            texto_detallado += f"🟩 **Servicio {id_tarea}:** `({u} → {v})`\n\n"
            
            arco_serv = tuple(sorted((u, v)))
            count_serv = conteo_arcos.get(arco_serv, 0)
            x0_s, y0_s, x1_s, y1_s = calcular_coordenadas_offset(pos[u][0], pos[u][1], pos[v][0], pos[v][1], count_serv)
            conteo_arcos[arco_serv] = count_serv + 1
            annotations.append(dict(ax=x0_s, ay=y0_s, axref='x', ayref='y', x=x1_s, y=y1_s, xref='x', yref='y',
                                    showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH+1.5, arrowcolor='#2ecc71'))
            nodo_actual = v

        if nodo_actual != deposito:
            camino_ret = nx.shortest_path(G, source=nodo_actual, target=deposito, weight='cost')
            str_ret = " → ".join(map(str, camino_ret))
            texto_detallado += f"🔸 *Viaje regreso (DH):* `{str_ret}`\n\n"
            
            for i in range(len(camino_ret) - 1):
                n1, n2 = camino_ret[i], camino_ret[i+1]
                arco_ret = tuple(sorted((n1, n2)))
                count_ret = conteo_arcos.get(arco_ret, 0)
                x0, y0, x1, y1 = calcular_coordenadas_offset(pos[n1][0], pos[n1][1], pos[n2][0], pos[n2][1], count_ret)
                conteo_arcos[arco_ret] = count_ret + 1
                annotations.append(dict(ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                                        showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH, arrowcolor='#e67e22'))

        texto_detallado += f"🏁 **Fin en Depósito ({deposito})**\n"
        texto_leyenda = texto_detallado

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#2ecc71', width=3), name='🟩 Servicio (Tarea)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#e67e22', width=3, dash='dash'), name='🔸 Deadheading (Viaje vacío)'))

        titulo_grafica = f"Instancia: {nombre_instancia} | Metaheurística: {metaheuristica} | Ruta {ruta_idx + 1}"

    # ==========================================
    # MODO GENERAL (Todas las rutas)
    # ==========================================
    else:
        colores = px.colors.qualitative.Plotly
        texto_leyenda = "### 📋 Secuencias Generales\n\n"

        for i, ruta in enumerate(solucion):
            if not ruta: continue
            color = colores[i % len(colores)]
            texto_leyenda += f"**Ruta {i+1}:**\n`{formatear_ruta_texto(ruta, info_tareas)}`\n\n"
            
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
                                        showarrow=True, arrowhead=2, arrowsize=ARR_SIZE, arrowwidth=ARR_WIDTH, arrowcolor=color, opacity=0.85))
            
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color, width=3), name=f'Ruta {i+1}'))

        titulo_grafica = f"Instancia: {nombre_instancia} | Metaheurística: {metaheuristica} | Rutas Completas"

    # 2. DIBUJAR NODOS (Formato Antiguo Restaurado)
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_text.append(f"<b>Nodo {node}</b>" + (" (Depósito)" if node == deposito else ""))
        if node == deposito:
            node_color.append('#ff4d4d'); node_size.append(22)
        else:
            node_color.append('#bdc3c7'); node_size.append(12)

    # Nodos limpios sin HTML forzado
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()], textposition="bottom center",
        hoverinfo="text", hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        showlegend=False
    ))

    # 3. MÁRGENES Y CONFIGURACIÓN FINAL (Forzando colores legibles)
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_pad = (max(x_vals) - min(x_vals)) * 0.15 
    y_pad = (max(y_vals) - min(y_vals)) * 0.15

    fig.update_layout(
        title=dict(
            text=titulo_grafica, 
            x=0.5, 
            font=dict(size=18, color='black') # FORZAR TÍTULO NEGRO
        ), 
        font=dict(color='black'), # FORZAR FUENTE GLOBAL NEGRA (Evita que desaparezcan leyendas)
        annotations=annotations,
        hovermode='closest',
        plot_bgcolor='white', # Fondo interno blanco
        paper_bgcolor='white', # Fondo externo blanco (CRUCIAL para que no haya transparencias invisibles)
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(x_vals)-x_pad, max(x_vals)+x_pad]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[min(y_vals)-y_pad, max(y_vals)+y_pad]),
        margin=dict(l=20, r=20, t=60, b=20),
        height=700,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01, 
            bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="lightgray", borderwidth=1,
            font=dict(color='black'), # FORZAR TEXTO DE LEYENDA NEGRO
            itemclick=False, itemdoubleclick=False 
        )
    )
    
    return fig, texto_leyenda