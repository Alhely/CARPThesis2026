
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

def formatear_ruta_texto(ruta, info_tareas):
    detalles = []
    for id_tarea in ruta:
        u, v = info_tareas[id_tarea]['nodos']
        detalles.append(f"{id_tarea}({u}→{v})")
    return "[" + ", ".join(detalles) + "]"

def dibujar_rutas(G, solucion, data, ruta_idx=None, k_layout=1.5):
    """Genera un grafo interactivo con Plotly usando flechas direccionales."""
    deposito = data.get('DEPOSITO', 1)
    info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}

    # Calculamos posiciones de los nodos (misma semilla para consistencia)
    pos = nx.spring_layout(G, k=k_layout, iterations=200, seed=42)
    fig = go.Figure()

    # 1. DIBUJAR LAS ARISTAS BASE DEL GRAFO (Fondo tenue)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#ecf0f1'),
        hoverinfo='none',
        showlegend=False,
        name='Red Base'
    ))

    annotations = []
    texto_leyenda = ""

    # ==========================================
    # MODO DETALLE (Una sola ruta)
    # ==========================================
    if ruta_idx is not None:
        if ruta_idx >= len(solucion) or not solucion[ruta_idx]:
            fig.update_layout(title=f"Ruta {ruta_idx + 1} está vacía.")
            return fig

        ruta = solucion[ruta_idx]
        nodo_actual = deposito
        texto_leyenda = f"<b>Secuencia Ruta {ruta_idx + 1}:</b><br>" + formatear_ruta_texto(ruta, info_tareas)

        for id_tarea in ruta:
            tarea = info_tareas[id_tarea]
            u, v = tarea['nodos']

            # A) Deadheading (Naranja)
            if nodo_actual != u:
                camino_dh = nx.shortest_path(G, source=nodo_actual, target=u, weight='cost')
                for i in range(len(camino_dh) - 1):
                    n1, n2 = camino_dh[i], camino_dh[i+1]
                    annotations.append(dict(
                        ax=pos[n1][0], ay=pos[n1][1], axref='x', ayref='y',
                        x=pos[n2][0], y=pos[n2][1], xref='x', yref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor='#e67e22'
                    ))

            # B) Servicio (Verde)
            annotations.append(dict(
                ax=pos[u][0], ay=pos[u][1], axref='x', ayref='y',
                x=pos[v][0], y=pos[v][1], xref='x', yref='y',
                showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=4.5, arrowcolor='#2ecc71'
            ))
            nodo_actual = v

        # C) Regreso al Depósito (Naranja)
        if nodo_actual != deposito:
            camino_ret = nx.shortest_path(G, source=nodo_actual, target=deposito, weight='cost')
            for i in range(len(camino_ret) - 1):
                n1, n2 = camino_ret[i], camino_ret[i+1]
                annotations.append(dict(
                    ax=pos[n1][0], ay=pos[n1][1], axref='x', ayref='y',
                    x=pos[n2][0], y=pos[n2][1], xref='x', yref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor='#e67e22'
                ))

        # "Traces" fantasma para que aparezcan en la leyenda
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#2ecc71', width=4), name='Servicio (Verde)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#e67e22', width=2), name='Deadheading (Naranja)'))

    # ==========================================
    # MODO GENERAL (Todas las rutas)
    # ==========================================
    else:
        colores = px.colors.qualitative.Plotly
        texto_leyenda = "<b>Secuencias:</b><br>"

        for i, ruta in enumerate(solucion):
            if not ruta: continue
            color = colores[i % len(colores)]
            texto_leyenda += f"<b>R{i+1}:</b> {formatear_ruta_texto(ruta, info_tareas)}<br>"
            
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
                annotations.append(dict(
                    ax=pos[n1][0], ay=pos[n1][1], axref='x', ayref='y',
                    x=pos[n2][0], y=pos[n2][1], xref='x', yref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor=color, opacity=0.8
                ))
            
            # Trace para la leyenda
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color, width=3), name=f'Ruta {i+1}'))

    # 2. DIBUJAR LOS NODOS (Por encima de todo)
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"<b>Nodo {node}</b>" + (" (Depósito)" if node == deposito else ""))
        if node == deposito:
            node_color.append('#ff4d4d')
            node_size.append(22)
        else:
            node_color.append('#bdc3c7')
            node_size.append(12)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[str(n) for n in G.nodes()], textposition="bottom center",
        hoverinfo="text", hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=1, color='white')),
        showlegend=False
    ))

    # 3. CAJA DE LEYENDA (Secuencia de tareas)
    fig.add_annotation(
        text=texto_leyenda, align='left', showarrow=False,
        xref='paper', yref='paper', x=0.01, y=0.99,
        bgcolor='rgba(255,255,255,0.9)', bordercolor='gray', borderwidth=1,
        font=dict(family="monospace", size=11)
    )

    # 4. CONFIGURACIÓN FINAL DEL LAYOUT
    fig.update_layout(
        annotations=annotations,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650  # Altura del gráfico interactivo
    )
    
    return fig