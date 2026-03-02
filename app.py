import os
import sys
import io
import tempfile
import streamlit as st
import pandas as pd
import carp_core as carp
import meta_sa as sa
import viz_routes as viz

st.set_page_config(page_title="CARP Optimizer", page_icon="🚛", layout="wide")
st.title("🚛 Optimizador CARP")

# ==========================================
# BARRA LATERAL - INICIALIZACIÓN
# ==========================================
st.sidebar.header("1. Cargar Instancia")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo .dat", type=["dat"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    if st.sidebar.button("🚀 Inicializar Entorno"):
        with st.spinner("Procesando instancia y distancias (Sin generar solución)..."):
            d_noreq = carp.leer_carplib_dat(temp_path)
            ruta_run, logger = carp.iniciar_ejecucion(d_noreq, base_dir="runs_carp_ui")
            
            grafo = carp.generar_grafo_limpio(d_noreq, carpeta_salida=ruta_run)
            distancias = carp.calcular_matriz_distancias(grafo)
            
            # NOTA: Ya NO generamos la solución aquí. Se inicializan variables vacías.
            st.session_state.update({
                'd_noreq': d_noreq, 'ruta_run': ruta_run, 'logger': logger,
                'grafo': grafo, 'distancias': distancias,
                'sol_inicial': None, 'costo_inicial': None,
                'mejor_solucion_global': None, 'mejor_costo_global': None,
                'meta_usada': 'Ninguna',
                'solucion_actual': None, 'costo_actual': None, 'reporte_actual': ""
            })
        st.sidebar.success("¡Entorno y Grafo listos!")

# ==========================================
# ÁREA PRINCIPAL - PESTAÑAS
# ==========================================
if 'd_noreq' in st.session_state:
    data = st.session_state['d_noreq']
    
    # --- MÉTRICAS GLOBALES ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instancia", data.get("NOMBRE", "N/A"))
    c2.metric("Vehículos", data.get("VEHICULOS", 0))
    c3.metric("Capacidad Máxima", data.get("CAPACIDAD", 0))
    # Mostramos "---" si aún no hay solución
    costo_str = st.session_state.get('mejor_costo_global') if st.session_state.get('mejor_costo_global') is not None else "---"
    c4.metric("🏆 Mejor Costo Actual", costo_str)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Datos de la Instancia", 
        "🎲 Solución y Vecindarios", 
        "🔥 Metaheurística (SA)", 
        "🗺️ Mapa de Rutas"
    ])

# ==========================================
    # PESTAÑA 1: DATOS DE LA INSTANCIA
    # ==========================================
    with tab1:
        st.markdown("### 🖼️ Grafo de la Red Original")
        
        with st.spinner("Generando red base interactiva..."):
            fig_base, _ = viz.dibujar_rutas(
                st.session_state['grafo'], [], data, ruta_idx=None, metaheuristica="Grafo Original"
            )
            st.plotly_chart(fig_base, use_container_width=True, key="plot_instancia_base")
        
        st.markdown("### 📋 Detalles de la Instancia")
        with st.expander("Ver Parámetros Extraídos", expanded=True):
            
            # 1. MOSTRAR LOS DATOS ESCALARES DIRECTO DEL DICCIONARIO
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Nombre:** `{data.get('NOMBRE', 'N/A')}`")
                st.markdown(f"**Comentario:** `{data.get('COMENTARIO', 'N/A')}`")
                st.markdown(f"**Depósito:** `Nodo {data.get('DEPOSITO', 'N/A')}`")
                st.markdown(f"**Vehículos:** `{data.get('VEHICULOS', 'N/A')}`")
                st.markdown(f"**Capacidad:** `{data.get('CAPACIDAD', 'N/A')}`")
            with col2:
                st.markdown(f"**Vértices (Nodos):** `{data.get('VERTICES', 'N/A')}`")
                st.markdown(f"**Aristas Requeridas:** `{data.get('ARISTAS_REQ', 'N/A')}`")
                st.markdown(f"**Aristas No Requeridas:** `{data.get('ARISTAS_NOREQ', 'N/A')}`")
                st.markdown(f"**Tipo de Costes:** `{data.get('TIPO_COSTES_ARISTAS', 'N/A')}`")
                st.markdown(f"**Coste Total Requerido:** `{data.get('COSTE_TOTAL_REQ', 'N/A')}`")
            
            st.divider()
            
            # 2. MOSTRAR LAS LISTAS COMO TABLAS (DataFrames)
            col_req, col_noreq = st.columns(2)
            with col_req:
                st.markdown("**Aristas Requeridas (Servicio)**")
                if 'LISTA_ARISTAS_REQ' in data:
                    st.dataframe(data['LISTA_ARISTAS_REQ'], use_container_width=True, hide_index=True)
                else:
                    st.info("No hay lista de aristas requeridas.")
                    
            with col_noreq:
                st.markdown("**Aristas No Requeridas (Solo tránsito)**")
                if 'LISTA_ARISTAS_NOREQ' in data:
                    st.dataframe(data['LISTA_ARISTAS_NOREQ'], use_container_width=True, hide_index=True)
                else:
                    st.info("No hay lista de aristas no requeridas.")

# ==========================================
    # PESTAÑA 2: SOLUCIÓN ALEATORIA Y VECINDARIOS
    # ==========================================
    with tab2:
        # --- 1. GENERACIÓN DE SOLUCIÓN INICIAL ---
        st.markdown("### 🌱 Generación de Solución Inicial")
        
        if st.button("🎲 Generar Solución Inicial Aleatoria", use_container_width=True, type="primary"):
            with st.spinner("Calculando rutas aleatorias..."):
                nueva_sol = carp.generar_solucion_inicial_aleatoria(data, st.session_state['distancias'])
                captura = io.StringIO()
                sys.stdout = captura
                _, n_costo, n_txt = carp.calcular_y_mostrar_rutas_compacto(nueva_sol, data, st.session_state['grafo'])
                sys.stdout = sys.__stdout__
                
                # Guardamos la inicial y limpiamos cualquier vecino previo
                st.session_state.update({
                    'sol_inicial': nueva_sol, 
                    'costo_inicial': n_costo,
                    'vecino_actual': None,  
                    'costo_vecino': None,
                    'reporte_vecino': None,
                    'mejor_solucion_global': nueva_sol, 
                    'mejor_costo_global': n_costo,
                    'meta_usada': 'Solución Aleatoria Inicial'
                })
                st.rerun()

        # Solo mostrar el contenido si la solución inicial existe
        if st.session_state.get('sol_inicial') is not None:
            st.info(f"**Costo de la Solución Base:** {st.session_state['costo_inicial']}")
            
            # FORMATO LIMPIO: RUTA 1: ARRAY[0], RUTA 2: ARRAY[1]...
            texto_rutas_base = ""
            for i, ruta in enumerate(st.session_state['sol_inicial']):
                texto_rutas_base += f"**RUTA {i+1}:** `{ruta}`\n\n"
            
            with st.container(border=True):
                st.markdown(texto_rutas_base)
                
            st.divider()
            
            # --- 2. SECUENCIAS DE LA SOLUCIÓN ACTUAL ---
            st.markdown("### 📋 Secuencias de la Solución Inicial")
            with st.spinner("Generando formato de bitácora..."):
                _, texto_leyenda_base = viz.dibujar_rutas(
                    st.session_state['grafo'], 
                    st.session_state['sol_inicial'], 
                    data, 
                    ruta_idx=None, 
                    metaheuristica="Solución Inicial"
                )
                with st.container(border=True):
                    st.markdown(texto_leyenda_base)
            
            st.divider()
            
            # --- 3. DEBUG: APLICACIÓN DE OPERADOR DE VECINDARIO ---
            st.markdown("### 🔍 Exploración Controlada de Vecindarios")
            
            # NUEVO: Explicación integral de las 3 reglas de Factibilidad
            cap_max = data.get('CAPACIDAD', 'N/A')
            vehiculos_max = data.get('VEHICULOS', 'N/A')
            
            st.info(
                f"⚖️ **Reglas de Factibilidad Activas:** El motor rechazará automáticamente el vecino y buscará otro si viola cualquiera de estas restricciones:\n"
                f"1. **Capacidad:** La demanda sumada de la ruta supera el máximo del vehículo (**{cap_max} unidades**).\n"
                f"2. **Flota Disponible:** Se utilizan más vehículos de los permitidos por la instancia (**{vehiculos_max} max**).\n"
                f"3. **Conectividad de Red:** No existe un camino válido en el grafo para viajar entre dos tareas adyacentes o hacia el depósito."
            )
            
            col_op1, col_op2, col_op3, col_op4 = st.columns([1, 1, 1, 2])
            with col_op1: 
                op_manual = st.selectbox("Operador Local:", ["swap", "insertion", "inversion"])
            with col_op2:
                p_inter_val = st.slider(
                    "Prob. Inter-Ruta:", 
                    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                    help="Si el número aleatorio es <= a esto, cruza rutas (Inter). Si es > cambia la misma ruta (Intra)."
                )
            with col_op3:
                max_intentos_val = st.number_input(
                    "Máx. Intentos:", 
                    min_value=1, max_value=1000, value=100, step=10,
                    help="Límite de veces que el motor intentará generar un vecino que cumpla con las reglas de factibilidad."
                )
            with col_op4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⚡ Generar Vecino", use_container_width=True):
                    vecino, c_vec, txt_vec = carp.aplicar_y_evaluar_vecindario(
                        st.session_state['sol_inicial'], data, st.session_state['grafo'], 
                        st.session_state['distancias'], operador=op_manual, 
                        p_inter=p_inter_val, max_intentos=max_intentos_val
                    )
                    
                    st.session_state['vecino_actual'] = vecino
                    st.session_state['costo_vecino'] = c_vec
                    st.session_state['reporte_vecino'] = txt_vec
                    
                    if c_vec < st.session_state['mejor_costo_global']:
                        st.session_state['mejor_solucion_global'] = vecino
                        st.session_state['mejor_costo_global'] = c_vec
                        st.session_state['meta_usada'] = f"Búsqueda Manual ({op_manual.capitalize()})"
                        st.toast("¡Nuevo óptimo global encontrado manualmente!", icon="🎉")

            # --- BEAUTIFY: PANEL DE COMPARACIÓN ---
            if st.session_state.get('vecino_actual') is not None:
                st.markdown("#### ⚖️ Comparativa: Solución Inicial vs Nuevo Vecino")
                
                c_base = st.session_state['costo_inicial']
                c_new = st.session_state['costo_vecino']
                delta = c_new - c_base
                
                # 1. Panel de Métricas de Costo
                with st.container(border=True):
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Costo Solución Inicial", c_base)
                    col_m2.metric("Costo Nuevo Vecino", c_new, delta=int(delta), delta_color="inverse")
                    
                    if delta < 0:
                        col_m3.success("✅ Mejora")
                    elif delta > 0:
                        col_m3.error("❌ Empeora")
                    else:
                        col_m3.info("⏸️ Sin cambio")

                # 2. Análisis Visual de Cambios
                st.markdown("#### 🔄 Análisis Estructural de la Operación")
                
                base_sol = st.session_state['sol_inicial']
                new_sol = st.session_state['vecino_actual']
                
                rutas_cambiadas = []
                for i in range(len(base_sol)):
                    if base_sol[i] != new_sol[i]:
                        rutas_cambiadas.append(i)
                
                if not rutas_cambiadas:
                    # El warning ahora abarca las 3 posibilidades
                    st.warning(f"⚠️ El operador se agotó tras {max_intentos_val} intentos. Todos los cambios resultaron idénticos o violaron las reglas de factibilidad (capacidad, flota o conectividad).")
                else:
                    if len(rutas_cambiadas) == 1:
                        st.info("📌 **Clasificación del Movimiento:** `INTRA-RUTA` (Reordenamiento dentro de un mismo vehículo)")
                    else:
                        st.warning("📌 **Clasificación del Movimiento:** `INTER-RUTA` (Intercambio o transferencia entre vehículos distintos)")
                    
                    for idx in rutas_cambiadas:
                        with st.container(border=True):
                            st.markdown(f"**🚛 Ruta {idx + 1} alterada:**")
                            st.markdown(f"🔴 **Antes:** `{base_sol[idx]}`")
                            st.markdown(f"🟢 **Ahora:** `{new_sol[idx]}`")
                
                # 3. Log Original Oculto
                with st.expander("Ver Log original de ejecución (carp_core)", expanded=False):
                    st.code(st.session_state['reporte_vecino'], language="text")
                        
        else:
            st.warning("👈 Oprime el botón superior '🎲 Generar Solución Inicial Aleatoria' para inicializar los datos.")


    # ==========================================
    # PESTAÑA 3: RECOCIDO SIMULADO (SA)
    # ==========================================
    with tab3:
        if st.session_state.get('sol_inicial') is None:
            st.warning("⚠️ Debes generar una Solución Inicial en la pestaña 'Solución y Vecindarios' antes de usar la Metaheurística.")
        else:
            st.markdown("### Configuración de Parámetros")
            col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
            with col_p1: t_inicial = st.number_input("Temp. Inicial ($T_0$)", value=1000.0, step=100.0)
            with col_p2: alfa = st.number_input("Tasa Enfriamiento ($\\alpha$)", value=0.95, step=0.01, format="%.2f")
            with col_p3: iter_por_t = st.number_input("Iteraciones (Iter/T)", value=100, step=10)
            with col_p4: t_final = st.number_input("Temp. de Paro", value=0.1, step=0.1)
            with col_p5: operador_sa = st.selectbox("Operador de Búsqueda", ["swap", "insertion", "inversion", "mixto"])

            if st.button("🔥 Iniciar Optimización SA", use_container_width=True):
                with st.spinner(f"Ejecutando Recocido Simulado usando: {operador_sa.upper()}..."):
                    mejor_sol, mejor_costo, historial, stats = sa.optimizar(
                        st.session_state['mejor_solucion_global'], data, st.session_state['distancias'],
                        t_inicial, alfa, iter_por_t, t_final, operador_sa
                    )
                    
                    if mejor_costo < st.session_state['mejor_costo_global']:
                        st.session_state['mejor_solucion_global'] = mejor_sol
                        st.session_state['mejor_costo_global'] = mejor_costo
                        st.session_state['meta_usada'] = "Recocido Simulado" 
                    
                    captura = io.StringIO()
                    sys.stdout = captura
                    _, _, txt_mejor = carp.calcular_y_mostrar_rutas_compacto(mejor_sol, data, st.session_state['grafo'])
                    sys.stdout = sys.__stdout__
                    
                    carp.guardar_objeto_automatico(st.session_state['ruta_run'], "SA_mejor_solucion", mejor_sol)
                    carp.guardar_objeto_automatico(st.session_state['ruta_run'], "SA_estadisticas", stats)
                    
                st.success("¡Optimización finalizada con éxito!")
                
                st.markdown("### 🔍 Transparencia del Algoritmo")
                st_col1, st_col2, st_col3, st_col4 = st.columns(4)
                st_col1.metric("Intentos Totales", f"{stats['iteraciones_totales']:,}")
                tasa_factible = (stats['vecinos_factibles'] / stats['iteraciones_totales']) * 100 if stats['iteraciones_totales'] > 0 else 0
                st_col2.metric("Vecinos Factibles", f"{stats['vecinos_factibles']:,}", f"{tasa_factible:.1f}% de éxito")
                st_col3.metric("Movimientos Aceptados", f"{stats['movimientos_aceptados']:,}")
                st_col4.metric("Nuevos Óptimos Encontrados", f"{stats['mejoras_globales']:,}")
                st.divider()
                
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("Costo Antes del SA", st.session_state['costo_inicial'])
                c_res2.metric("Costo Tras SA", mejor_costo, delta=int(mejor_costo - st.session_state['costo_inicial']), delta_color="inverse")
                mejora_pct = ((st.session_state['costo_inicial'] - mejor_costo) / st.session_state['costo_inicial']) * 100
                c_res3.metric("% de Mejora", f"{mejora_pct:.2f}%")

                df_historial = pd.DataFrame(historial, columns=["Costo"])
                st.line_chart(df_historial, use_container_width=True)

    # ==========================================
    # PESTAÑA 4: MAPA DE RUTAS (INTERACTIVO)
    # ==========================================
    with tab4:
        if st.session_state.get('mejor_solucion_global') is None:
            st.warning("⚠️ No hay rutas para mostrar. Genera una Solución Inicial en la pestaña 2 primero.")
        else:
            st.markdown("### 🗺️ Mapa Interactivo de Rutas")
            mejor_solucion = st.session_state['mejor_solucion_global']
            
            captura = io.StringIO()
            sys.stdout = captura
            costos_por_ruta, costo_total_mejor, _ = carp.calcular_y_mostrar_rutas_compacto(mejor_solucion, data, st.session_state['grafo'])
            sys.stdout = sys.__stdout__
            
            opciones = ["Visión General (Todas las Rutas)"] + [f"Ruta {i+1}" for i in range(len(mejor_solucion)) if mejor_solucion[i]]
            vista = st.selectbox("Selecciona qué deseas visualizar:", opciones)
            
            st.divider()
            
            if vista == "Visión General (Todas las Rutas)":
                st.info(f"**Costo Total de la Solución:** {costo_total_mejor}")
                
                with st.spinner("Dibujando el mapa completo..."):
                    nombre_meta = st.session_state.get('meta_usada', 'Solución Inicial')
                    fig, texto_leyenda = viz.dibujar_rutas(
                        st.session_state['grafo'], mejor_solucion, data, 
                        ruta_idx=None, metaheuristica=nombre_meta
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="plot_general")
                    
                    with st.expander("Ver Secuencias de Tareas", expanded=True):
                        st.markdown(texto_leyenda)
                    
            else:
                idx = int(vista.replace("Ruta ", "")) - 1
                cap_max = data.get('CAPACIDAD', 0)
                info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}
                
                demanda_total_ruta = sum(info_tareas[t]['demanda'] for t in mejor_solucion[idx])
                st.info(f"🏁 **Proyección Final de la Ruta {idx + 1}:** Costo Total Estimado = {costos_por_ruta[idx]} | Demanda Total = {demanda_total_ruta} / {cap_max}")
                st.divider()

                @st.fragment
                def reproductor_aislado():
                    G = st.session_state['grafo']
                    movimientos = viz.obtener_secuencia_movimientos(G, mejor_solucion[idx], data.get('DEPOSITO', 1), info_tareas)
                    total_pasos = len(movimientos)
                    
                    clave_paso = f"paso_ruta_{idx}"
                    if clave_paso not in st.session_state:
                        st.session_state[clave_paso] = 0 
                    
                    st.markdown("### ⏯️ Controles de Reproducción")
                    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                    with col_b1:
                        if st.button("⏪ Inicio", use_container_width=True): st.session_state[clave_paso] = 0
                    with col_b2:
                        if st.button("⏮️ Anterior", use_container_width=True): 
                            if st.session_state[clave_paso] > 0: st.session_state[clave_paso] -= 1
                    with col_b3:
                        if st.button("Siguiente ⏭️", use_container_width=True, type="primary"):
                            if st.session_state[clave_paso] < total_pasos: st.session_state[clave_paso] += 1
                    with col_b4:
                        if st.button("Fin ⏩", use_container_width=True): st.session_state[clave_paso] = total_pasos
                            
                    paso_actual = st.session_state[clave_paso]
                    
                    progreso = paso_actual / total_pasos if total_pasos > 0 else 0
                    st.progress(progreso)
                    st.caption(f"Mostrando movimiento **{paso_actual}** de **{total_pasos}**")
                    
                    costo_acumulado = 0
                    demanda_acumulada = 0
                    
                    for i in range(paso_actual):
                        mov = movimientos[i]
                        if mov['tipo'] == 'DH':
                            costo_acumulado += G[mov['u']][mov['v']]['cost']
                        else:
                            tarea_id = mov['tarea']
                            costo_acumulado += info_tareas[tarea_id]['costo']
                            demanda_acumulada += info_tareas[tarea_id]['demanda']
                    
                    col_dyn1, col_dyn2, col_dyn3 = st.columns(3)
                    col_dyn1.metric("💸 Costo Acumulado", costo_acumulado)
                    col_dyn2.metric("📦 Carga en el Camión", f"{demanda_acumulada} / {cap_max}")
                    
                    capacidad_restante = cap_max - demanda_acumulada
                    if capacidad_restante >= 0:
                        col_dyn3.metric("✅ Capacidad Restante", capacidad_restante)
                    else:
                        col_dyn3.metric("⚠️ Exceso de Carga", capacidad_restante)

                    with st.spinner(f"Trazando el recorrido..."):
                        nombre_meta = st.session_state.get('meta_usada', 'Solución Inicial')
                        fig, texto_leyenda = viz.dibujar_rutas(
                            G, mejor_solucion, data, 
                            ruta_idx=idx, metaheuristica=nombre_meta, paso_limite=paso_actual
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_ruta_{idx}")
                    
                    with st.expander("Ver Bitácora de Viaje Detallada", expanded=True):
                        st.markdown(texto_leyenda)

                reproductor_aislado()