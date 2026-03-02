import os
import sys
import io
import tempfile
import streamlit as st
import pandas as pd
import carp_core as carp
import viz_routes as viz
import meta_sa as sa
import meta_ts as ts  

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
    # PESTAÑA 3: CENTRO DE METAHEURÍSTICAS
    # ==========================================
    with tab3:
        if st.session_state.get('sol_inicial') is None:
            st.warning("⚠️ Debes generar una Solución Inicial en la pestaña 'Solución y Vecindarios' antes de optimizar.")
        else:
            st.markdown("### ⚙️ Motor de Optimización Avanzada")
            
            # 1. SELECTOR DINÁMICO DE ALGORITMO
            meta_seleccionada = st.selectbox(
                "🧠 Selecciona la Metaheurística a ejecutar:",
                ["Recocido Simulado (SA)", "Búsqueda Tabú (TS)"],
                help="El algoritmo partirá desde la Mejor Solución Global encontrada hasta el momento."
            )
            
            st.divider()
            
            # 2. PANEL DE PARÁMETROS DINÁMICOS
            st.markdown(f"#### 🎛️ Configuración para {meta_seleccionada}")
            
            if meta_seleccionada == "Recocido Simulado (SA)":
                st.info("🔥 **Comportamiento:** Acepta peores soluciones con una probabilidad que disminuye con la temperatura (Enfriamiento). Ideal para salir de óptimos locales suavemente.")
                col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
                with col_p1: t_inicial = st.number_input("Temp. Inicial ($T_0$)", value=1000.0, step=100.0)
                with col_p2: alfa = st.number_input("Enfriamiento ($\\alpha$)", value=0.95, step=0.01, format="%.2f")
                with col_p3: iter_por_t = st.number_input("Iteraciones / T", value=100, step=10)
                with col_p4: t_final = st.number_input("Temp. de Paro", value=0.1, step=0.1)
                with col_p5: operador_meta = st.selectbox("Operador SA", ["swap", "insertion", "inversion", "mixto"], key="op_sa")

            elif meta_seleccionada == "Búsqueda Tabú (TS)":
                st.info("🧠 **Comportamiento:** Explora múltiples candidatos por iteración y toma el mejor. Usa una memoria estricta para no repetir configuraciones recientes (Anti-Ciclos).")
                
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1: tenencia_tabu = st.number_input("Tenencia Tabú (Memoria)", value=15, step=1, help="Cuántas soluciones anteriores se prohíben visitar.")
                with col_t2: max_iteraciones = st.number_input("Iteraciones Máximas", value=100, step=10)
                with col_t3: tam_vecindario = st.number_input("Candidatos por Iteración", value=20, step=5, help="Cuántos vecinos genera y evalúa a la vez.")
                
                # Fila 2 de parámetros de vecindario (Específicos porque la función TS los recibe)
                col_t4, col_t5, col_t6 = st.columns(3)
                with col_t4: operador_meta = st.selectbox("Operador Local", ["swap", "insertion", "inversion", "mixto"], key="op_ts")
                with col_t5: p_inter_meta = st.slider("Prob. Inter-Ruta ($p_{inter}$)", 0.0, 1.0, 0.5, step=0.1, key="p_ts")
                with col_t6: max_int_meta = st.number_input("Máx. Intentos Factibilidad", value=100, step=10, key="int_ts")

            # 3. EJECUCIÓN UNIFICADA
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🚀 Iniciar Optimización Global", use_container_width=True, type="primary"):
                
                with st.spinner(f"Ejecutando {meta_seleccionada}... Esto puede tomar unos segundos."):
                    # Enrutamos la ejecución según lo que eligió el usuario
                    if meta_seleccionada == "Recocido Simulado (SA)":
                        mejor_sol, mejor_costo, historial, stats = sa.optimizar(
                            st.session_state['mejor_solucion_global'], data, st.session_state['distancias'],
                            t_inicial, alfa, iter_por_t, t_final, operador_meta
                        )
                        nombre_algoritmo = "Recocido Simulado"
                        
                    elif meta_seleccionada == "Búsqueda Tabú (TS)":
                        # Llamamos al nuevo motor pasando todos sus argumentos específicos
                        mejor_sol, mejor_costo, historial, stats = ts.optimizar(
                            solucion_inicial=st.session_state['mejor_solucion_global'], 
                            costo_inicial=st.session_state['mejor_costo_global'], 
                            data=data, 
                            grafo=st.session_state['grafo'], 
                            distancias=st.session_state['distancias'], 
                            tenencia_tabu=tenencia_tabu, 
                            max_iteraciones=max_iteraciones, 
                            tam_vecindario=tam_vecindario, 
                            operador=operador_meta, 
                            p_inter=p_inter_meta, 
                            max_intentos_vecino=max_int_meta
                        )
                        nombre_algoritmo = "Búsqueda Tabú"

                    # 4. ACTUALIZACIÓN DEL ESTADO GLOBAL
                    if mejor_costo < st.session_state['mejor_costo_global']:
                        st.session_state['mejor_solucion_global'] = mejor_sol
                        st.session_state['mejor_costo_global'] = mejor_costo
                        st.session_state['meta_usada'] = nombre_algoritmo
                        st.toast(f"¡{nombre_algoritmo} encontró un nuevo récord global!", icon="🏆")
                    
                    # Guardamos resultados en disco para auditoría
                    carp.guardar_objeto_automatico(st.session_state['ruta_run'], f"{nombre_algoritmo.replace(' ', '_')}_mejor", mejor_sol)
                    carp.guardar_objeto_automatico(st.session_state['ruta_run'], f"{nombre_algoritmo.replace(' ', '_')}_stats", stats)
                    
                # 5. BEAUTIFY: PANEL DE RESULTADOS
                st.success(f"¡✅ Optimización finalizada con éxito mediante {meta_seleccionada}!")
                
                st.markdown("### 📊 Panel de Rendimiento del Algoritmo")
                
                # Métricas dinámicas dependiendo de qué stats escupa el algoritmo
                metric_cols = st.columns(4)
                metric_cols[0].metric("Iteraciones Totales", f"{stats.get('iteraciones_totales', 0):,}")
                
                if meta_seleccionada == "Recocido Simulado (SA)":
                    tasa_f = (stats.get('vecinos_factibles', 0) / stats.get('iteraciones_totales', 1)) * 100
                    metric_cols[1].metric("Vecinos Factibles", f"{stats.get('vecinos_factibles', 0):,}", f"{tasa_f:.1f}%")
                    metric_cols[2].metric("Movs. Aceptados", f"{stats.get('movimientos_aceptados', 0):,}")
                elif meta_seleccionada == "Búsqueda Tabú (TS)":
                    metric_cols[1].metric("Candidatos Evaluados", f"{stats.get('candidatos_evaluados', 0):,}")
                    metric_cols[2].metric("Aspiraciones Tabú", f"{stats.get('movimientos_tabu_aspirados', 0):,}", help="Movimientos prohibidos que fueron aceptados por ser récords históricos.")
                
                metric_cols[3].metric("Nuevos Óptimos Globales", f"{stats.get('mejoras_globales', 0):,}")
                
                st.divider()
                
                # Impacto en el costo global
                c_res1, c_res2, c_res3 = st.columns(3)
                costo_arranque = historial[0] if historial else st.session_state['costo_inicial']
                c_res1.metric("Costo Antes de Ejecución", costo_arranque)
                
                delta_ejecucion = int(mejor_costo - costo_arranque)
                c_res2.metric("Costo Tras Ejecución", mejor_costo, delta=delta_ejecucion, delta_color="inverse")
                
                mejora_pct = ((costo_arranque - mejor_costo) / costo_arranque) * 100 if costo_arranque > 0 else 0
                c_res3.metric("% de Mejora en esta ronda", f"{mejora_pct:.2f}%")

                # Gráfica de convergencia interactiva
                st.markdown("#### 📉 Curva de Convergencia (Costo vs Iteración)")
                df_historial = pd.DataFrame(historial, columns=["Costo Récord"])
                st.line_chart(df_historial, use_container_width=True)   


# ==========================================
    # PESTAÑA 4: MAPA DE RUTAS (INTERACTIVO)
    # ==========================================
    with tab4:
        if st.session_state.get('mejor_solucion_global') is None:
            st.warning("⚠️ No hay rutas para mostrar. Genera una Solución Inicial primero.")
        else:
            st.markdown("### 🗺️ Visualizador de Rutas")
            mejor_solucion = st.session_state['mejor_solucion_global']
            
            captura = io.StringIO()
            sys.stdout = captura
            costos_por_ruta, costo_total_mejor, _ = carp.calcular_y_mostrar_rutas_compacto(mejor_solucion, data, st.session_state['grafo'])
            sys.stdout = sys.__stdout__
            
            opciones = ["Visión General (Todas las Rutas)"] + [f"Ruta {i+1}" for i in range(len(mejor_solucion)) if mejor_solucion[i]]
            vista = st.selectbox("Selecciona vista:", opciones, label_visibility="collapsed") 
            
            if vista == "Visión General (Todas las Rutas)":
                st.info(f"**Costo Total de la Solución:** {costo_total_mejor}")
                
                with st.spinner("Dibujando mapa completo..."):
                    nombre_meta = st.session_state.get('meta_usada', 'Solución Inicial')
                    fig, texto_leyenda = viz.dibujar_rutas(
                        st.session_state['grafo'], mejor_solucion, data, 
                        ruta_idx=None, metaheuristica=nombre_meta
                    )
                    st.plotly_chart(fig, use_container_width=True, key="plot_general")
                    
                    with st.expander("Ver Secuencias de Tareas", expanded=False): 
                        st.markdown(texto_leyenda)
                    
            else:
                idx = int(vista.replace("Ruta ", "")) - 1
                cap_max = data.get('CAPACIDAD', 0)
                info_tareas = {t['tarea']: t for t in data.get('LISTA_ARISTAS_REQ', [])}

                # ---------------------------------------------------------
                # MODO CINE (Ventana Emergente Gigante con diseño lateral)
                # ---------------------------------------------------------
                @st.dialog(f"🎬 MODO CINE: Explorando Ruta {idx + 1}", width="large")
                def reproductor_fullscreen():
                    G = st.session_state['grafo']
                    movimientos = viz.obtener_secuencia_movimientos(G, mejor_solucion[idx], data.get('DEPOSITO', 1), info_tareas)
                    total_pasos = len(movimientos)
                    
                    clave_paso = f"paso_ruta_fs_{idx}"
                    if clave_paso not in st.session_state:
                        st.session_state[clave_paso] = 0 
                    
                    # Diseño 70/30 en Modo Cine
                    col_mapa_fs, col_ctrl_fs = st.columns([7, 3])
                    
                    with col_ctrl_fs:
                        st.markdown("#### 🎛️ Controles")
                        
                        # Cuadrícula 2x2 para botones
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("⏪ Inicio", use_container_width=True, key="btn_ini_fs"): st.session_state[clave_paso] = 0
                            if st.button("⏮️ Ant.", use_container_width=True, key="btn_ant_fs"): 
                                if st.session_state[clave_paso] > 0: st.session_state[clave_paso] -= 1
                        with c2:
                            if st.button("Fin ⏩", use_container_width=True, key="btn_fin_fs"): st.session_state[clave_paso] = total_pasos
                            if st.button("Siguiente ⏭️", use_container_width=True, type="primary", key="btn_sig_fs"):
                                if st.session_state[clave_paso] < total_pasos: st.session_state[clave_paso] += 1
                                
                        paso_actual = st.session_state[clave_paso]
                        st.progress(paso_actual / total_pasos if total_pasos > 0 else 0)
                        st.caption(f"Paso **{paso_actual}** de **{total_pasos}**")
                        
                        # Cálculos dinámicos
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
                        
                        # Métricas apiladas lateralmente
                        st.divider()
                        st.metric("💵 Costo Acumulado", int(costo_acumulado))
                        st.metric("📦 Carga Actual", f"{demanda_acumulada} / {cap_max}")
                        cap_restante = cap_max - demanda_acumulada
                        st.metric("✅ Cap. Restante" if cap_restante >= 0 else "⚠️ Exceso", cap_restante)

                    with col_mapa_fs:
                        with st.spinner("Trazando..."):
                            nombre_meta = st.session_state.get('meta_usada', 'Solución Inicial')
                            fig_fs, _ = viz.dibujar_rutas(
                                G, mejor_solucion, data, 
                                ruta_idx=idx, metaheuristica=nombre_meta, paso_limite=paso_actual
                            )
                            fig_fs.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0)) # Ajuste de márgenes
                            st.plotly_chart(fig_fs, use_container_width=True, key=f"plot_ruta_fs_{idx}")

                # ---------------------------------------------------------
                # VISTA ESTÁNDAR INTEGRADA (También con diseño lateral)
                # ---------------------------------------------------------
                @st.fragment
                def reproductor_integrado():
                    G = st.session_state['grafo']
                    movimientos = viz.obtener_secuencia_movimientos(G, mejor_solucion[idx], data.get('DEPOSITO', 1), info_tareas)
                    total_pasos = len(movimientos)
                    
                    clave_paso = f"paso_ruta_{idx}"
                    if clave_paso not in st.session_state:
                        st.session_state[clave_paso] = 0 
                        
                    # Diseño 70/30 en Pestaña Normal
                    col_mapa, col_ctrl = st.columns([7, 3])
                    
                    with col_ctrl:
                        st.markdown("#### 🎛️ Controles")
                        
                        # Cuadrícula 2x2
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("⏪ Inicio", use_container_width=True, key=f"btn_ini_{idx}"): st.session_state[clave_paso] = 0
                            if st.button("⏮️ Ant.", use_container_width=True, key=f"btn_ant_{idx}"): 
                                if st.session_state[clave_paso] > 0: st.session_state[clave_paso] -= 1
                        with c2:
                            if st.button("Fin ⏩", use_container_width=True, key=f"btn_fin_{idx}"): st.session_state[clave_paso] = total_pasos
                            if st.button("Siguiente ⏭️", use_container_width=True, type="primary", key=f"btn_sig_{idx}"):
                                if st.session_state[clave_paso] < total_pasos: st.session_state[clave_paso] += 1
                                
                        paso_actual = st.session_state[clave_paso]
                        st.progress(paso_actual / total_pasos if total_pasos > 0 else 0)
                        st.caption(f"Paso **{paso_actual}** de **{total_pasos}**")
                        
                        # Cálculos dinámicos
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
                                
                        st.divider()
                        
                        # Métricas más compactas para no empujar la pantalla
                        mc1, mc2 = st.columns(2)
                        mc1.metric("💵 Costo", int(costo_acumulado))
                        mc2.metric("📦 Carga", f"{demanda_acumulada}/{cap_max}")
                        
                        st.divider()
                        
                        # El botón de Modo Cine al final del panel de control
                        if st.button("🎬 Abrir Modo Cine", use_container_width=True, type="secondary", key=f"btn_cine_{idx}"):
                            reproductor_fullscreen()

                    with col_mapa:
                        with st.spinner("Trazando..."):
                            nombre_meta = st.session_state.get('meta_usada', 'Solución Inicial')
                            fig, texto_leyenda = viz.dibujar_rutas(
                                G, mejor_solucion, data, 
                                ruta_idx=idx, metaheuristica=nombre_meta, paso_limite=paso_actual
                            )
                            fig.update_layout(height=550, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig, use_container_width=True, key=f"plot_ruta_{idx}")
                    
                    with st.expander("📜 Ver Bitácora Detallada del Viaje", expanded=False):
                        st.markdown(texto_leyenda)

                reproductor_integrado()