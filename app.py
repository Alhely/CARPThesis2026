import os
import sys
import io
import tempfile
import time
import streamlit as st
import pandas as pd
import numpy as np
import carp_core as carp
import meta_sa as sa
import meta_ts as ts
import meta_cs as cs
import viz_routes as viz

st.set_page_config(page_title="CARP Optimizer", page_icon="🚛", layout="wide")
st.title("🚛 Optimizador CARP")

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def obtener_benchmarks_y_gap(nombre_instancia, df_benchmarks, costo_actual):
    """Busca la instancia en el CSV y calcula el GAP devolviendo también los límites."""
    if df_benchmarks is None or costo_actual is None:
        return None, None, None, None, None, None
        
    fila = df_benchmarks[df_benchmarks['Instances'].str.strip().str.lower() == str(nombre_instancia).strip().lower()]
    
    if fila.empty:
        return None, None, None, None, None, None
        
    def a_numero(val):
        try:
            if isinstance(val, str):
                val = val.replace(',', '').strip()
            return float(val)
        except (ValueError, TypeError):
            return np.nan 

    bks = a_numero(fila['BKS'].values[0]) if 'BKS' in fila.columns else np.nan
    blb = a_numero(fila['BLB'].values[0]) if 'BLB' in fila.columns else np.nan
    bub = a_numero(fila['BUB'].values[0]) if 'BUB' in fila.columns else np.nan
    
    gap, ref_nombre, ref_valor = None, None, None
    
    # Jerarquía: BKS -> BLB -> BUB
    if pd.notna(bks) and bks > 0:
        gap = ((costo_actual - bks) / bks) * 100
        ref_nombre, ref_valor = "BKS", bks
    elif pd.notna(blb) and blb > 0:
        gap = ((costo_actual - blb) / blb) * 100
        ref_nombre, ref_valor = "BLB", blb
    elif pd.notna(bub) and bub > 0:
        gap = ((costo_actual - bub) / bub) * 100
        ref_nombre, ref_valor = "BUB", bub
        
    return gap, ref_nombre, ref_valor, bks, blb, bub

# ==========================================
# BARRA LATERAL - INICIALIZACIÓN
# ==========================================
st.sidebar.header("1. Cargar Datos")
uploaded_file = st.sidebar.file_uploader("📁 Archivo de Instancia (.dat)", type=["dat"])
uploaded_csv = st.sidebar.file_uploader("📊 Benchmarks Opcional (.csv)", type=["csv"], help="CSV con columnas: Instances, BKS, BLB, BUB")

algoritmo_dist = st.sidebar.selectbox(
    "Algoritmo de Caminos Cortos:", 
    ["dijkstra", "floyd-warshall", "bellman-ford"],
    format_func=lambda x: x.replace("-", " ").title()
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    if st.sidebar.button("🚀 Inicializar Entorno", use_container_width=True, type="primary"):
        with st.spinner(f"Calculando matriz de distancias con {algoritmo_dist.title()}..."):
            d_noreq = carp.leer_carplib_dat(temp_path)
            ruta_run, logger = carp.iniciar_ejecucion(d_noreq, base_dir="runs_carp_ui")
            
            grafo = carp.generar_grafo_limpio(d_noreq, carpeta_salida=ruta_run)
            distancias = carp.calcular_matriz_distancias(grafo, algoritmo=algoritmo_dist)
            
            df_bks = None
            if uploaded_csv is not None:
                df_bks = pd.read_csv(uploaded_csv)
            
            st.session_state.update({
                'd_noreq': d_noreq, 'ruta_run': ruta_run, 'logger': logger,
                'grafo': grafo, 'distancias': distancias,
                'df_benchmarks': df_bks,
                'sol_inicial': None, 'costo_inicial': None,
                'mejor_solucion_global': None, 'mejor_costo_global': None,
                'meta_usada': 'Ninguna',
                'solucion_actual': None, 'costo_actual': None, 'reporte_actual': ""
            })
        st.sidebar.success("¡Entorno y Grafo listos!")

# ==========================================
# ÁREA PRINCIPAL - MÉTRICAS Y PESTAÑAS
# ==========================================
if 'd_noreq' in st.session_state:
    data = st.session_state['d_noreq']
    
    # --- MÉTRICAS GLOBALES ---
    st.markdown("### 📊 Panel General de la Instancia")
    
    # FILA 1: Datos físicos y Costo de tu algoritmo
    c1, c2, c3, c4 = st.columns(4)
    
    nombre_inst = data.get("NOMBRE", "N/A")
    c1.metric("Instancia", nombre_inst)
    c2.metric("Vehículos", data.get("VEHICULOS", 0))
    c3.metric("Capacidad Máx.", data.get("CAPACIDAD", 0))
    
    costo_actual = st.session_state.get('mejor_costo_global')
    costo_str = int(costo_actual) if costo_actual is not None else "---"
    c4.metric("🏆 Mejor Costo Actual", costo_str)
    
    # FILA 2: Benchmarks y Cálculo de GAP
    st.markdown("#### 📖 Comparativa vs Literatura (Benchmarks)")
    b1, b2, b3, b4 = st.columns(4)
    
    gap_val, ref_nombre, ref_valor, bks, blb, bub = obtener_benchmarks_y_gap(
        nombre_instancia=nombre_inst, 
        df_benchmarks=st.session_state.get('df_benchmarks'), 
        costo_actual=costo_actual
    )
    
    def formato_bench(val):
        return int(val) if pd.notna(val) else "N/D"

    if st.session_state.get('df_benchmarks') is not None:
        b1.metric("BKS (Mejor Conocido)", formato_bench(bks))
        b2.metric("BLB (Límite Inferior)", formato_bench(blb))
        b3.metric("BUB (Límite Superior)", formato_bench(bub))
        
        if gap_val is not None:
            label_gap = f"🎯 GAP (vs {ref_nombre})"
            if gap_val <= 0.0:
                b4.metric(label_gap, f"{gap_val:.2f}%", "¡Óptimo o Mejor! 🌟")
            else:
                b4.metric(label_gap, f"+{gap_val:.2f}%", delta_color="inverse")
        else:
            b4.metric("🎯 GAP", "Faltan valores base")
    else:
        st.info("💡 Sube un archivo CSV de Benchmarks en la barra lateral para ver límites y calcular el GAP.")
        
    st.divider()

    # --- ESTRUCTURA DE PESTAÑAS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Datos de la Instancia", 
        "🎲 Solución y Vecindarios", 
        "🔥 Centro de Metaheurísticas", 
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
        st.markdown("### 🌱 Generación de Solución Inicial")
        
        if st.button("🎲 Generar Solución Inicial Aleatoria", use_container_width=True, type="primary"):
            with st.spinner("Calculando rutas aleatorias..."):
                nueva_sol = carp.generar_solucion_inicial_aleatoria(data, st.session_state['distancias'])
                captura = io.StringIO()
                sys.stdout = captura
                _, n_costo, n_txt = carp.calcular_y_mostrar_rutas_compacto(nueva_sol, data, st.session_state['grafo'])
                sys.stdout = sys.__stdout__
                
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

        if st.session_state.get('sol_inicial') is not None:
            st.info(f"**Costo de la Solución Base:** {st.session_state['costo_inicial']}")
            
            texto_rutas_base = ""
            for i, ruta in enumerate(st.session_state['sol_inicial']):
                texto_rutas_base += f"**RUTA {i+1}:** `{ruta}`\n\n"
            
            with st.container(border=True):
                st.markdown(texto_rutas_base)
                
            st.divider()
            
            st.markdown("### 📋 Secuencias de la Solución Inicial")
            with st.spinner("Generando formato de bitácora..."):
                _, texto_leyenda_base = viz.dibujar_rutas(
                    st.session_state['grafo'], st.session_state['sol_inicial'], data, 
                    ruta_idx=None, metaheuristica="Solución Inicial"
                )
                with st.container(border=True):
                    st.markdown(texto_leyenda_base)
            
            st.divider()
            
            st.markdown("### 🔍 Exploración Controlada de Vecindarios")
            
            cap_max = data.get('CAPACIDAD', 'N/A')
            vehiculos_max = data.get('VEHICULOS', 'N/A')
            
            st.info(
                f"⚖️ **Reglas de Factibilidad Activas:** El motor rechazará automáticamente el vecino si:\n"
                f"1. **Capacidad:** La demanda sumada supera el máximo del vehículo (**{cap_max} unidades**).\n"
                f"2. **Flota Disponible:** Se utilizan más vehículos de los permitidos (**{vehiculos_max} max**).\n"
                f"3. **Conectividad:** No existe un camino válido en el grafo para viajar entre dos tareas."
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
                    help="Límite de intentos para generar un vecino factible."
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

            if st.session_state.get('vecino_actual') is not None:
                st.markdown("#### ⚖️ Comparativa: Solución Inicial vs Nuevo Vecino")
                
                c_base = st.session_state['costo_inicial']
                c_new = st.session_state['costo_vecino']
                delta = c_new - c_base
                
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

                st.markdown("#### 🔄 Análisis Estructural de la Operación")
                
                base_sol = st.session_state['sol_inicial']
                new_sol = st.session_state['vecino_actual']
                
                rutas_cambiadas = []
                for i in range(len(base_sol)):
                    if base_sol[i] != new_sol[i]:
                        rutas_cambiadas.append(i)
                
                if not rutas_cambiadas:
                    st.warning(f"⚠️ El operador se agotó tras {max_intentos_val} intentos. Cambios idénticos o violaron factibilidad.")
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
            
            meta_seleccionada = st.selectbox(
                "🧠 Selecciona la Metaheurística a ejecutar:",
                ["Recocido Simulado (SA)", "Búsqueda Tabú (TS)", "Cuckoo Search (CS)"],
                help="El algoritmo partirá desde la Mejor Solución Global encontrada hasta el momento."
            )
            
            st.divider()
            
            st.markdown(f"#### 🎛️ Configuración para {meta_seleccionada}")
            
            if meta_seleccionada == "Recocido Simulado (SA)":
                st.info("🔥 **Comportamiento:** Acepta peores soluciones con una probabilidad térmica. Búsqueda de un solo agente.")
                col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
                with col_p1: t_inicial = st.number_input("Temp. Inicial ($T_0$)", value=1000.0, step=100.0)
                with col_p2: alfa = st.number_input("Enfriamiento ($\\alpha$)", value=0.95, step=0.01, format="%.2f")
                with col_p3: iter_por_t = st.number_input("Iteraciones / T", value=100, step=10)
                with col_p4: t_final = st.number_input("Temp. de Paro", value=0.1, step=0.1)
                with col_p5: operador_meta = st.selectbox("Operador SA", ["swap", "insertion", "inversion", "mixto"], key="op_sa")

            elif meta_seleccionada == "Búsqueda Tabú (TS)":
                st.info("🧠 **Comportamiento:** Explora múltiples vecinos a la vez y usa memoria estricta para evitar ciclos.")
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1: tenencia_tabu = st.number_input("Tenencia Tabú (Memoria)", value=15, step=1)
                with col_t2: max_iteraciones = st.number_input("Iteraciones Máximas", value=100, step=10, key="it_ts")
                with col_t3: tam_vecindario = st.number_input("Candidatos por Iteración", value=20, step=5)
                col_t4, col_t5, col_t6 = st.columns(3)
                with col_t4: operador_meta = st.selectbox("Operador Local", ["swap", "insertion", "inversion", "mixto"], key="op_ts")
                with col_t5: p_inter_meta = st.slider("Prob. Inter-Ruta ($p_{inter}$)", 0.0, 1.0, 0.5, step=0.1, key="p_ts")
                with col_t6: max_int_meta = st.number_input("Máx. Intentos Factibilidad", value=100, step=10, key="int_ts")

            elif meta_seleccionada == "Cuckoo Search (CS)":
                st.info("🪺 **Comportamiento:** Basado en población. Usa Vuelos de Lévy y destruye soluciones periódicamente.")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1: num_nidos = st.number_input("Cantidad de Nidos (Población)", value=15, step=5)
                with col_c2: p_a = st.slider("Tasa de Abandono ($p_a$)", 0.0, 1.0, 0.25, step=0.05)
                with col_c3: max_iteraciones = st.number_input("Generaciones Máximas", value=100, step=10, key="it_cs")
                col_c4, col_c5, col_c6 = st.columns(3)
                with col_c4: operador_meta = st.selectbox("Operador Vuelo Lévy", ["swap", "insertion", "inversion", "mixto"], key="op_cs")
                with col_c5: p_inter_meta = st.slider("Prob. Inter-Ruta ($p_{inter}$)", 0.0, 1.0, 0.5, step=0.1, key="p_cs")
                with col_c6: max_int_meta = st.number_input("Máx. Intentos Factibilidad", value=100, step=10, key="int_cs")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🚀 Iniciar Optimización Global", use_container_width=True, type="primary"):
                
                with st.spinner(f"Ejecutando {meta_seleccionada}... Esto puede tomar un momento."):
                    t_inicio_cpu = time.process_time()
                    
                    if meta_seleccionada == "Recocido Simulado (SA)":
                        mejor_sol, mejor_costo, historial, stats = sa.optimizar(
                            st.session_state['mejor_solucion_global'], data, st.session_state['distancias'],
                            t_inicial, alfa, iter_por_t, t_final, operador_meta
                        )
                        nombre_algoritmo = "Recocido Simulado"
                        
                    elif meta_seleccionada == "Búsqueda Tabú (TS)":
                        mejor_sol, mejor_costo, historial, stats = ts.optimizar(
                            st.session_state['mejor_solucion_global'], st.session_state['mejor_costo_global'], 
                            data, st.session_state['grafo'], st.session_state['distancias'], 
                            tenencia_tabu, max_iteraciones, tam_vecindario, 
                            operador_meta, p_inter_meta, max_int_meta
                        )
                        nombre_algoritmo = "Búsqueda Tabú"
                        
                    elif meta_seleccionada == "Cuckoo Search (CS)":
                        mejor_sol, mejor_costo, historial, stats = cs.optimizar(
                            st.session_state['mejor_solucion_global'], st.session_state['mejor_costo_global'], 
                            data, st.session_state['grafo'], st.session_state['distancias'], 
                            num_nidos, p_a, max_iteraciones, 
                            operador_meta, p_inter_meta, max_int_meta
                        )
                        nombre_algoritmo = "Cuckoo Search"

                    t_fin_cpu = time.process_time()
                    tiempo_cpu = t_fin_cpu - t_inicio_cpu

                    if mejor_costo < st.session_state['mejor_costo_global']:
                        st.session_state['mejor_solucion_global'] = mejor_sol
                        st.session_state['mejor_costo_global'] = mejor_costo
                        st.session_state['meta_usada'] = nombre_algoritmo
                        st.toast(f"¡{nombre_algoritmo} encontró un récord en {tiempo_cpu:.2f}s!", icon="🏆")
                    
                    stats['tiempo_cpu_segundos'] = tiempo_cpu
                    carp.guardar_objeto_automatico(st.session_state['ruta_run'], f"{nombre_algoritmo.replace(' ', '_')}_mejor", mejor_sol)
                    carp.guardar_objeto_automatico(st.session_state['ruta_run'], f"{nombre_algoritmo.replace(' ', '_')}_stats", stats)
                    
                st.success(f"¡✅ Optimización finalizada con éxito mediante {meta_seleccionada}!")
                st.markdown("### 📊 Panel de Rendimiento del Algoritmo")
                
                metric_cols = st.columns(4)
                metric_cols[0].metric("Iteraciones/Generaciones", f"{stats.get('iteraciones_totales', 0):,}")
                
                if meta_seleccionada == "Recocido Simulado (SA)":
                    tasa_f = (stats.get('vecinos_factibles', 0) / stats.get('iteraciones_totales', 1)) * 100
                    metric_cols[1].metric("Vecinos Factibles", f"{stats.get('vecinos_factibles', 0):,}", f"{tasa_f:.1f}%")
                    metric_cols[2].metric("Movs. Aceptados", f"{stats.get('movimientos_aceptados', 0):,}")
                elif meta_seleccionada == "Búsqueda Tabú (TS)":
                    metric_cols[1].metric("Candidatos Evaluados", f"{stats.get('candidatos_evaluados', 0):,}")
                    metric_cols[2].metric("Aspiraciones Tabú", f"{stats.get('movimientos_tabu_aspirados', 0):,}")
                elif meta_seleccionada == "Cuckoo Search (CS)":
                    metric_cols[1].metric("Nidos Abandonados", f"{stats.get('nidos_abandonados', 0):,}")
                    metric_cols[2].metric("Cuckoos Exitosos", f"{stats.get('huevos_cuckoo_exitosos', 0):,}")
                
                metric_cols[3].metric("Nuevos Óptimos Globales", f"{stats.get('mejoras_globales', 0):,}")
                
                st.divider()
                
                c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                costo_arranque = historial[0] if historial else st.session_state['costo_inicial']
                c_res1.metric("Costo Antes de Ejecución", costo_arranque)
                
                delta_ejecucion = int(mejor_costo - costo_arranque)
                c_res2.metric("Costo Tras Ejecución", mejor_costo, delta=delta_ejecucion, delta_color="inverse")
                
                mejora_pct = ((costo_arranque - mejor_costo) / costo_arranque) * 100 if costo_arranque > 0 else 0
                c_res3.metric("% de Mejora (Ronda)", f"{mejora_pct:.2f}%")
                c_res4.metric("⏱️ Tiempo de CPU", f"{tiempo_cpu:.3f} s")

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

                @st.dialog(f"🎬 MODO CINE: Explorando Ruta {idx + 1}", width="large")
                def reproductor_fullscreen():
                    G = st.session_state['grafo']
                    movimientos = viz.obtener_secuencia_movimientos(G, mejor_solucion[idx], data.get('DEPOSITO', 1), info_tareas)
                    total_pasos = len(movimientos)
                    
                    clave_paso = f"paso_ruta_fs_{idx}"
                    if clave_paso not in st.session_state:
                        st.session_state[clave_paso] = 0 
                    
                    col_mapa_fs, col_ctrl_fs = st.columns([7, 3])
                    
                    with col_ctrl_fs:
                        st.markdown("#### 🎛️ Controles")
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
                            fig_fs.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0))
                            st.plotly_chart(fig_fs, use_container_width=True, key=f"plot_ruta_fs_{idx}")

                @st.fragment
                def reproductor_integrado():
                    G = st.session_state['grafo']
                    movimientos = viz.obtener_secuencia_movimientos(G, mejor_solucion[idx], data.get('DEPOSITO', 1), info_tareas)
                    total_pasos = len(movimientos)
                    
                    clave_paso = f"paso_ruta_{idx}"
                    if clave_paso not in st.session_state:
                        st.session_state[clave_paso] = 0 
                        
                    col_mapa, col_ctrl = st.columns([7, 3])
                    
                    with col_ctrl:
                        st.markdown("#### 🎛️ Controles")
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
                        mc1, mc2 = st.columns(2)
                        mc1.metric("💵 Costo", int(costo_acumulado))
                        mc2.metric("📦 Carga", f"{demanda_acumulada}/{cap_max}")
                        
                        st.divider()
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