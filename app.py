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
        with st.spinner("Procesando instancia y distancias..."):
            d_noreq = carp.leer_carplib_dat(temp_path)
            ruta_run, logger = carp.iniciar_ejecucion(d_noreq, base_dir="runs_carp_ui")
            
            grafo = carp.generar_grafo_limpio(d_noreq, carpeta_salida=ruta_run)
            distancias = carp.calcular_matriz_distancias(grafo)
            init_sol = carp.generar_solucion_inicial_aleatoria(d_noreq, distancias)
            
            captura = io.StringIO()
            sys.stdout = captura
            costos_init, costo_tot, txt_reporte = carp.calcular_y_mostrar_rutas_compacto(init_sol, d_noreq, grafo)
            sys.stdout = sys.__stdout__
            
            st.session_state.update({
                'd_noreq': d_noreq, 'ruta_run': ruta_run, 'logger': logger,
                'grafo': grafo, 'distancias': distancias,
                'sol_inicial': init_sol, 'costo_inicial': costo_tot,
                'mejor_solucion_global': init_sol, 'mejor_costo_global': costo_tot,
                'solucion_actual': init_sol, 'costo_actual': costo_tot, 'reporte_actual': txt_reporte
            })
        st.sidebar.success("¡Entorno listo!")

# ==========================================
# ÁREA PRINCIPAL - PESTAÑAS
# ==========================================
if 'd_noreq' in st.session_state:
    data = st.session_state['d_noreq']
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instancia", data.get("NOMBRE", "N/A"))
    c2.metric("Vehículos", data.get("VEHICULOS", 0))
    c3.metric("Capacidad Máxima", data.get("CAPACIDAD", 0))
    c4.metric("🏆 Mejor Costo Actual", st.session_state['mejor_costo_global'])
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🧬 Modo Manual", "🔥 Metaheurística (SA)", "🗺️ Mapa de Rutas"])

    # --- PESTAÑA 1: MODO MANUAL ---
    with tab1:
        col_izq, col_der = st.columns([1, 1])
        with col_izq:
            ruta_img = os.path.join(st.session_state['ruta_run'], f"{data.get('NOMBRE', 'instancia')}_grafo.jpg")
            if os.path.exists(ruta_img): st.image(ruta_img, use_container_width=True)
                
        with col_der:
            st.metric("Costo de Solución en Debug", st.session_state['costo_actual'])
            with st.expander("Ver Reporte Detallado", expanded=True):
                st.text(st.session_state['reporte_actual'])
                
        st.markdown("### Operadores Locales")
        co1, co2 = st.columns([1, 2])
        with co1: op_manual = st.selectbox("Operador:", ["swap", "insertion", "inversion"])
        with co2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Aplicar 1 vez"):
                vecino, c_vec, txt_vec = carp.aplicar_y_evaluar_vecindario(
                    st.session_state['solucion_actual'], data, st.session_state['grafo'], 
                    st.session_state['distancias'], operador=op_manual, max_intentos=50
                )
                if vecino != st.session_state['solucion_actual']:
                    if c_vec < st.session_state['mejor_costo_global']:
                         st.session_state['mejor_solucion_global'] = vecino
                         st.session_state['mejor_costo_global'] = c_vec
                         st.toast("¡Nuevo óptimo global encontrado manualmente!", icon="🎉")

                    st.session_state.update({'solucion_actual': vecino, 'costo_actual': c_vec, 'reporte_actual': txt_vec})
                    st.rerun()

    # --- PESTAÑA 2: RECOCIDO SIMULADO ---
    with tab2:
        st.markdown("### Configuración de Parámetros")
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        with col_p1: t_inicial = st.number_input("Temp. Inicial ($T_0$)", value=1000.0, step=100.0)
        with col_p2: alfa = st.number_input("Tasa Enfriamiento ($\\alpha$)", value=0.95, step=0.01, format="%.2f")
        with col_p3: iter_por_t = st.number_input("Tamaño Vecindario (Iter/T)", value=100, step=10)
        with col_p4: t_final = st.number_input("Temp. de Paro", value=0.1, step=0.1)
        with col_p5: operador_sa = st.selectbox("Operador de Búsqueda", ["swap", "insertion", "inversion", "mixto"])

        if st.button("🔥 Iniciar Optimización", use_container_width=True):
            with st.spinner(f"Ejecutando Recocido Simulado usando el operador: {operador_sa.upper()}..."):
                mejor_sol, mejor_costo, historial, stats = sa.optimizar(
                    st.session_state['mejor_solucion_global'], data, st.session_state['distancias'],
                    t_inicial, alfa, iter_por_t, t_final, operador_sa
                )
                if mejor_costo < st.session_state['mejor_costo_global']:
                    st.session_state['mejor_solucion_global'] = mejor_sol
                    st.session_state['mejor_costo_global'] = mejor_costo
                
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

    # --- PESTAÑA 3: MAPA DE RUTAS (LIMPIA Y ORDENADA) ---
    with tab3:
        st.markdown("### 🗺️ Mapa Interactivo de Rutas")
        mejor_solucion = st.session_state['mejor_solucion_global']
        
        captura = io.StringIO()
        sys.stdout = captura
        costos_por_ruta, costo_total_mejor, _ = carp.calcular_y_mostrar_rutas_compacto(mejor_solucion, data, st.session_state['grafo'])
        sys.stdout = sys.__stdout__
        
        opciones = ["Visión General (Todas las Rutas)"] + [f"Ruta {i+1}" for i in range(len(mejor_solucion)) if mejor_solucion[i]]
        vista = st.selectbox("Selecciona qué deseas visualizar:", opciones)
        
        st.divider()
        
        # Dividimos la pantalla: 70% Gráfica, 30% Cuadro de Texto de la Leyenda
        col_grafica, col_texto = st.columns([7, 3])
        
        if vista == "Visión General (Todas las Rutas)":
            with col_texto:
                st.info(f"**Costo Total:** {costo_total_mejor}")
                
            with st.spinner("Dibujando el mapa..."):
                fig, texto_leyenda = viz.dibujar_rutas(st.session_state['grafo'], mejor_solucion, data, ruta_idx=None)
                with col_grafica:
                    st.plotly_chart(fig, use_container_width=True)
                with col_texto:
                    # El texto se muestra limpio y sin estorbar a la gráfica
                    st.markdown(texto_leyenda)
                
        else:
            idx = int(vista.replace("Ruta ", "")) - 1
            cap_max = data.get('CAPACIDAD', 0)
            info_tareas = {t['tarea']: t['demanda'] for t in data.get('LISTA_ARISTAS_REQ', [])}
            demanda_ruta = sum(info_tareas[t] for t in mejor_solucion[idx])
            
            with col_texto:
                st.info(f"**Costo Ruta:** {costos_por_ruta[idx]}\n\n**Demanda:** {demanda_ruta} / {cap_max}")
                if demanda_ruta > cap_max:
                    st.error("⚠️ Capacidad excedida.")
                else:
                    st.success("✅ Capacidad respetada.")
                    
            with st.spinner(f"Trazando el recorrido de la {vista}..."):
                fig, texto_leyenda = viz.dibujar_rutas(st.session_state['grafo'], mejor_solucion, data, ruta_idx=idx)
                with col_grafica:
                    st.plotly_chart(fig, use_container_width=True)
                with col_texto:
                    # El texto se muestra limpio y sin estorbar a la gráfica
                    st.markdown(texto_leyenda)