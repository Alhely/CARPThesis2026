import os
import tempfile
import streamlit as st
import carp_core as carp

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="CARP Optimizer", page_icon="🚛", layout="wide")

st.title("🚛 Optimizador CARP (Búsqueda por Vecindarios)")
st.markdown("Sube tu instancia, visualiza la red y aplica operadores de vecindario en tiempo real.")

# ==========================================
# BARRA LATERAL (SIDEBAR) - CONTROLES
# ==========================================
st.sidebar.header("1. Cargar Instancia")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo .dat", type=["dat"])

if uploaded_file is not None:
    # 1. Guardar el archivo subido temporalmente para que carp_core pueda leerlo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    st.sidebar.success("Archivo cargado correctamente.")

    # Botón para iniciar todo el proceso
    if st.sidebar.button("🚀 Inicializar y Generar Solución"):
        with st.spinner("Procesando instancia y generando grafo..."):
            # Leer datos e iniciar carpeta
            d_noreq = carp.leer_carplib_dat(temp_path)
            ruta_run, logger = carp.iniciar_ejecucion(d_noreq, base_dir="runs_carp_ui")
            
            # Guardar en la "memoria" de Streamlit para no perderlos al hacer clics
            st.session_state['d_noreq'] = d_noreq
            st.session_state['ruta_run'] = ruta_run
            st.session_state['logger'] = logger
            
            # Generar Grafo y Matriz
            grafo = carp.generar_grafo_limpio(d_noreq, carpeta_salida=ruta_run)
            distancias = carp.calcular_matriz_distancias(grafo)
            
            st.session_state['grafo'] = grafo
            st.session_state['distancias'] = distancias
            
            # Generar solución inicial
            init_sol = carp.generar_solucion_inicial_aleatoria(d_noreq, distancias)
            costos, costo_tot, txt_reporte = carp.calcular_y_mostrar_rutas_compacto(init_sol, d_noreq, grafo)
            
            st.session_state['solucion_actual'] = init_sol
            st.session_state['costo_actual'] = costo_tot
            st.session_state['reporte_actual'] = txt_reporte

# ==========================================
# ÁREA PRINCIPAL (MAIN) - VISUALIZACIÓN
# ==========================================

# Solo mostrar si ya inicializamos los datos
if 'd_noreq' in st.session_state:
    data = st.session_state['d_noreq']
    
    # --- MÉTRICAS ---
    st.header("📊 Detalles de la Instancia")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre", data.get("NOMBRE", "N/A"))
    col2.metric("Vehículos", data.get("VEHICULOS", 0))
    col3.metric("Capacidad", data.get("CAPACIDAD", 0))
    col4.metric("Nodos", data.get("VERTICES", 0))
    
    st.divider()
    
    # --- VISUALIZACIÓN Y REPORTE ---
    col_izq, col_der = st.columns([1, 1])
    
    with col_izq:
        st.subheader("Red del Problema")
        # Mostrar la imagen del grafo que se guardó en la carpeta
        ruta_imagen = os.path.join(st.session_state['ruta_run'], f"{data.get('NOMBRE', 'instancia')}_grafo.jpg")
        if os.path.exists(ruta_imagen):
            st.image(ruta_imagen, use_container_width=True)
            
    with col_der:
        st.subheader("🏆 Solución Actual")
        st.metric("Costo Total", st.session_state['costo_actual'])
        
        # Un "acordeón" (expander) para no saturar la pantalla con texto
        with st.expander("Ver Reporte Detallado de las Rutas", expanded=True):
            st.text(st.session_state['reporte_actual'])

    st.divider()

    # ==========================================
    # SECCIÓN DE VECINDARIOS (MUTACIÓN)
    # ==========================================
    st.header("🧬 Exploración de Vecindarios")
    st.markdown("Prueba los operadores de búsqueda local sobre la solución actual.")
    
    col_op1, col_op2, col_op3 = st.columns([2, 1, 2])
    
    with col_op1:
        operador_elegido = st.selectbox("Selecciona un operador:", ["swap", "insertion", "inversion"])
    
    with col_op2:
        st.markdown("<br>", unsafe_allow_html=True) # Espacio para alinear el botón
        if st.button("🔄 Generar Nuevo Vecino"):
            with st.spinner(f"Aplicando operador {operador_elegido}..."):
                # Llamar a nuestra función de vecindario
                vecino, costo_vecino, txt_vecino = carp.aplicar_y_evaluar_vecindario(
                    st.session_state['solucion_actual'], 
                    st.session_state['d_noreq'], 
                    st.session_state['grafo'], 
                    st.session_state['distancias'], 
                    operador=operador_elegido, 
                    p_inter=0.5
                )
                
                # Si mejoró o cambió, actualizamos la "memoria" para que el nuevo vecino sea el actual
                if vecino != st.session_state['solucion_actual']:
                    st.session_state['solucion_actual'] = vecino
                    st.session_state['costo_actual'] = costo_vecino
                    st.session_state['reporte_actual'] = txt_vecino
                    st.success(f"¡Movimiento factible encontrado! Nuevo costo: {costo_vecino}")
                    st.rerun() # Recarga la página para actualizar los gráficos
                else:
                    st.warning("No se encontró un vecino factible tras 100 intentos.")

    # Mostrar el detalle de la mutación
    with st.expander("Ver Debug (Qué cambió exactamente)", expanded=False):
        st.text(st.session_state.get('reporte_actual', ''))