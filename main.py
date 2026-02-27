import carp_core as carp

def main():
    # 1. Configurar la ruta de tu instancia
    archivo_instancia = "instancia.dat" # <-- ¡Asegúrate de cambiar esto al nombre de tu archivo .dat!
    
    print("Iniciando flujo CARP...")
    
    try:
        # ==========================================
        # FASE 1: Lectura, Validación e Inicialización
        # ==========================================
        d_noreq = carp.leer_carplib_dat(archivo_instancia)
        val_dict = carp.validate_instance(d_noreq, archivo_instancia)
        
        # Iniciar carpeta de ejecución
        ruta_run, logger = carp.iniciar_ejecucion(d_noreq, base_dir="runs_carp")
        
        # Guardar Raw Dict y Validación
        carp.guardar_objeto_automatico(ruta_run, "01_diccionario_raw", d_noreq)
        carp.guardar_objeto_automatico(ruta_run, "02_validacion", val_dict)
        logger.info("Instancia leída y validada correctamente.")

        # ==========================================
        # FASE 2: Grafo y Matriz de Distancias
        # ==========================================
        print("Generando Grafo y calculando Matriz de Distancias...")
        # Al pasarle 'ruta_run', la imagen .jpg se guardará sola
        grafo_noreq = carp.generar_grafo_limpio(d_noreq, carpeta_salida=ruta_run)
        carp.guardar_objeto_automatico(ruta_run, "03_grafo_objeto", grafo_noreq) # Guarda el .pkl
        
        # Calculamos Dijkstra y lo guardamos (.pkl porque es un dict con float('inf'))
        distancias = carp.calcular_matriz_distancias(grafo_noreq, algoritmo='dijkstra')
        carp.guardar_objeto_automatico(ruta_run, "04_matriz_distancias", distancias)
        logger.info("Grafo y matriz de distancias generados y guardados.")

        # ==========================================
        # FASE 3: Solución Inicial Aleatoria
        # ==========================================
        print("Generando Solución Inicial...")
        init_sol = carp.generar_solucion_inicial_aleatoria(d_noreq, distancias)
        
        # Evaluar solución inicial (imprime en consola y nos devuelve el texto)
        costos, costo_tot, txt_reporte = carp.calcular_y_mostrar_rutas_compacto(init_sol, d_noreq, grafo_noreq)
        
        # Guardar solución en JSON y su reporte en TXT
        carp.guardar_objeto_automatico(ruta_run, "05_solucion_inicial_raw", init_sol)
        carp.guardar_objeto_automatico(ruta_run, "05_reporte_inicial", txt_reporte)
        logger.info(f"Solución inicial factible generada. Costo Total: {costo_tot}")

        # ==========================================
        # FASE 4: Aplicar Vecindario (Mutación)
        # ==========================================
        print("Aplicando operador Swap para generar vecino...")
        vecino, costo_vecino, txt_vecino = carp.aplicar_y_evaluar_vecindario(
            init_sol, d_noreq, grafo_noreq, distancias, operador="swap", p_inter=0.5
        )
        
        # Guardar el vecino generado y su debug log
        carp.guardar_objeto_automatico(ruta_run, "06_vecino_raw", vecino)
        carp.guardar_objeto_automatico(ruta_run, "06_reporte_vecino", txt_vecino)
        logger.info(f"Vecino generado exitosamente. Nuevo Costo: {costo_vecino}")

        # ==========================================
        # FASE 5: Finalizar con Éxito
        # ==========================================
        carp.finalizar_ejecucion(ruta_run, logger, success=True)
        print(f"\n¡Flujo terminado con éxito! Revisa tu nueva carpeta en: {ruta_run}")

    except Exception as e:
        # Si algo falla por conectividad o capacidad, el error se atrapa aquí
        print(f"\n[ERROR CRÍTICO] Ocurrió un fallo en la ejecución: {e}")
        try:
            # Intentamos guardar el error si la carpeta alcanzó a crearse
            carp.finalizar_ejecucion(ruta_run, logger, success=False, reason=str(e))
        except:
            pass

if __name__ == "__main__":
    main()