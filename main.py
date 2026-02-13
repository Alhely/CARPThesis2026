"""
Archivo principal para ejecutar el código de CARP (Capacitated Arc Routing Problem)
Autor: Tesis CARP 2026
"""

import sys
import os
from carplib_metaheuristics.modelo import CarpLib


def main():
    """
    Función principal que ejecuta el código CARP
    """
    print("=" * 80)
    print("SISTEMA DE RESOLUCIÓN CARP - CARPThesis2026")
    print("=" * 80)
    
    # Crear instancia de CarpLib
    carp = CarpLib()
    
    # Obtener ruta del archivo de instancia desde argumentos de línea de comandos
    if len(sys.argv) > 1:
        ruta_instancia = sys.argv[1]
    else:
        # Buscar archivos .dat en el directorio actual y subdirectorios
        archivos_dat = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.dat'):
                    archivos_dat.append(os.path.join(root, file))
        
        if archivos_dat:
            # Usar el primer archivo encontrado
            ruta_instancia = archivos_dat[0]
            print(f"\nArchivo de instancia encontrado: {ruta_instancia}")
        else:
            print("\nERROR: No se encontró ningún archivo .dat")
            print("Uso: python main.py <ruta_archivo.dat>")
            print("\nO coloca un archivo .dat en el directorio del proyecto.")
            return
    
    # Verificar que el archivo existe
    if not os.path.exists(ruta_instancia):
        print(f"\nERROR: El archivo {ruta_instancia} no existe.")
        return
    
    # Cargar la instancia
    try:
        # Puedes cambiar "dijkstra" por "floyd-warshall" si prefieres ese algoritmo
        carp.cargar_instancia(ruta_instancia, algoritmo_dist="dijkstra")
        
        print("\n" + "=" * 80)
        print("INSTANCIA CARGADA EXITOSAMENTE")
        print("=" * 80)
        
        # Aquí puedes agregar más código para ejecutar metaheurísticas, etc.
        # Por ejemplo:
        # - Ejecutar algoritmos de solución
        # - Generar reportes
        # - Guardar resultados
        
    except Exception as e:
        print(f"\nERROR al cargar la instancia: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
