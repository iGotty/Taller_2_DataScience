"""
Funciones de utilidad para el proyecto HabitAlpes
Predicción de Precios de Apartamentos
Autor: Equipo de Data Science
Fecha: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def obtener_ruta_raiz():
    """Obtiene la ruta raíz del proyecto."""
    return Path(__file__).parent.parent


def obtener_ruta_datos(nombre_archivo='apartamentos.csv'):
    """Obtiene la ruta al archivo de datos."""
    return obtener_ruta_raiz() / 'data' / nombre_archivo


def obtener_ruta_procesados(nombre_archivo):
    """Obtiene la ruta a archivo de datos procesados."""
    return obtener_ruta_raiz() / 'data' / 'processed' / nombre_archivo


def obtener_ruta_resultados(nombre_archivo):
    """Obtiene la ruta a archivo de resultados."""
    ruta_resultados = obtener_ruta_raiz() / 'data' / 'results'
    ruta_resultados.mkdir(parents=True, exist_ok=True)
    return ruta_resultados / nombre_archivo


def obtener_ruta_modelo(nombre_archivo):
    """Obtiene la ruta a archivo de modelo."""
    ruta_modelos = obtener_ruta_raiz() / 'models'
    ruta_modelos.mkdir(parents=True, exist_ok=True)
    return ruta_modelos / nombre_archivo


def obtener_ruta_figura(nombre_archivo):
    """Obtiene la ruta a archivo de figura."""
    ruta_figuras = obtener_ruta_raiz() / 'reports' / 'figures'
    ruta_figuras.mkdir(parents=True, exist_ok=True)
    return ruta_figuras / nombre_archivo


def guardar_figura(nombre_archivo, dpi=300, bbox_inches='tight'):
    """Guarda la figura actual de matplotlib."""
    ruta = obtener_ruta_figura(nombre_archivo)
    plt.savefig(ruta, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figura guardada: {ruta}")


def cargar_datos(ruta=None):
    """Carga el dataset de apartamentos."""
    if ruta is None:
        ruta = obtener_ruta_datos()

    print(f"Cargando datos desde {ruta}...")
    df = pd.read_csv(ruta)
    print(f"Cargados {len(df):,} registros con {len(df.columns)} columnas")
    return df


def resumen_valores_faltantes(df):
    """Crea resumen de valores faltantes en el dataframe."""
    faltantes = pd.DataFrame({
        'columna': df.columns,
        'conteo_faltantes': df.isnull().sum().values,
        'porcentaje_faltantes': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    faltantes = faltantes[faltantes['conteo_faltantes'] > 0].sort_values(
        'porcentaje_faltantes', ascending=False
    )
    return faltantes


def imprimir_encabezado(texto, caracter='='):
    """Imprime un encabezado formateado."""
    print(f"\n{caracter * 80}")
    print(f"  {texto}")
    print(f"{caracter * 80}\n")


def formatear_cop(valor):
    """Formatea valores en Pesos Colombianos (COP)."""
    if pd.isna(valor):
        return 'N/A'
    return f"${valor:,.0f}"


def formatear_porcentaje(valor, decimales=2):
    """Formatea valores porcentuales."""
    if pd.isna(valor):
        return 'N/A'
    return f"{valor:.{decimales}f}%"


def calcular_estadisticas_basicas(serie, nombre='Variable'):
    """Calcula y muestra estadísticas básicas para una serie."""
    stats = {
        'Conteo': len(serie),
        'Faltantes': serie.isnull().sum(),
        'Faltantes %': (serie.isnull().sum() / len(serie) * 100),
        'Media': serie.mean(),
        'Mediana': serie.median(),
        'Desv. Est.': serie.std(),
        'Mínimo': serie.min(),
        'Máximo': serie.max(),
        'Q1': serie.quantile(0.25),
        'Q3': serie.quantile(0.75),
        'IQR': serie.quantile(0.75) - serie.quantile(0.25)
    }

    print(f"\nEstadísticas para {nombre}:")
    print("-" * 40)
    for clave, valor in stats.items():
        if isinstance(valor, float):
            print(f"{clave:15s}: {valor:,.2f}")
        else:
            print(f"{clave:15s}: {valor:,}")

    return stats


def imprimir_metricas_modelo(y_real, y_pred, nombre_modelo='Modelo'):
    """Calcula e imprime métricas de regresión."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

    print(f"\nMétricas de {nombre_modelo}:")
    print("-" * 50)
    print(f"MAE (Error Absoluto Medio):       {formatear_cop(mae)}")
    print(f"RMSE (Raíz Error Cuadrático):     {formatear_cop(rmse)}")
    print(f"R² (Coef. Determinación):          {r2:.4f}")
    print(f"MAPE (Error % Absoluto Medio):     {mape:.2f}%")

    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def guardar_resultados(resultados, nombre_archivo):
    """Guarda resultados a archivo CSV."""
    ruta = obtener_ruta_resultados(nombre_archivo)

    if isinstance(resultados, dict):
        resultados_df = pd.DataFrame([resultados])
    else:
        resultados_df = resultados

    resultados_df.to_csv(ruta, index=False)
    print(f"Resultados guardados: {ruta}")

    return ruta


if __name__ == '__main__':
    print("Módulo de Utilidades de HabitAlpes")
    print(f"Ruta raíz del proyecto: {obtener_ruta_raiz()}")

    # Probar carga de datos
    try:
        df = cargar_datos()
        print(f"\nForma del dataset: {df.shape}")
        print(f"Columnas: {list(df.columns[:5])}...")
    except Exception as e:
        print(f"Error cargando datos: {e}")
