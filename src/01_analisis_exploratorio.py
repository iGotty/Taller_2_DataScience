"""
01 - Análisis Exploratorio de Datos (EDA)
Proyecto HabitAlpes - Predicción de Precios de Apartamentos

Este script realiza un análisis exploratorio comprehensivo del dataset de apartamentos
en Bogotá, Colombia. Incluye análisis de estructura, valores faltantes, distribuciones,
correlaciones y patrones geográficos.

Autor: Equipo de Data Science
Fecha: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    cargar_datos, imprimir_encabezado, resumen_valores_faltantes,
    calcular_estadisticas_basicas, guardar_figura, obtener_ruta_resultados,
    formatear_cop
)
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo de gráficos
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def analizar_estructura_datos(df):
    """
    Analiza la estructura básica del dataset.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ESTRUCTURA DEL DATASET", "-")

    print(f"Número de registros: {df.shape[0]:,}")
    print(f"Número de columnas: {df.shape[1]}")
    print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\nNombres de columnas:")
    for i, columna in enumerate(df.columns, 1):
        tipo = str(df[columna].dtype)
        print(f"  {i:2d}. {columna:30s} ({tipo})")

    print(f"\nResumen de tipos de datos:")
    tipos_datos = df.dtypes.value_counts()
    for tipo, conteo in tipos_datos.items():
        print(f"  {str(tipo):15s}: {conteo} columnas")

    return df.dtypes


def analizar_valores_faltantes(df):
    """
    Analiza y visualiza los valores faltantes en el dataset.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE VALORES FALTANTES", "-")

    faltantes = resumen_valores_faltantes(df)

    if len(faltantes) > 0:
        print("Columnas con valores faltantes:\n")
        print(faltantes.to_string(index=False))

        # Visualización de valores faltantes
        fig, ax = plt.subplots(figsize=(12, max(8, len(faltantes) * 0.3)))

        top_faltantes = faltantes.head(20)
        colores = plt.cm.Reds(top_faltantes['porcentaje_faltantes'] / 100)

        barras = ax.barh(range(len(top_faltantes)), top_faltantes['porcentaje_faltantes'],
                         color=colores, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_faltantes)))
        ax.set_yticklabels(top_faltantes['columna'])
        ax.set_xlabel('Porcentaje de Valores Faltantes (%)', fontsize=12)
        ax.set_title('Top 20 Columnas con Valores Faltantes',
                     fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Agregar etiquetas de porcentaje
        for i, (idx, row) in enumerate(top_faltantes.iterrows()):
            ax.text(row['porcentaje_faltantes'] + 0.5, i,
                   f"{row['porcentaje_faltantes']:.1f}%",
                   va='center', fontsize=9)

        plt.tight_layout()
        guardar_figura('01_valores_faltantes.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 01_valores_faltantes.png")
    else:
        print("¡No se encontraron valores faltantes en el dataset!")


def analizar_variable_objetivo(df):
    """
    Analiza la variable objetivo (precio_venta).

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE VARIABLE OBJETIVO: precio_venta", "-")

    # Limpiar datos
    precio_venta_limpio = df['precio_venta'].dropna()

    # Estadísticas descriptivas
    calcular_estadisticas_basicas(precio_venta_limpio, 'precio_venta')

    # Percentiles adicionales
    print("\nPercentiles adicionales:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        valor = precio_venta_limpio.quantile(p/100)
        print(f"  Percentil {p:2d}: {formatear_cop(valor)}")

    # Filtrar outliers extremos para visualización
    # Usamos percentil 99 para evitar que outliers extremos distorsionen el histograma
    limite_viz = precio_venta_limpio.quantile(0.99)
    precio_viz = precio_venta_limpio[precio_venta_limpio <= limite_viz]

    # Visualización: Distribución original
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histograma (sin outliers extremos para mejor visualización)
    axes[0, 0].hist(precio_viz, bins=100, edgecolor='black',
                    alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Precio de Venta (COP)', fontsize=11)
    axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
    axes[0, 0].set_title('Distribución de Precios de Venta (hasta percentil 99)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(precio_venta_limpio.median(), color='red',
                       linestyle='--', linewidth=2, label='Mediana')
    axes[0, 0].axvline(precio_venta_limpio.mean(), color='green',
                       linestyle='--', linewidth=2, label='Media')
    axes[0, 0].legend()

    # Agregar nota sobre outliers
    n_outliers_extremos = len(precio_venta_limpio) - len(precio_viz)
    pct_outliers = (n_outliers_extremos / len(precio_venta_limpio)) * 100
    axes[0, 0].text(0.98, 0.98, f'Nota: {n_outliers_extremos:,} valores extremos\n({pct_outliers:.2f}%) excluidos para\nmejor visualización',
                    transform=axes[0, 0].transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Boxplot
    bp = axes[0, 1].boxplot(precio_venta_limpio, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    axes[0, 1].set_ylabel('Precio de Venta (COP)', fontsize=11)
    axes[0, 1].set_title('Diagrama de Caja - Precios',
                         fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Distribución logarítmica
    axes[1, 0].hist(np.log10(precio_venta_limpio), bins=50,
                    edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].set_xlabel('log10(Precio de Venta)', fontsize=11)
    axes[1, 0].set_ylabel('Frecuencia', fontsize=11)
    axes[1, 0].set_title('Distribución Logarítmica de Precios',
                         fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Gráfico Q-Q para normalidad
    from scipy import stats
    stats.probplot(precio_venta_limpio, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Gráfico Q-Q (Prueba de Normalidad)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_figura('02_distribucion_precio_venta.png')
    plt.close()

    print(f"\n✓ Gráfico guardado: 02_distribucion_precio_venta.png")

    # Detección de outliers
    print("\nDetección de outliers en precio_venta:")
    Q1 = precio_venta_limpio.quantile(0.25)
    Q3 = precio_venta_limpio.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = (precio_venta_limpio < limite_inferior) | (precio_venta_limpio > limite_superior)
    print(f"  Outliers detectados: {outliers.sum():,} ({outliers.sum()/len(precio_venta_limpio)*100:.2f}%)")
    print(f"  Límite inferior: {formatear_cop(limite_inferior)}")
    print(f"  Límite superior: {formatear_cop(limite_superior)}")


def analizar_caracteristicas_numericas(df):
    """
    Analiza las características numéricas del dataset.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE CARACTERÍSTICAS NUMÉRICAS", "-")

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Se encontraron {len(columnas_numericas)} columnas numéricas")

    # Estadísticas descriptivas
    print("\nEstadísticas Descriptivas Generales:")
    desc_stats = df[columnas_numericas].describe()
    print(desc_stats.to_string())

    # Guardar estadísticas
    desc_stats.to_csv(obtener_ruta_resultados('estadisticas_numericas.csv'))
    print(f"\n✓ Estadísticas guardadas: estadisticas_numericas.csv")

    # Características clave para analizar
    caracteristicas_clave = ['area', 'habitaciones', 'banos', 'parqueaderos',
                            'estrato', 'piso', 'administracion', 'antiguedad']

    caracteristicas_disponibles = [col for col in caracteristicas_clave if col in df.columns]

    if len(caracteristicas_disponibles) > 0:
        # Visualizar distribuciones
        n_cols = 4
        n_rows = (len(caracteristicas_disponibles) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.ravel() if n_rows > 1 else [axes]

        for i, columna in enumerate(caracteristicas_disponibles):
            if i < len(axes):
                datos_limpios = df[columna].dropna()

                axes[i].hist(datos_limpios, bins=30, edgecolor='black',
                           alpha=0.7, color='teal')
                axes[i].set_xlabel(columna, fontsize=10)
                axes[i].set_ylabel('Frecuencia', fontsize=10)
                axes[i].set_title(f'Distribución: {columna}',
                                fontsize=11, fontweight='bold')
                axes[i].grid(True, alpha=0.3)

                # Agregar estadísticas en el gráfico
                media = datos_limpios.mean()
                mediana = datos_limpios.median()
                axes[i].axvline(media, color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'Media: {media:.1f}')
                axes[i].axvline(mediana, color='green', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'Mediana: {mediana:.1f}')
                axes[i].legend(fontsize=8)

        # Ocultar ejes no utilizados
        for i in range(len(caracteristicas_disponibles), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        guardar_figura('03_distribucion_caracteristicas_numericas.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 03_distribucion_caracteristicas_numericas.png")


def analizar_caracteristicas_categoricas(df):
    """
    Analiza las características categóricas del dataset.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE CARACTERÍSTICAS CATEGÓRICAS", "-")

    columnas_categoricas = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Se encontraron {len(columnas_categoricas)} columnas categóricas")

    # Características clave categóricas
    caracteristicas_clave = ['tipo_propiedad', 'tipo_operacion', 'localidad',
                            'sector', 'estado', 'compañia']

    caracteristicas_disponibles = [col for col in caracteristicas_clave if col in df.columns]

    for columna in caracteristicas_disponibles:
        print(f"\nConteo de valores para '{columna}':")
        conteo_valores = df[columna].value_counts()
        print(conteo_valores.head(10).to_string())
        print(f"  Total de valores únicos: {df[columna].nunique()}")

    # Visualización: Localidad
    if 'localidad' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))

        localidad_counts = df['localidad'].value_counts().head(15)
        colores = plt.cm.viridis(np.linspace(0, 1, len(localidad_counts)))

        barras = ax.barh(range(len(localidad_counts)), localidad_counts.values,
                        color=colores, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(localidad_counts)))
        ax.set_yticklabels(localidad_counts.index)
        ax.set_xlabel('Número de Apartamentos', fontsize=12)
        ax.set_title('Top 15 Localidades con Más Apartamentos',
                    fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Agregar etiquetas de conteo
        for i, valor in enumerate(localidad_counts.values):
            ax.text(valor + max(localidad_counts.values) * 0.01, i,
                   f'{valor:,}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        guardar_figura('04_top_localidades.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 04_top_localidades.png")

    # Visualización: Tipo de propiedad
    if 'tipo_propiedad' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))

        tipo_counts = df['tipo_propiedad'].value_counts()
        colores = plt.cm.Set3(np.linspace(0, 1, len(tipo_counts)))

        wedges, texts, autotexts = ax.pie(tipo_counts.values, labels=tipo_counts.index,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=colores, textprops={'fontsize': 10})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Distribución por Tipo de Propiedad',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        guardar_figura('05_distribucion_tipo_propiedad.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 05_distribucion_tipo_propiedad.png")


def analizar_amenidades(df):
    """
    Analiza la disponibilidad de amenidades en los apartamentos.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE AMENIDADES", "-")

    amenidades = ['jacuzzi', 'piscina', 'salon_comunal', 'terraza', 'vigilancia',
                 'chimenea', 'permite_mascotas', 'gimnasio', 'ascensor', 'conjunto_cerrado']

    amenidades_disponibles = [col for col in amenidades if col in df.columns]

    if len(amenidades_disponibles) > 0:
        conteo_amenidades = {}

        for amenidad in amenidades_disponibles:
            # Contar valores positivos (1, True, 'Si', etc.)
            valores = df[amenidad].value_counts()
            if 1 in valores.index:
                conteo_amenidades[amenidad] = valores[1]
            elif True in valores.index:
                conteo_amenidades[amenidad] = valores[True]

        amenidad_df = pd.DataFrame.from_dict(conteo_amenidades, orient='index',
                                             columns=['conteo'])
        amenidad_df = amenidad_df.sort_values('conteo', ascending=True)

        print("\nDisponibilidad de amenidades:")
        print(amenidad_df.to_string())

        # Visualización
        fig, ax = plt.subplots(figsize=(12, 8))

        colores = plt.cm.Blues(np.linspace(0.4, 0.9, len(amenidad_df)))
        barras = ax.barh(range(len(amenidad_df)), amenidad_df['conteo'].values,
                        color=colores, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(amenidad_df)))
        ax.set_yticklabels(amenidad_df.index)
        ax.set_xlabel('Número de Propiedades', fontsize=12)
        ax.set_title('Disponibilidad de Amenidades en Apartamentos',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        # Agregar etiquetas
        for i, valor in enumerate(amenidad_df['conteo'].values):
            porcentaje = (valor / len(df)) * 100
            ax.text(valor + max(amenidad_df['conteo'].values) * 0.01, i,
                   f'{valor:,} ({porcentaje:.1f}%)',
                   va='center', fontsize=9)

        plt.tight_layout()
        guardar_figura('06_disponibilidad_amenidades.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 06_disponibilidad_amenidades.png")


def analizar_correlaciones(df):
    """
    Analiza las correlaciones entre variables numéricas.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE CORRELACIONES", "-")

    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    # Correlación con la variable objetivo
    if 'precio_venta' in columnas_numericas:
        correlaciones = df[columnas_numericas].corr()['precio_venta'].sort_values(ascending=False)

        print("\nTop 15 características más correlacionadas con precio_venta:")
        print(correlaciones.head(15).to_string())

        print("\n10 características menos correlacionadas con precio_venta:")
        print(correlaciones.tail(10).to_string())

        # Visualización: Correlación con target
        fig, ax = plt.subplots(figsize=(12, 10))

        top_corr = correlaciones.head(20)
        colores = ['green' if x > 0 else 'red' for x in top_corr.values]

        barras = ax.barh(range(len(top_corr)), top_corr.values, color=colores,
                        edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index)
        ax.set_xlabel('Coeficiente de Correlación', fontsize=12)
        ax.set_title('Top 20 Características Correlacionadas con Precio de Venta',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # Agregar etiquetas
        for i, valor in enumerate(top_corr.values):
            ax.text(valor + 0.01 if valor > 0 else valor - 0.01, i,
                   f'{valor:.3f}',
                   va='center', ha='left' if valor > 0 else 'right',
                   fontsize=9, fontweight='bold')

        plt.tight_layout()
        guardar_figura('07_correlacion_con_precio.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 07_correlacion_con_precio.png")

        # Matriz de correlación (top features)
        top_features = correlaciones.head(15).index.tolist()

        fig, ax = plt.subplots(figsize=(14, 12))

        matriz_corr = df[top_features].corr()

        im = ax.imshow(matriz_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        # Configurar ticks
        ax.set_xticks(np.arange(len(top_features)))
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(top_features, fontsize=9)

        # Agregar valores en las celdas
        for i in range(len(top_features)):
            for j in range(len(top_features)):
                valor = matriz_corr.iloc[i, j]
                color = 'white' if abs(valor) > 0.5 else 'black'
                ax.text(j, i, f'{valor:.2f}',
                       ha='center', va='center', color=color, fontsize=8)

        ax.set_title('Matriz de Correlación - Top 15 Características',
                    fontsize=14, fontweight='bold', pad=20)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coeficiente de Correlación', fontsize=11)

        plt.tight_layout()
        guardar_figura('08_matriz_correlacion.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 08_matriz_correlacion.png")


def analizar_geografia(df):
    """
    Analiza los patrones geográficos del dataset.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS GEOGRÁFICO", "-")

    if 'latitud' in df.columns and 'longitud' in df.columns and 'precio_venta' in df.columns:
        # Filtrar datos con coordenadas válidas
        df_geo = df.dropna(subset=['latitud', 'longitud', 'precio_venta'])

        print(f"Propiedades con coordenadas válidas: {len(df_geo):,}")

        # Visualización geográfica
        fig, ax = plt.subplots(figsize=(14, 12))

        scatter = ax.scatter(df_geo['longitud'], df_geo['latitud'],
                           c=np.log10(df_geo['precio_venta']),
                           cmap='viridis', alpha=0.6, s=15,
                           edgecolor='black', linewidth=0.1)

        ax.set_xlabel('Longitud', fontsize=12)
        ax.set_ylabel('Latitud', fontsize=12)
        ax.set_title('Distribución Geográfica de Apartamentos en Bogotá\n(coloreado por log10(precio))',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log10(Precio de Venta)', fontsize=11)

        plt.tight_layout()
        guardar_figura('09_distribucion_geografica.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 09_distribucion_geografica.png")

    # Análisis de precio por localidad
    if 'localidad' in df.columns and 'precio_venta' in df.columns:
        precio_por_localidad = df.groupby('localidad')['precio_venta'].agg([
            ('promedio', 'mean'),
            ('mediana', 'median'),
            ('conteo', 'count'),
            ('std', 'std')
        ]).sort_values('promedio', ascending=False)

        top_15_localidades = precio_por_localidad.head(15)

        print("\nTop 15 Localidades por Precio Promedio:")
        print(top_15_localidades.to_string())

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Gráfico 1: Promedio y Mediana
        x = range(len(top_15_localidades))
        width = 0.35

        axes[0].barh([i - width/2 for i in x], top_15_localidades['promedio'].values,
                    width, label='Promedio', color='steelblue', edgecolor='black')
        axes[0].barh([i + width/2 for i in x], top_15_localidades['mediana'].values,
                    width, label='Mediana', color='coral', edgecolor='black')

        axes[0].set_yticks(x)
        axes[0].set_yticklabels(top_15_localidades.index, fontsize=10)
        axes[0].set_xlabel('Precio (COP)', fontsize=12)
        axes[0].set_title('Precio Promedio y Mediana por Localidad (Top 15)',
                         fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')

        # Gráfico 2: Conteo de propiedades
        colores = plt.cm.viridis(np.linspace(0, 1, len(top_15_localidades)))
        axes[1].barh(x, top_15_localidades['conteo'].values,
                    color=colores, edgecolor='black')
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(top_15_localidades.index, fontsize=10)
        axes[1].set_xlabel('Número de Apartamentos', fontsize=12)
        axes[1].set_title('Cantidad de Apartamentos por Localidad (Top 15)',
                         fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')

        # Agregar etiquetas
        for i, valor in enumerate(top_15_localidades['conteo'].values):
            axes[1].text(valor + max(top_15_localidades['conteo'].values) * 0.01, i,
                        f'{valor:,}', va='center', fontsize=9)

        plt.tight_layout()
        guardar_figura('10_precio_por_localidad.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 10_precio_por_localidad.png")


def analizar_precio_por_caracteristicas(df):
    """
    Analiza cómo el precio varía según diferentes características.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("ANÁLISIS DE PRECIO POR CARACTERÍSTICAS", "-")

    # Precio por número de habitaciones
    if 'habitaciones' in df.columns and 'precio_venta' in df.columns:
        precio_por_habitaciones = df.groupby('habitaciones')['precio_venta'].agg([
            ('promedio', 'mean'),
            ('mediana', 'median'),
            ('conteo', 'count')
        ])

        print("\nPrecio por número de habitaciones:")
        print(precio_por_habitaciones.to_string())

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Boxplot
        df_hab = df[df['habitaciones'].between(1, 6)].copy()
        datos_boxplot = [df_hab[df_hab['habitaciones'] == i]['precio_venta'].dropna()
                        for i in range(1, 7)]

        bp = axes[0].boxplot(datos_boxplot, labels=range(1, 7), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_edgecolor('black')

        axes[0].set_xlabel('Número de Habitaciones', fontsize=12)
        axes[0].set_ylabel('Precio de Venta (COP)', fontsize=12)
        axes[0].set_title('Distribución de Precios por Número de Habitaciones',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Barplot con promedio
        habitaciones_validas = precio_por_habitaciones[precio_por_habitaciones['conteo'] >= 10]

        x = habitaciones_validas.index
        axes[1].bar(x, habitaciones_validas['promedio'].values,
                   alpha=0.7, color='steelblue', edgecolor='black', label='Promedio')
        axes[1].plot(x, habitaciones_validas['mediana'].values,
                    'ro-', linewidth=2, markersize=8, label='Mediana')

        axes[1].set_xlabel('Número de Habitaciones', fontsize=12)
        axes[1].set_ylabel('Precio (COP)', fontsize=12)
        axes[1].set_title('Precio Promedio y Mediana por Número de Habitaciones',
                         fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        guardar_figura('11_precio_por_habitaciones.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 11_precio_por_habitaciones.png")

    # Precio por estrato
    if 'estrato' in df.columns and 'precio_venta' in df.columns:
        precio_por_estrato = df.groupby('estrato')['precio_venta'].agg([
            ('promedio', 'mean'),
            ('mediana', 'median'),
            ('conteo', 'count')
        ])

        print("\nPrecio por estrato socioeconómico:")
        print(precio_por_estrato.to_string())

        # Visualización
        fig, ax = plt.subplots(figsize=(12, 7))

        x = precio_por_estrato.index
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], precio_por_estrato['promedio'].values,
                      width, label='Promedio', color='steelblue', edgecolor='black')
        bars2 = ax.bar([i + width/2 for i in x], precio_por_estrato['mediana'].values,
                      width, label='Mediana', color='coral', edgecolor='black')

        ax.set_xlabel('Estrato Socioeconómico', fontsize=12)
        ax.set_ylabel('Precio (COP)', fontsize=12)
        ax.set_title('Precio de Apartamentos por Estrato Socioeconómico',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        guardar_figura('12_precio_por_estrato.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 12_precio_por_estrato.png")

    # Precio por metro cuadrado
    if 'area' in df.columns and 'precio_venta' in df.columns:
        df_copy = df.copy()
        df_copy['precio_m2'] = df_copy['precio_venta'] / df_copy['area']
        df_precio_m2 = df_copy.dropna(subset=['precio_m2'])
        df_precio_m2 = df_precio_m2[df_precio_m2['precio_m2'] > 0]

        print("\nEstadísticas de precio por metro cuadrado:")
        calcular_estadisticas_basicas(df_precio_m2['precio_m2'], 'precio_m2')

        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histograma
        axes[0].hist(df_precio_m2['precio_m2'], bins=50, edgecolor='black',
                    alpha=0.7, color='teal')
        axes[0].set_xlabel('Precio por m² (COP)', fontsize=12)
        axes[0].set_ylabel('Frecuencia', fontsize=12)
        axes[0].set_title('Distribución de Precio por Metro Cuadrado',
                         fontsize=12, fontweight='bold')
        axes[0].axvline(df_precio_m2['precio_m2'].median(), color='red',
                       linestyle='--', linewidth=2, label='Mediana')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter: Area vs Precio
        sample = df_precio_m2.sample(min(5000, len(df_precio_m2)))
        axes[1].scatter(sample['area'], sample['precio_venta'],
                       alpha=0.5, s=20, c=sample['precio_m2'],
                       cmap='viridis', edgecolor='none')
        axes[1].set_xlabel('Área (m²)', fontsize=12)
        axes[1].set_ylabel('Precio de Venta (COP)', fontsize=12)
        axes[1].set_title('Relación entre Área y Precio',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        guardar_figura('13_precio_por_metro_cuadrado.png')
        plt.close()

        print(f"\n✓ Gráfico guardado: 13_precio_por_metro_cuadrado.png")


def generar_resumen_ejecutivo(df):
    """
    Genera un resumen ejecutivo del análisis exploratorio.

    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset de apartamentos
    """
    imprimir_encabezado("RESUMEN EJECUTIVO DEL ANÁLISIS EXPLORATORIO", "-")

    resumen = {
        'total_registros': len(df),
        'total_columnas': len(df.columns),
        'columnas_numericas': len(df.select_dtypes(include=[np.number]).columns),
        'columnas_categoricas': len(df.select_dtypes(include=['object']).columns),
        'total_valores_faltantes': df.isnull().sum().sum(),
        'porcentaje_completitud': ((df.size - df.isnull().sum().sum()) / df.size * 100)
    }

    if 'precio_venta' in df.columns:
        precio_limpio = df['precio_venta'].dropna()
        resumen.update({
            'precio_promedio': precio_limpio.mean(),
            'precio_mediana': precio_limpio.median(),
            'precio_min': precio_limpio.min(),
            'precio_max': precio_limpio.max(),
            'precio_std': precio_limpio.std()
        })

    print("\n═══════════════════════════════════════════════════════════════")
    print("                     RESUMEN DEL DATASET")
    print("═══════════════════════════════════════════════════════════════")
    print(f"\n📊 Dimensiones:")
    print(f"   • Total de registros: {resumen['total_registros']:,}")
    print(f"   • Total de columnas: {resumen['total_columnas']}")
    print(f"   • Columnas numéricas: {resumen['columnas_numericas']}")
    print(f"   • Columnas categóricas: {resumen['columnas_categoricas']}")

    print(f"\n📋 Calidad de Datos:")
    print(f"   • Total valores faltantes: {resumen['total_valores_faltantes']:,}")
    print(f"   • Completitud: {resumen['porcentaje_completitud']:.2f}%")

    if 'precio_promedio' in resumen:
        print(f"\n💰 Estadísticas de Precio:")
        print(f"   • Promedio: {formatear_cop(resumen['precio_promedio'])}")
        print(f"   • Mediana: {formatear_cop(resumen['precio_mediana'])}")
        print(f"   • Mínimo: {formatear_cop(resumen['precio_min'])}")
        print(f"   • Máximo: {formatear_cop(resumen['precio_max'])}")
        print(f"   • Desv. Estándar: {formatear_cop(resumen['precio_std'])}")

    print("\n═══════════════════════════════════════════════════════════════")

    # Guardar resumen
    resumen_df = pd.DataFrame([resumen])
    resumen_df.to_csv(obtener_ruta_resultados('resumen_eda.csv'), index=False)
    print(f"\n✓ Resumen guardado: resumen_eda.csv")

    return resumen


def main():
    """Función principal que ejecuta todo el análisis exploratorio."""

    imprimir_encabezado("ANÁLISIS EXPLORATORIO DE DATOS - PROYECTO HABITALPIES")

    # Cargar datos
    df = cargar_datos()

    # Ejecutar todos los análisis
    analizar_estructura_datos(df)
    analizar_valores_faltantes(df)
    analizar_variable_objetivo(df)
    analizar_caracteristicas_numericas(df)
    analizar_caracteristicas_categoricas(df)
    analizar_amenidades(df)
    analizar_correlaciones(df)
    analizar_geografia(df)
    analizar_precio_por_caracteristicas(df)
    generar_resumen_ejecutivo(df)

    imprimir_encabezado("✓ ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
    print("Todos los gráficos se guardaron en: reports/figures/")
    print("Todos los resultados se guardaron en: data/results/")


if __name__ == '__main__':
    main()
