#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar visualizaciones faltantes del notebook 01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importar utilidades
from utils import cargar_datos

print("Generando visualizaciones faltantes para notebook 01...")

# Cargar datos
df = cargar_datos()

# Configurar directorio de salida
FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def guardar_figura(nombre_archivo):
    """Guarda la figura actual en el directorio de reportes."""
    ruta_salida = FIGURES_DIR / nombre_archivo
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    print(f"✓ {nombre_archivo}")

# ============================================================================
# 03: DISTRIBUCIÓN CARACTERÍSTICAS NUMÉRICAS
# ============================================================================

print("\n1. Generando distribución características numéricas...")

caracteristicas_clave = ['area', 'habitaciones', 'banos', 'parqueaderos',
                        'estrato', 'administracion']
caracteristicas_disponibles = [col for col in caracteristicas_clave if col in df.columns]

if len(caracteristicas_disponibles) > 0:
    n_cols = 3
    n_rows = (len(caracteristicas_disponibles) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.ravel() if n_rows > 1 else [axes]

    for i, columna in enumerate(caracteristicas_disponibles):
        if i < len(axes):
            datos_limpios = df[columna].dropna()

            if pd.api.types.is_numeric_dtype(df[columna]):
                axes[i].hist(datos_limpios, bins=30, edgecolor='black',
                           alpha=0.7, color='teal')
                axes[i].set_xlabel(columna, fontsize=10)
                axes[i].set_ylabel('Frecuencia', fontsize=10)
                axes[i].set_title(f'Distribución: {columna}',
                                fontsize=11, fontweight='bold')
                axes[i].grid(True, alpha=0.3)

                media = datos_limpios.mean()
                mediana = datos_limpios.median()
                axes[i].axvline(media, color='red', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'Media: {media:.1f}')
                axes[i].axvline(mediana, color='green', linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f'Mediana: {mediana:.1f}')
                axes[i].legend(fontsize=8)

    for i in range(len(caracteristicas_disponibles), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    guardar_figura('03_distribucion_caracteristicas_numericas.png')
    plt.close()

# ============================================================================
# 04: TOP LOCALIDADES
# ============================================================================

print("\n2. Generando top localidades...")

if 'localidad' in df.columns:
    localidad_counts = df['localidad'].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    colores = plt.cm.viridis(np.linspace(0, 1, len(localidad_counts)))
    ax.barh(range(len(localidad_counts)), localidad_counts.values,
           color=colores, edgecolor='black')
    ax.set_yticks(range(len(localidad_counts)))
    ax.set_yticklabels(localidad_counts.index, fontsize=10)
    ax.set_xlabel('Número de Apartamentos', fontsize=12)
    ax.set_title('Top 15 Localidades con Más Apartamentos',
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for i, valor in enumerate(localidad_counts.values):
        ax.text(valor + max(localidad_counts.values) * 0.01, i,
               f'{valor:,}', va='center', fontsize=9)

    plt.tight_layout()
    guardar_figura('04_top_localidades.png')
    plt.close()

# ============================================================================
# 06: DISPONIBILIDAD AMENIDADES
# ============================================================================

print("\n3. Generando disponibilidad amenidades...")

amenidades = ['piscina', 'gimnasio', 'ascensor', 'vigilancia',
              'conjunto_cerrado', 'terraza']
amenidades_disponibles = [col for col in amenidades if col in df.columns]

if amenidades_disponibles:
    datos_amenidades = []
    for amenidad in amenidades_disponibles:
        conteo = df[amenidad].sum()
        porcentaje = (conteo / len(df)) * 100
        datos_amenidades.append({'Amenidad': amenidad.replace('_', ' ').title(),
                                 'Cantidad': int(conteo),
                                 'Porcentaje': porcentaje})

    df_amenidades = pd.DataFrame(datos_amenidades).sort_values('Porcentaje', ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colores = plt.cm.viridis(np.linspace(0, 1, len(df_amenidades)))
    ax1.barh(range(len(df_amenidades)), df_amenidades['Porcentaje'].values,
            color=colores, edgecolor='black')
    ax1.set_yticks(range(len(df_amenidades)))
    ax1.set_yticklabels(df_amenidades['Amenidad'], fontsize=10)
    ax1.set_xlabel('Porcentaje de Apartamentos (%)', fontsize=11)
    ax1.set_title('Disponibilidad de Amenidades (% del Total)',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    for i, (porc, cant) in enumerate(zip(df_amenidades['Porcentaje'].values,
                                          df_amenidades['Cantidad'].values)):
        ax1.text(porc + 2, i, f'{porc:.1f}% ({cant:,})',
                va='center', fontsize=9)

    ax2.bar(range(len(df_amenidades)), df_amenidades['Cantidad'].values,
           color=colores, edgecolor='black')
    ax2.set_xticks(range(len(df_amenidades)))
    ax2.set_xticklabels(df_amenidades['Amenidad'], rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Número de Apartamentos', fontsize=11)
    ax2.set_title('Cantidad Absoluta de Apartamentos con Amenidad',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    guardar_figura('06_disponibilidad_amenidades.png')
    plt.close()

# ============================================================================
# 07: CORRELACIÓN CON PRECIO
# ============================================================================

print("\n4. Generando correlación con precio...")

if 'precio_venta' in df.columns:
    correlaciones = df.select_dtypes(include=[np.number]).corr()['precio_venta'].sort_values(ascending=False)
    correlaciones = correlaciones.drop('precio_venta').head(20)

    fig, ax = plt.subplots(figsize=(10, 10))
    colores = ['green' if x > 0 else 'red' for x in correlaciones.values]
    ax.barh(range(len(correlaciones)), correlaciones.values,
           color=colores, edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(correlaciones)))
    ax.set_yticklabels(correlaciones.index, fontsize=10)
    ax.set_xlabel('Correlación con Precio de Venta', fontsize=12)
    ax.set_title('Top 20 Características Correlacionadas con Precio de Venta',
                fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    for i, valor in enumerate(correlaciones.values):
        ax.text(valor + 0.01 if valor > 0 else valor - 0.01, i,
               f'{valor:.3f}', va='center',
               ha='left' if valor > 0 else 'right', fontsize=8)

    plt.tight_layout()
    guardar_figura('07_correlacion_con_precio.png')
    plt.close()

# ============================================================================
# 08: MATRIZ CORRELACIÓN
# ============================================================================

print("\n5. Generando matriz de correlación...")

caracteristicas_principales = ['precio_venta', 'area', 'habitaciones', 'banos',
                               'parqueaderos', 'estrato', 'administracion',
                               'gimnasio', 'ascensor', 'piscina']
caracteristicas_principales = [col for col in caracteristicas_principales if col in df.columns]

if len(caracteristicas_principales) > 1:
    matriz_corr = df[caracteristicas_principales].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
               ax=ax)
    ax.set_title('Matriz de Correlación - Características Principales',
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    guardar_figura('08_matriz_correlacion.png')
    plt.close()

# ============================================================================
# 11, 12, 13: PRECIO POR CARACTERÍSTICAS
# ============================================================================

print("\n6. Generando precio por habitaciones...")

if 'habitaciones' in df.columns and 'precio_venta' in df.columns:
    df_temp = df.dropna(subset=['habitaciones', 'precio_venta'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_temp.boxplot(column='precio_venta', by='habitaciones', ax=ax1)
    ax1.set_xlabel('Número de Habitaciones', fontsize=12)
    ax1.set_ylabel('Precio de Venta (COP)', fontsize=12)
    ax1.set_title('Distribución de Precios por Número de Habitaciones',
                 fontsize=12, fontweight='bold')
    ax1.get_figure().suptitle('')

    precio_promedio = df_temp.groupby('habitaciones')['precio_venta'].mean().sort_index()
    ax2.plot(precio_promedio.index, precio_promedio.values, marker='o',
            markersize=8, linewidth=2, color='steelblue')
    ax2.set_xlabel('Número de Habitaciones', fontsize=12)
    ax2.set_ylabel('Precio Promedio (COP)', fontsize=12)
    ax2.set_title('Precio Promedio por Número de Habitaciones',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_figura('11_precio_por_habitaciones.png')
    plt.close()

print("\n7. Generando precio por estrato...")

if 'estrato' in df.columns and 'precio_venta' in df.columns:
    df_temp = df.dropna(subset=['estrato', 'precio_venta'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_temp.boxplot(column='precio_venta', by='estrato', ax=ax1)
    ax1.set_xlabel('Estrato Socioeconómico', fontsize=12)
    ax1.set_ylabel('Precio de Venta (COP)', fontsize=12)
    ax1.set_title('Distribución de Precios por Estrato',
                 fontsize=12, fontweight='bold')
    ax1.get_figure().suptitle('')

    precio_promedio = df_temp.groupby('estrato')['precio_venta'].mean().sort_index()
    ax2.bar(precio_promedio.index, precio_promedio.values,
           color=plt.cm.viridis(np.linspace(0, 1, len(precio_promedio))),
           edgecolor='black')
    ax2.set_xlabel('Estrato Socioeconómico', fontsize=12)
    ax2.set_ylabel('Precio Promedio (COP)', fontsize=12)
    ax2.set_title('Precio Promedio por Estrato',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    guardar_figura('12_precio_por_estrato.png')
    plt.close()

print("\n8. Generando precio por metro cuadrado...")

if 'area' in df.columns and 'precio_venta' in df.columns:
    df_temp = df.dropna(subset=['area', 'precio_venta'])
    df_temp = df_temp[(df_temp['area'] > 0) & (df_temp['area'] < 500)]  # Filtrar outliers
    df_temp['precio_m2'] = df_temp['precio_venta'] / df_temp['area']

    # Filtrar outliers de precio_m2
    q99 = df_temp['precio_m2'].quantile(0.99)
    df_temp = df_temp[df_temp['precio_m2'] <= q99]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(df_temp['area'], df_temp['precio_m2'], alpha=0.3, s=10)
    ax1.set_xlabel('Área (m²)', fontsize=12)
    ax1.set_ylabel('Precio por m² (COP)', fontsize=12)
    ax1.set_title('Precio por Metro Cuadrado vs Área',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.hist(df_temp['precio_m2'], bins=50, edgecolor='black',
            alpha=0.7, color='coral')
    ax2.set_xlabel('Precio por m² (COP)', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución de Precio por Metro Cuadrado',
                 fontsize=12, fontweight='bold')
    ax2.axvline(df_temp['precio_m2'].median(), color='red',
               linestyle='--', linewidth=2, label='Mediana')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    guardar_figura('13_precio_por_metro_cuadrado.png')
    plt.close()

print("\n✓ Todas las visualizaciones generadas exitosamente!")
print(f"   Ubicación: {FIGURES_DIR}")
