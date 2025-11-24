#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script rápido para generar visualizaciones geográficas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Importar utilidades
from utils import cargar_datos

print("Generando visualizaciones geográficas...")

# Cargar datos
df = cargar_datos()

# Configurar directorio de salida
FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def guardar_figura(nombre_archivo):
    """Guarda la figura actual en el directorio de reportes."""
    ruta_salida = FIGURES_DIR / nombre_archivo
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    print(f"✓ Guardado: {nombre_archivo}")

# ============================================================================
# 1. DISTRIBUCIÓN GEOGRÁFICA
# ============================================================================

print("\n1. Generando distribución geográfica...")

if 'latitud' in df.columns and 'longitud' in df.columns and 'precio_venta' in df.columns:
    # Filtrar datos con coordenadas válidas
    df_geo = df.dropna(subset=['latitud', 'longitud', 'precio_venta'])

    print(f"   Propiedades con coordenadas válidas: {len(df_geo):,}")

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

# ============================================================================
# 2. PRECIO POR LOCALIDAD
# ============================================================================

print("\n2. Generando precio por localidad...")

if 'localidad' in df.columns and 'precio_venta' in df.columns:
    precio_por_localidad = df.groupby('localidad')['precio_venta'].agg([
        ('promedio', 'mean'),
        ('mediana', 'median'),
        ('conteo', 'count'),
        ('std', 'std')
    ]).sort_values('promedio', ascending=False)

    top_15_localidades = precio_por_localidad.head(15)

    print(f"   Top 15 localidades por precio promedio")

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

print("\n✓ Todas las visualizaciones geográficas generadas exitosamente!")
print(f"   Ubicación: {FIGURES_DIR}")
