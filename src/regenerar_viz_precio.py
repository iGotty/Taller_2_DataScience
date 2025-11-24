#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script rápido para regenerar solo la visualización de precio_venta
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Importar utilidades
from utils import cargar_datos, formatear_cop

print("Regenerando visualización de precio_venta...")

# Cargar datos
df = cargar_datos()

# Limpiar datos
precio_venta_limpio = df['precio_venta'].dropna()

print(f"\nDatos cargados: {len(precio_venta_limpio):,} precios válidos")
print(f"Rango: {formatear_cop(precio_venta_limpio.min())} - {formatear_cop(precio_venta_limpio.max())}")

# Filtrar outliers extremos para visualización
# Usamos percentil 99 para evitar que outliers extremos distorsionen el histograma
limite_viz = precio_venta_limpio.quantile(0.99)
precio_viz = precio_venta_limpio[precio_venta_limpio <= limite_viz]

print(f"\nPercentil 99: {formatear_cop(limite_viz)}")
print(f"Datos para visualización: {len(precio_viz):,} ({len(precio_viz)/len(precio_venta_limpio)*100:.2f}%)")

# Crear figura
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
stats.probplot(precio_venta_limpio, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Gráfico Q-Q (Prueba de Normalidad)',
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Guardar figura
FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
output_path = FIGURES_DIR / '02_distribucion_precio_venta.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Visualización regenerada exitosamente!")
print(f"   Ubicación: {output_path}")
