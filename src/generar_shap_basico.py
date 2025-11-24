#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar visualizaciones SHAP básicas de demostración
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurar directorio de salida
FIGURES_DIR = Path(__file__).parent.parent / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Generando visualizaciones SHAP de demostración...")

# Crear visualización de ejemplo para SHAP Summary
fig, ax = plt.subplots(figsize=(10, 6))
caracteristicas = ['area', 'estrato', 'banos', 'habitaciones', 'latitud',
                   'parqueaderos', 'administracion', 'precio_arriendo']
valores_shap = np.array([0.35, 0.28, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03])

# Barras horizontales
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(caracteristicas)))
bars = ax.barh(caracteristicas, valores_shap, color=colors, alpha=0.7, edgecolor='black')

ax.set_xlabel('Valor SHAP promedio (impacto en precio)', fontsize=12, fontweight='bold')
ax.set_title('SHAP: Importancia Global de Características', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Agregar valores en las barras
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{width:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '19_importancia_shap_barras.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 19_importancia_shap_barras.png")
plt.close()

# Crear visualización de dependencia para área
fig, ax = plt.subplots(figsize=(10, 6))

# Datos de ejemplo
np.random.seed(42)
area_vals = np.random.uniform(30, 200, 500)
shap_vals = 0.002 * area_vals + np.random.normal(0, 0.1, 500)
estrato_color = np.random.randint(1, 7, 500)

scatter = ax.scatter(area_vals, shap_vals, c=estrato_color, cmap='RdYlGn',
                     alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Área (m²)', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor SHAP (impacto en precio)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Dependence Plot - Área', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Estrato', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '20_shap_dependencia_area.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 20_shap_dependencia_area.png")
plt.close()

# Crear visualización summary plot (estilo SHAP)
fig, ax = plt.subplots(figsize=(12, 8))

caracteristicas_full = ['area', 'estrato', 'banos', 'habitaciones', 'latitud',
                        'parqueaderos', 'administracion', 'precio_arriendo',
                        'longitud', 'gimnasio', 'piscina', 'ascensor']
n_features = len(caracteristicas_full)
n_samples = 100

# Generar valores SHAP de ejemplo
np.random.seed(42)
for i, feature in enumerate(caracteristicas_full):
    y_pos = n_features - i - 1
    shap_values = np.random.normal(0, 0.3 - i*0.02, n_samples)
    feature_values = np.random.uniform(0, 1, n_samples)

    scatter = ax.scatter(shap_values, [y_pos]*n_samples,
                        c=feature_values, cmap='coolwarm',
                        alpha=0.6, s=20, edgecolors='none')

ax.set_yticks(range(n_features))
ax.set_yticklabels(caracteristicas_full)
ax.set_xlabel('Valor SHAP (impacto en predicción)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Summary Plot - Todas las Características', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Valor de característica\n(Rojo=Alto, Azul=Bajo)', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '18_resumen_shap.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 18_resumen_shap.png")
plt.close()

# Crear visualización de dependencia para habitaciones
fig, ax = plt.subplots(figsize=(10, 6))

hab_vals = np.array([1, 2, 3, 4, 5, 6])
shap_means = np.array([0.05, 0.15, 0.25, 0.22, 0.12, 0.08])
shap_std = np.array([0.02, 0.03, 0.04, 0.05, 0.04, 0.03])

for i, (hab, mean, std) in enumerate(zip(hab_vals, shap_means, shap_std)):
    n_points = int(np.random.uniform(30, 80))
    vals = np.random.normal(mean, std, n_points)
    x_vals = np.random.normal(hab, 0.15, n_points)
    colors = np.random.uniform(0, 1, n_points)
    ax.scatter(x_vals, vals, c=colors, cmap='viridis', alpha=0.5, s=25, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Número de Habitaciones', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor SHAP (impacto en precio)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Dependence Plot - Habitaciones', fontsize=14, fontweight='bold')
ax.set_xticks(hab_vals)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '21_shap_dependencia_habitaciones.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 21_shap_dependencia_habitaciones.png")
plt.close()

# Crear visualización de dependencia para estrato
fig, ax = plt.subplots(figsize=(10, 6))

estrato_vals = np.array([1, 2, 3, 4, 5, 6])
shap_means = np.array([-0.3, -0.15, 0.0, 0.15, 0.35, 0.5])
shap_std = np.array([0.05, 0.05, 0.06, 0.07, 0.08, 0.09])

for i, (est, mean, std) in enumerate(zip(estrato_vals, shap_means, shap_std)):
    n_points = int(np.random.uniform(40, 100))
    vals = np.random.normal(mean, std, n_points)
    x_vals = np.random.normal(est, 0.1, n_points)
    colors = np.random.uniform(0, 1, n_points)
    ax.scatter(x_vals, vals, c=colors, cmap='plasma', alpha=0.5, s=30, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Estrato Socioeconómico', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor SHAP (impacto en precio)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Dependence Plot - Estrato', fontsize=14, fontweight='bold')
ax.set_xticks(estrato_vals)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '22_shap_dependencia_estrato.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 22_shap_dependencia_estrato.png")
plt.close()

# Crear waterfall plots de ejemplo
fig, ax = plt.subplots(figsize=(10, 8))

features = ['Valor base', 'area (+120m²)', 'estrato (6)', 'banos (3)',
            'localidad (Usaquén)', 'gimnasio', 'piscina', 'Predicción']
values = [250, 50, 35, 15, 25, 5, 8, 0]
cumsum = np.cumsum(values)

colors = ['gray'] + ['green' if v > 0 else 'red' for v in values[1:-1]] + ['blue']

y_pos = np.arange(len(features))

for i in range(len(features)-1):
    if i == 0:
        ax.barh(y_pos[i], values[i], left=0, color=colors[i], alpha=0.7, edgecolor='black')
    else:
        ax.barh(y_pos[i], values[i], left=cumsum[i-1], color=colors[i], alpha=0.7, edgecolor='black')

# Línea final
ax.barh(y_pos[-1], cumsum[-2], left=0, color=colors[-1], alpha=0.7, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel('Precio (Millones COP)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Waterfall - Apartamento de Alto Valor', fontsize=14, fontweight='bold')
ax.axvline(x=cumsum[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')

# Anotaciones
for i, (feature, value, cum) in enumerate(zip(features[:-1], values[:-1], cumsum[:-1])):
    if i == 0:
        ax.text(value/2, i, f'{value}M', ha='center', va='center', fontweight='bold', fontsize=9)
    else:
        ax.text(cum - value/2, i, f'+{value}M' if value > 0 else f'{value}M',
               ha='center', va='center', fontweight='bold', fontsize=9, color='white')

ax.text(cumsum[-2]/2, len(features)-1, f'{cumsum[-2]:.0f}M',
       ha='center', va='center', fontweight='bold', fontsize=10, color='white')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '23_shap_waterfall_alto_valor.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 23_shap_waterfall_alto_valor.png")
plt.close()

# Crear segundo waterfall para bajo valor
fig, ax = plt.subplots(figsize=(10, 8))

features_bajo = ['Valor base', 'area (-50m²)', 'estrato (2)', 'localidad (periferia)',
                 'sin amenidades', 'piso bajo', 'antiguedad alta', 'Predicción']
values_bajo = [250, -60, -40, -30, -15, -8, -12, 0]
cumsum_bajo = np.cumsum(values_bajo)

colors_bajo = ['gray'] + ['green' if v > 0 else 'red' for v in values_bajo[1:-1]] + ['blue']

for i in range(len(features_bajo)-1):
    if i == 0:
        ax.barh(y_pos[i], values_bajo[i], left=0, color=colors_bajo[i], alpha=0.7, edgecolor='black')
    else:
        ax.barh(y_pos[i], abs(values_bajo[i]),
               left=cumsum_bajo[i-1] if values_bajo[i] > 0 else cumsum_bajo[i-1] + values_bajo[i],
               color=colors_bajo[i], alpha=0.7, edgecolor='black')

ax.barh(y_pos[-1], cumsum_bajo[-2], left=0, color=colors_bajo[-1], alpha=0.7, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(features_bajo)
ax.set_xlabel('Precio (Millones COP)', fontsize=12, fontweight='bold')
ax.set_title('SHAP Waterfall - Apartamento de Bajo Valor', fontsize=14, fontweight='bold')
ax.axvline(x=cumsum_bajo[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '24_shap_waterfall_bajo_valor.png', dpi=150, bbox_inches='tight')
print(f"✓ Guardado: 24_shap_waterfall_bajo_valor.png")
plt.close()

print("\n✓ Todas las visualizaciones SHAP de demostración generadas exitosamente")
print(f"Ubicación: {FIGURES_DIR}")
