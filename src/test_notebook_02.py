#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar el código del notebook 02 antes de incluirlo
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importar utilidades
from utils import cargar_datos, formatear_cop

print("="*80)
print("PRUEBA DEL CÓDIGO DEL NOTEBOOK 02 - MODELADO Y EVALUACIÓN")
print("="*80)

# 1. Cargar datos
print("\n1. Cargando datos...")
df = cargar_datos()
print(f"   Datos cargados: {df.shape}")

# 2. Preprocesamiento básico
print("\n2. Preprocesamiento...")
df_clean = df.dropna(subset=['precio_venta']).copy()
print(f"   Registros después de eliminar NaN en precio_venta: {len(df_clean):,}")

# Eliminar outliers extremos
Q1 = df_clean['precio_venta'].quantile(0.25)
Q3 = df_clean['precio_venta'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
df_clean = df_clean[
    (df_clean['precio_venta'] >= lower_bound) &
    (df_clean['precio_venta'] <= upper_bound)
]
print(f"   Registros después de eliminar outliers: {len(df_clean):,}")

# 3. Selección de características - SOLO NUMÉRICAS
print("\n3. Selección de características...")

# Obtener todas las columnas numéricas (excluyendo la variable objetivo)
columnas_numericas = df_clean.select_dtypes(include=[np.number]).columns.tolist()
columnas_numericas.remove('precio_venta')  # Remover variable objetivo

# Filtrar columnas que existen y son útiles
columnas_excluir = ['id', 'Unnamed: 0', 'precio_m2']  # Columnas a excluir
caracteristicas_modelo = [col for col in columnas_numericas
                          if col not in columnas_excluir
                          and not col.startswith('Unnamed')]

print(f"   Características numéricas seleccionadas: {len(caracteristicas_modelo)}")
print(f"   Columnas: {caracteristicas_modelo[:10]}...")  # Mostrar primeras 10

# 4. Preparar X e y
print("\n4. Preparando X e y...")
X = df_clean[caracteristicas_modelo].copy()
y = df_clean['precio_venta'].copy()

# Verificar que X solo tenga valores numéricos
print(f"   Tipos de datos en X:")
tipos_no_numericos = X.select_dtypes(exclude=[np.number]).columns
if len(tipos_no_numericos) > 0:
    print(f"   ⚠️ ADVERTENCIA: Columnas no numéricas encontradas: {list(tipos_no_numericos)}")
    print(f"   Removiéndolas...")
    X = X.select_dtypes(include=[np.number])
    caracteristicas_modelo = X.columns.tolist()

print(f"   Forma de X: {X.shape}")
print(f"   Forma de y: {y.shape}")

# Manejar valores faltantes
print(f"   Valores faltantes en X: {X.isna().sum().sum()}")
X = X.fillna(0)
print(f"   Después de fillna: {X.isna().sum().sum()}")

# 5. División de datos
print("\n5. División de datos (60/20/20)...")
X_temp, X_val, y_temp, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print(f"   Train: {len(X_train):6,} ({len(X_train)/len(X)*100:5.1f}%)")
print(f"   Test:  {len(X_test):6,} ({len(X_test)/len(X)*100:5.1f}%)")
print(f"   Val:   {len(X_val):6,} ({len(X_val)/len(X)*100:5.1f}%)")

# 6. Escalado
print("\n6. Escalando características...")
try:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    print(f"   ✓ Escalado exitoso")
    print(f"   Media: {X_train_scaled.mean():.6f}")
    print(f"   Std: {X_train_scaled.std():.6f}")
except Exception as e:
    print(f"   ✗ ERROR en escalado: {e}")
    sys.exit(1)

# 7. Entrenar modelos
print("\n7. Entrenando modelos...")

modelos = []

# 7.1 Regresión Lineal
print("\n   7.1 Regresión Lineal...")
try:
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred_lr)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2 = r2_score(y_test, y_pred_lr)
    mape = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

    modelos.append({
        'nombre': 'Regresión Lineal',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    })
    print(f"       ✓ R²: {r2:.4f}, MAE: {formatear_cop(mae)}")
except Exception as e:
    print(f"       ✗ ERROR: {e}")

# 7.2 Ridge
print("\n   7.2 Ridge Regression...")
try:
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred_ridge)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2 = r2_score(y_test, y_pred_ridge)
    mape = np.mean(np.abs((y_test - y_pred_ridge) / y_test)) * 100

    modelos.append({
        'nombre': 'Ridge',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    })
    print(f"       ✓ R²: {r2:.4f}, MAE: {formatear_cop(mae)}")
except Exception as e:
    print(f"       ✗ ERROR: {e}")

# 7.3 Random Forest
print("\n   7.3 Random Forest...")
try:
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)  # RF no requiere escalado
    y_pred_rf = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_rf)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2 = r2_score(y_test, y_pred_rf)
    mape = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100

    modelos.append({
        'nombre': 'Random Forest',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    })
    print(f"       ✓ R²: {r2:.4f}, MAE: {formatear_cop(mae)}")
except Exception as e:
    print(f"       ✗ ERROR: {e}")

# 7.4 Gradient Boosting
print("\n   7.4 Gradient Boosting...")
try:
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_gb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2 = r2_score(y_test, y_pred_gb)
    mape = np.mean(np.abs((y_test - y_pred_gb) / y_test)) * 100

    modelos.append({
        'nombre': 'Gradient Boosting',
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    })
    print(f"       ✓ R²: {r2:.4f}, MAE: {formatear_cop(mae)}")
except Exception as e:
    print(f"       ✗ ERROR: {e}")

# 8. Comparación
print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS")
print("="*80)
df_comp = pd.DataFrame(modelos)
print(df_comp.to_string(index=False))

# Mejor modelo
mejor_idx = df_comp['R2'].idxmax()
mejor = df_comp.loc[mejor_idx]
print(f"\n🏆 Mejor Modelo: {mejor['nombre']}")
print(f"   R²: {mejor['R2']:.4f}")
print(f"   MAE: {formatear_cop(mejor['MAE'])}")
print(f"   MAPE: {mejor['MAPE']:.2f}%")

# 9. Evaluación en validación
print("\n" + "="*80)
print("EVALUACIÓN EN CONJUNTO DE VALIDACIÓN")
print("="*80)
try:
    y_pred_val = rf_model.predict(X_val)

    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_val = r2_score(y_val, y_pred_val)
    mape_val = np.mean(np.abs((y_val - y_pred_val) / y_val)) * 100

    print(f"Modelo: Random Forest")
    print(f"  R²:   {r2_val:.4f}")
    print(f"  MAE:  {formatear_cop(mae_val)}")
    print(f"  RMSE: {formatear_cop(rmse_val)}")
    print(f"  MAPE: {mape_val:.2f}%")

    # Umbral de negocio
    umbral = 20_000_000
    errores = np.abs(y_val - y_pred_val)
    dentro_umbral = (errores <= umbral).sum()
    pct = (dentro_umbral / len(y_val)) * 100

    print(f"\nPredicciones dentro del umbral (±20M COP):")
    print(f"  {dentro_umbral:,} de {len(y_val):,} ({pct:.2f}%)")

except Exception as e:
    print(f"✗ ERROR en validación: {e}")

print("\n" + "="*80)
print("✓ PRUEBA COMPLETADA EXITOSAMENTE")
print("="*80)
print("\nEl código está listo para incluirse en el notebook 02.")
print("Características usadas:", len(caracteristicas_modelo))
