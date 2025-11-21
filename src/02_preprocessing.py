"""
02 - Data Preprocessing and Splitting
HabitAlpes Apartment Price Prediction

This script cleans the data, handles missing values, encodes categorical variables,
and splits the dataset into train, test, and validation sets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import (
    load_data, print_section_header, get_processed_data_path,
    summarize_missing_values
)
import warnings
warnings.filterwarnings('ignore')


def clean_data(df):
    """Clean the dataset by handling missing values and data quality issues."""

    print_section_header("DATA CLEANING", "-")

    df_clean = df.copy()

    # ========================================
    # 1. Filter relevant data
    # ========================================
    print("Filtering dataset...")

    # Keep only sales (tipo_operacion == 'VENTA')
    if 'tipo_operacion' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['tipo_operacion'] == 'VENTA']
        print(f"  - Filtered for VENTA: {before:,} -> {len(df_clean):,} records")

    # Remove records without target variable
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['precio_venta'])
    print(f"  - Removed records without precio_venta: {before:,} -> {len(df_clean):,}")

    # ========================================
    # 2. Handle outliers in target variable
    # ========================================
    print("\nHandling outliers in precio_venta...")

    # Remove extreme outliers (using percentiles)
    Q1 = df_clean['precio_venta'].quantile(0.01)
    Q99 = df_clean['precio_venta'].quantile(0.99)

    before = len(df_clean)
    df_clean = df_clean[
        (df_clean['precio_venta'] >= Q1) &
        (df_clean['precio_venta'] <= Q99)
    ]
    print(f"  - Removed extreme outliers (1st-99th percentile): {before:,} -> {len(df_clean):,}")
    print(f"  - Price range: ${Q1:,.0f} - ${Q99:,.0f}")

    # ========================================
    # 3. Handle missing values in key features
    # ========================================
    print("\nHandling missing values...")

    # For numeric features, fill with median
    numeric_features = ['area', 'habitaciones', 'banos', 'parqueaderos',
                       'estrato', 'piso', 'administracion', 'antiguedad']

    for col in numeric_features:
        if col in df_clean.columns:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                median_value = df_clean[col].median()
                df_clean[col].fillna(median_value, inplace=True)
                print(f"  - {col}: filled {missing_before:,} missing values with median ({median_value:.2f})")

    # For binary/amenity features, fill with 0 (absence)
    binary_features = ['jacuzzi', 'piscina', 'salon_comunal', 'terraza',
                      'vigilancia', 'chimenea', 'permite_mascotas',
                      'gimnasio', 'ascensor', 'conjunto_cerrado']

    for col in binary_features:
        if col in df_clean.columns:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                df_clean[col].fillna(0, inplace=True)
                print(f"  - {col}: filled {missing_before:,} missing values with 0")

    # For categorical features, fill with 'UNKNOWN'
    categorical_features = ['localidad', 'barrio', 'sector', 'estado',
                           'tipo_propiedad', 'estacion_tm_cercana', 'parque_cercano']

    for col in categorical_features:
        if col in df_clean.columns:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                df_clean[col].fillna('UNKNOWN', inplace=True)
                print(f"  - {col}: filled {missing_before:,} missing values with 'UNKNOWN'")

    # For distance features, fill with a large value (indicating far away)
    distance_features = ['distancia_estacion_tm_m', 'distancia_parque_m']

    for col in distance_features:
        if col in df_clean.columns:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                max_distance = df_clean[col].quantile(0.95)
                df_clean[col].fillna(max_distance, inplace=True)
                print(f"  - {col}: filled {missing_before:,} missing values with 95th percentile ({max_distance:.0f}m)")

    # For proximity binary features
    proximity_binary = ['is_cerca_estacion_tm', 'is_cerca_parque']
    for col in proximity_binary:
        if col in df_clean.columns:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                df_clean[col].fillna(0, inplace=True)
                print(f"  - {col}: filled {missing_before:,} missing values with 0")

    # ========================================
    # 4. Data quality checks
    # ========================================
    print("\nData quality checks...")

    # Remove records with area = 0 or negative
    if 'area' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['area'] > 0]
        removed = before - len(df_clean)
        if removed > 0:
            print(f"  - Removed {removed:,} records with area <= 0")

    # Remove records with negative prices
    before = len(df_clean)
    df_clean = df_clean[df_clean['precio_venta'] > 0]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"  - Removed {removed:,} records with precio_venta <= 0")

    # Remove unrealistic values for bedrooms (e.g., > 10)
    if 'habitaciones' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean[df_clean['habitaciones'] <= 10]
        removed = before - len(df_clean)
        if removed > 0:
            print(f"  - Removed {removed:,} records with > 10 bedrooms")

    # ========================================
    # 5. Drop unnecessary columns
    # ========================================
    print("\nDropping unnecessary columns...")

    # Columns to drop (IDs, URLs, descriptions, etc.)
    cols_to_drop = ['_id', 'codigo', 'descripcion', 'url', 'website',
                   'last_view', 'datetime', 'timeline', 'direccion',
                   'compañia', 'coords_modified', 'tipo_operacion', 'precio_arriendo']

    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]

    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"  - Dropped {len(cols_to_drop)} columns: {cols_to_drop}")

    # ========================================
    # 6. Summary of cleaned data
    # ========================================
    print("\nCleaned dataset summary:")
    print(f"  - Final shape: {df_clean.shape}")
    print(f"  - Remaining missing values: {df_clean.isnull().sum().sum()}")

    missing_summary = summarize_missing_values(df_clean)
    if len(missing_summary) > 0:
        print("\nColumns still with missing values:")
        print(missing_summary.to_string(index=False))

    return df_clean


def encode_categorical_features(df):
    """Encode categorical features for machine learning."""

    print_section_header("ENCODING CATEGORICAL FEATURES", "-")

    df_encoded = df.copy()

    # Identify categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

    print(f"Found {len(categorical_cols)} categorical columns to encode")

    # For high-cardinality features (like barrio), use frequency encoding
    high_cardinality_cols = ['barrio', 'sector', 'estacion_tm_cercana', 'parque_cercano']

    for col in high_cardinality_cols:
        if col in categorical_cols:
            freq_encoding = df_encoded[col].value_counts(normalize=True).to_dict()
            df_encoded[f'{col}_freq'] = df_encoded[col].map(freq_encoding)
            print(f"  - {col}: frequency encoded (unique values: {df_encoded[col].nunique()})")

    # For low-cardinality features, use label encoding or one-hot encoding
    low_cardinality_cols = ['tipo_propiedad', 'localidad', 'estado']

    for col in low_cardinality_cols:
        if col in categorical_cols:
            unique_values = df_encoded[col].nunique()

            if unique_values <= 10:
                # One-hot encoding for very low cardinality
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                print(f"  - {col}: one-hot encoded ({unique_values} unique values)")
            else:
                # Label encoding for medium cardinality
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                print(f"  - {col}: label encoded ({unique_values} unique values)")

    # Drop original categorical columns
    df_encoded = df_encoded.select_dtypes(exclude=['object'])

    print(f"\nFinal encoded dataset shape: {df_encoded.shape}")

    return df_encoded


def split_data(df, target_col='precio_venta', test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, test, and validation sets."""

    print_section_header("SPLITTING DATA", "-")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train vs val (from temp)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"\nSplit sizes:")
    print(f"  - Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  - Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")

    # Combine X and y for saving
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    val = pd.concat([X_val, y_val], axis=1)

    return train, test, val


def save_processed_data(train, test, val):
    """Save processed datasets."""

    print_section_header("SAVING PROCESSED DATA", "-")

    train.to_csv(get_processed_data_path('train.csv'), index=False)
    print(f"  - Train set saved: {get_processed_data_path('train.csv')}")

    test.to_csv(get_processed_data_path('test.csv'), index=False)
    print(f"  - Test set saved: {get_processed_data_path('test.csv')}")

    val.to_csv(get_processed_data_path('validation.csv'), index=False)
    print(f"  - Validation set saved: {get_processed_data_path('validation.csv')}")


def main():
    """Main preprocessing execution function."""

    print_section_header("DATA PREPROCESSING - HABITALPIES PROJECT")

    # Load data
    print_section_header("LOADING RAW DATA", "-")
    df = load_data()

    # Clean data
    df_clean = clean_data(df)

    # Encode categorical features
    df_encoded = encode_categorical_features(df_clean)

    # Split data
    train, test, val = split_data(df_encoded)

    # Save processed data
    save_processed_data(train, test, val)

    print_section_header("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"Processed data saved to: data/processed/")


if __name__ == '__main__':
    main()
