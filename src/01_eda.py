"""
01 - Exploratory Data Analysis (EDA)
HabitAlpes Apartment Price Prediction

This script performs comprehensive exploratory data analysis on the apartment dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    load_data, print_section_header, summarize_missing_values,
    calculate_basic_stats, detect_outliers_iqr, plot_distribution,
    plot_correlation_heatmap, save_figure, get_results_path, format_cop
)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def main():
    """Main EDA execution function."""

    print_section_header("EXPLORATORY DATA ANALYSIS - HABITALPIES PROJECT")

    # =================================================================
    # 1. LOAD DATA
    # =================================================================
    print_section_header("1. LOADING DATA", "-")
    df = load_data()

    # =================================================================
    # 2. DATASET DIMENSIONS AND STRUCTURE
    # =================================================================
    print_section_header("2. DATASET DIMENSIONS", "-")
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    # Data types
    print(f"\nData types summary:")
    print(df.dtypes.value_counts())

    # =================================================================
    # 3. MISSING VALUES ANALYSIS
    # =================================================================
    print_section_header("3. MISSING VALUES ANALYSIS", "-")
    missing_summary = summarize_missing_values(df)

    if len(missing_summary) > 0:
        print("Columns with missing values:\n")
        print(missing_summary.to_string(index=False))

        # Visualize missing values
        fig, ax = plt.subplots(figsize=(12, 8))
        top_missing = missing_summary.head(20)
        ax.barh(range(len(top_missing)), top_missing['missing_pct'].values)
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels(top_missing['column'].values)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Top 20 Columns with Missing Values', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure('01_missing_values.png')
        plt.close()
    else:
        print("No missing values found in the dataset!")

    # =================================================================
    # 4. TARGET VARIABLE ANALYSIS (precio_venta)
    # =================================================================
    print_section_header("4. TARGET VARIABLE ANALYSIS: precio_venta", "-")

    # Remove missing values for analysis
    precio_venta_clean = df['precio_venta'].dropna()

    # Statistics
    calculate_basic_stats(precio_venta_clean, 'precio_venta')

    # Distribution plot
    fig = plot_distribution(df, 'precio_venta', bins=100)
    save_figure('02_precio_venta_distribution.png')
    plt.close()

    # Log-transformed distribution (to handle skewness)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(precio_venta_clean, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Precio Venta (COP)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Original Distribution')
    axes[0].ticklabel_format(style='plain', axis='x')

    axes[1].hist(np.log10(precio_venta_clean), bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('log10(Precio Venta)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-transformed Distribution')

    plt.tight_layout()
    save_figure('03_precio_venta_log_distribution.png')
    plt.close()

    # Outliers detection
    print("\nOutlier detection for precio_venta:")
    outliers_mask = detect_outliers_iqr(precio_venta_clean)

    # =================================================================
    # 5. NUMERIC FEATURES ANALYSIS
    # =================================================================
    print_section_header("5. NUMERIC FEATURES ANALYSIS", "-")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric columns:")
    print(numeric_cols)

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    desc_stats = df[numeric_cols].describe()
    print(desc_stats)

    # Save to CSV
    desc_stats.to_csv(get_results_path('numeric_descriptive_stats.csv'))

    # Key features to analyze
    key_features = ['area', 'habitaciones', 'banos', 'parqueaderos',
                    'estrato', 'piso', 'administracion', 'antiguedad']

    # Filter existing columns
    key_features = [col for col in key_features if col in df.columns]

    # Distribution of key features
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, col in enumerate(key_features):
        if i < len(axes):
            df[col].dropna().hist(bins=30, ax=axes[i], edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    save_figure('04_numeric_features_distributions.png')
    plt.close()

    # =================================================================
    # 6. CATEGORICAL FEATURES ANALYSIS
    # =================================================================
    print_section_header("6. CATEGORICAL FEATURES ANALYSIS", "-")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Found {len(categorical_cols)} categorical columns")

    # Key categorical features
    key_categorical = ['tipo_propiedad', 'tipo_operacion', 'localidad',
                       'sector', 'estado', 'compañia']

    key_categorical = [col for col in key_categorical if col in df.columns]

    # Value counts for key categorical features
    for col in key_categorical:
        print(f"\nValue counts for {col}:")
        value_counts = df[col].value_counts()
        print(value_counts.head(10))

    # Visualize top categorical features
    if 'localidad' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        localidad_counts = df['localidad'].value_counts().head(15)
        ax.barh(range(len(localidad_counts)), localidad_counts.values)
        ax.set_yticks(range(len(localidad_counts)))
        ax.set_yticklabels(localidad_counts.index)
        ax.set_xlabel('Count')
        ax.set_title('Top 15 Localidades', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure('05_top_localidades.png')
        plt.close()

    if 'tipo_propiedad' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        tipo_counts = df['tipo_propiedad'].value_counts()
        ax.pie(tipo_counts.values, labels=tipo_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Property Type Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure('06_property_type_distribution.png')
        plt.close()

    # =================================================================
    # 7. BINARY/AMENITIES FEATURES
    # =================================================================
    print_section_header("7. AMENITIES ANALYSIS", "-")

    amenities = ['jacuzzi', 'piscina', 'salon_comunal', 'terraza', 'vigilancia',
                 'chimenea', 'permite_mascotas', 'gimnasio', 'ascensor', 'conjunto_cerrado']

    amenities = [col for col in amenities if col in df.columns]

    if amenities:
        amenity_counts = {}
        for amenity in amenities:
            counts = df[amenity].value_counts()
            if 1 in counts.index or True in counts.index:
                amenity_counts[amenity] = counts.get(1, counts.get(True, 0))

        amenity_df = pd.DataFrame.from_dict(amenity_counts, orient='index', columns=['count'])
        amenity_df = amenity_df.sort_values('count', ascending=False)

        print("\nAmenities availability:")
        print(amenity_df)

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(amenity_df)), amenity_df['count'].values, color='teal')
        ax.set_yticks(range(len(amenity_df)))
        ax.set_yticklabels(amenity_df.index)
        ax.set_xlabel('Number of Properties')
        ax.set_title('Amenities Availability', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure('07_amenities_availability.png')
        plt.close()

    # =================================================================
    # 8. CORRELATION ANALYSIS
    # =================================================================
    print_section_header("8. CORRELATION ANALYSIS", "-")

    # Select numeric columns for correlation
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlation with target
    if 'precio_venta' in numeric_features:
        correlations = df[numeric_features].corr()['precio_venta'].sort_values(ascending=False)
        print("\nTop 15 features most correlated with precio_venta:")
        print(correlations.head(15))

        print("\nBottom 10 features (least correlated with precio_venta):")
        print(correlations.tail(10))

        # Plot correlation with target
        fig, ax = plt.subplots(figsize=(10, 12))
        top_corr = correlations.head(20)
        ax.barh(range(len(top_corr)), top_corr.values, color='steelblue')
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index)
        ax.set_xlabel('Correlation with precio_venta')
        ax.set_title('Top 20 Features Correlated with Price', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        save_figure('08_correlation_with_target.png')
        plt.close()

    # Full correlation heatmap (top features)
    top_features_for_heatmap = correlations.head(15).index.tolist()
    corr_matrix = plot_correlation_heatmap(df, columns=top_features_for_heatmap, figsize=(12, 10))
    save_figure('09_correlation_heatmap.png')
    plt.close()

    # =================================================================
    # 9. GEOGRAPHIC ANALYSIS
    # =================================================================
    print_section_header("9. GEOGRAPHIC ANALYSIS", "-")

    if 'latitud' in df.columns and 'longitud' in df.columns:
        # Remove missing coordinates
        df_geo = df.dropna(subset=['latitud', 'longitud', 'precio_venta'])

        print(f"Properties with valid coordinates: {len(df_geo):,}")

        # Scatter plot of properties
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(df_geo['longitud'], df_geo['latitud'],
                            c=np.log10(df_geo['precio_venta']),
                            cmap='viridis', alpha=0.5, s=10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geographic Distribution of Apartments (colored by log10(price))',
                     fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='log10(precio_venta)')
        plt.tight_layout()
        save_figure('10_geographic_distribution.png')
        plt.close()

        # Price by location
        if 'localidad' in df.columns:
            price_by_localidad = df.groupby('localidad')['precio_venta'].agg(['mean', 'median', 'count'])
            price_by_localidad = price_by_localidad.sort_values('mean', ascending=False).head(15)

            print("\nTop 15 Localidades by average price:")
            print(price_by_localidad)

            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(price_by_localidad))
            ax.bar(x, price_by_localidad['mean'].values, alpha=0.7, label='Mean')
            ax.plot(x, price_by_localidad['median'].values, 'ro-', label='Median', linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(price_by_localidad.index, rotation=45, ha='right')
            ax.set_ylabel('Price (COP)')
            ax.set_title('Average Price by Localidad (Top 15)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.ticklabel_format(style='plain', axis='y')
            plt.tight_layout()
            save_figure('11_price_by_localidad.png')
            plt.close()

    # =================================================================
    # 10. PRICE ANALYSIS BY KEY FEATURES
    # =================================================================
    print_section_header("10. PRICE ANALYSIS BY KEY FEATURES", "-")

    # Price by number of rooms
    if 'habitaciones' in df.columns:
        price_by_rooms = df.groupby('habitaciones')['precio_venta'].agg(['mean', 'median', 'count'])
        print("\nPrice by number of rooms:")
        print(price_by_rooms)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot
        df_rooms = df[df['habitaciones'].between(0, 6)]  # Filter extreme values
        df_rooms.boxplot(column='precio_venta', by='habitaciones', ax=axes[0])
        axes[0].set_xlabel('Number of Bedrooms')
        axes[0].set_ylabel('Price (COP)')
        axes[0].set_title('Price Distribution by Number of Bedrooms')
        axes[0].get_figure().suptitle('')  # Remove default title

        # Bar plot
        axes[1].bar(price_by_rooms.index, price_by_rooms['mean'].values, alpha=0.7)
        axes[1].set_xlabel('Number of Bedrooms')
        axes[1].set_ylabel('Average Price (COP)')
        axes[1].set_title('Average Price by Number of Bedrooms')
        axes[1].ticklabel_format(style='plain', axis='y')

        plt.tight_layout()
        save_figure('12_price_by_bedrooms.png')
        plt.close()

    # Price by estrato
    if 'estrato' in df.columns:
        price_by_estrato = df.groupby('estrato')['precio_venta'].agg(['mean', 'median', 'count'])
        print("\nPrice by estrato:")
        print(price_by_estrato)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = price_by_estrato.index
        ax.bar(x, price_by_estrato['mean'].values, alpha=0.7, label='Mean', color='steelblue')
        ax.plot(x, price_by_estrato['median'].values, 'ro-', label='Median', linewidth=2, markersize=8)
        ax.set_xlabel('Estrato')
        ax.set_ylabel('Price (COP)')
        ax.set_title('Price by Socioeconomic Stratum', fontsize=14, fontweight='bold')
        ax.legend()
        ax.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        save_figure('13_price_by_estrato.png')
        plt.close()

    # Price per square meter analysis
    if 'area' in df.columns:
        df['precio_m2'] = df['precio_venta'] / df['area']
        df_price_m2 = df.dropna(subset=['precio_m2'])

        print("\nPrice per square meter statistics:")
        calculate_basic_stats(df_price_m2['precio_m2'], 'precio_m2')

        fig = plot_distribution(df_price_m2, 'precio_m2', bins=100)
        save_figure('14_price_per_m2_distribution.png')
        plt.close()

    # =================================================================
    # 11. SUMMARY REPORT
    # =================================================================
    print_section_header("11. EDA SUMMARY REPORT", "-")

    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values_total': df.isnull().sum().sum(),
        'precio_venta_mean': df['precio_venta'].mean(),
        'precio_venta_median': df['precio_venta'].median(),
        'precio_venta_min': df['precio_venta'].min(),
        'precio_venta_max': df['precio_venta'].max()
    }

    print("\nDataset Summary:")
    for key, value in summary.items():
        if 'precio' in key and value is not None:
            print(f"  {key:30s}: {format_cop(value)}")
        else:
            print(f"  {key:30s}: {value:,}")

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(get_results_path('eda_summary.csv'), index=False)

    print_section_header("EDA COMPLETED SUCCESSFULLY!")
    print(f"All figures saved to: reports/figures/")
    print(f"All results saved to: data/results/")


if __name__ == '__main__':
    main()
