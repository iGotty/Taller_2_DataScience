"""
03 - Feature Engineering
HabitAlpes Apartment Price Prediction

This script creates additional features from the preprocessed data
to improve model performance.
"""

import numpy as np
import pandas as pd
from utils import (
    print_section_header, get_processed_data_path
)
import warnings
warnings.filterwarnings('ignore')


def create_engineered_features(df):
    """Create new features from existing ones."""

    print_section_header("FEATURE ENGINEERING", "-")

    df_fe = df.copy()
    features_created = []

    # ========================================
    # 1. Price-related features
    # ========================================
    print("Creating price-related features...")

    if 'area' in df_fe.columns and df_fe['area'].sum() > 0:
        # Price per square meter (already in preprocessing, but ensure it's there)
        if 'precio_venta' in df_fe.columns:
            df_fe['precio_m2'] = df_fe['precio_venta'] / df_fe['area']
            features_created.append('precio_m2')
            print("  - precio_m2: price per square meter")

    # ========================================
    # 2. Space efficiency features
    # ========================================
    print("\nCreating space efficiency features...")

    if 'area' in df_fe.columns:
        # Area per room
        if 'habitaciones' in df_fe.columns:
            df_fe['area_per_room'] = df_fe['area'] / (df_fe['habitaciones'] + 1)  # +1 to avoid division by zero
            features_created.append('area_per_room')
            print("  - area_per_room")

        # Total rooms
        total_rooms_cols = [c for c in ['habitaciones', 'banos'] if c in df_fe.columns]
        if total_rooms_cols:
            df_fe['total_rooms'] = df_fe[total_rooms_cols].sum(axis=1)
            features_created.append('total_rooms')
            print("  - total_rooms")

    # Bathroom ratio
    if 'banos' in df_fe.columns and 'habitaciones' in df_fe.columns:
        df_fe['bath_to_bed_ratio'] = df_fe['banos'] / (df_fe['habitaciones'] + 1)
        features_created.append('bath_to_bed_ratio')
        print("  - bath_to_bed_ratio")

    # ========================================
    # 3. Property quality score
    # ========================================
    print("\nCreating property quality score...")

    # Amenities score (sum of all amenities)
    amenity_cols = [col for col in df_fe.columns if col in [
        'jacuzzi', 'piscina', 'salon_comunal', 'terraza', 'vigilancia',
        'chimenea', 'permite_mascotas', 'gimnasio', 'ascensor', 'conjunto_cerrado'
    ]]

    if amenity_cols:
        df_fe['amenities_score'] = df_fe[amenity_cols].sum(axis=1)
        features_created.append('amenities_score')
        print(f"  - amenities_score (from {len(amenity_cols)} amenities)")

    # ========================================
    # 4. Location quality features
    # ========================================
    print("\nCreating location quality features...")

    # Distance features (inverse to make closer = higher value)
    if 'distancia_estacion_tm_m' in df_fe.columns:
        # Proximity score (closer = higher score)
        df_fe['proximity_transit_score'] = 1 / (df_fe['distancia_estacion_tm_m'] / 1000 + 1)  # +1 to avoid div by zero
        features_created.append('proximity_transit_score')
        print("  - proximity_transit_score")

    if 'distancia_parque_m' in df_fe.columns:
        df_fe['proximity_park_score'] = 1 / (df_fe['distancia_parque_m'] / 1000 + 1)
        features_created.append('proximity_park_score')
        print("  - proximity_park_score")

    # Combined proximity score
    if 'proximity_transit_score' in df_fe.columns and 'proximity_park_score' in df_fe.columns:
        df_fe['combined_proximity_score'] = (
            df_fe['proximity_transit_score'] * 0.6 +
            df_fe['proximity_park_score'] * 0.4
        )
        features_created.append('combined_proximity_score')
        print("  - combined_proximity_score")

    # ========================================
    # 5. Interaction features
    # ========================================
    print("\nCreating interaction features...")

    # Area × Estrato (high-estrato properties might value space more)
    if 'area' in df_fe.columns and 'estrato' in df_fe.columns:
        df_fe['area_x_estrato'] = df_fe['area'] * df_fe['estrato']
        features_created.append('area_x_estrato')
        print("  - area_x_estrato")

    # Rooms × Estrato
    if 'habitaciones' in df_fe.columns and 'estrato' in df_fe.columns:
        df_fe['rooms_x_estrato'] = df_fe['habitaciones'] * df_fe['estrato']
        features_created.append('rooms_x_estrato')
        print("  - rooms_x_estrato")

    # Amenities × Estrato
    if 'amenities_score' in df_fe.columns and 'estrato' in df_fe.columns:
        df_fe['amenities_x_estrato'] = df_fe['amenities_score'] * df_fe['estrato']
        features_created.append('amenities_x_estrato')
        print("  - amenities_x_estrato")

    # ========================================
    # 6. Floor-related features
    # ========================================
    print("\nCreating floor-related features...")

    if 'piso' in df_fe.columns:
        # Binary: high floor (> 5)
        df_fe['is_high_floor'] = (df_fe['piso'] > 5).astype(int)
        features_created.append('is_high_floor')
        print("  - is_high_floor")

        # Binary: ground floor
        df_fe['is_ground_floor'] = (df_fe['piso'] == 0).astype(int)
        features_created.append('is_ground_floor')
        print("  - is_ground_floor")

    # ========================================
    # 7. Age-related features
    # ========================================
    print("\nCreating age-related features...")

    if 'antiguedad' in df_fe.columns:
        # Binary: new property (< 5 years)
        df_fe['is_new'] = (df_fe['antiguedad'] < 5).astype(int)
        features_created.append('is_new')
        print("  - is_new")

        # Binary: old property (> 20 years)
        df_fe['is_old'] = (df_fe['antiguedad'] > 20).astype(int)
        features_created.append('is_old')
        print("  - is_old")

    # ========================================
    # 8. Parking features
    # ========================================
    print("\nCreating parking features...")

    if 'parqueaderos' in df_fe.columns:
        # Binary: has parking
        df_fe['has_parking'] = (df_fe['parqueaderos'] > 0).astype(int)
        features_created.append('has_parking')
        print("  - has_parking")

        # Binary: multiple parking
        df_fe['multiple_parking'] = (df_fe['parqueaderos'] > 1).astype(int)
        features_created.append('multiple_parking')
        print("  - multiple_parking")

    # ========================================
    # 9. Luxury indicator
    # ========================================
    print("\nCreating luxury indicator...")

    # Define luxury based on multiple factors
    luxury_conditions = []

    if 'estrato' in df_fe.columns:
        luxury_conditions.append(df_fe['estrato'] >= 5)

    if 'area' in df_fe.columns:
        luxury_conditions.append(df_fe['area'] >= df_fe['area'].quantile(0.75))

    if 'amenities_score' in df_fe.columns:
        luxury_conditions.append(df_fe['amenities_score'] >= 3)

    if luxury_conditions:
        df_fe['is_luxury'] = sum(luxury_conditions).astype(int)
        features_created.append('is_luxury')
        print("  - is_luxury (composite score)")

    # ========================================
    # 10. Handle infinite and missing values
    # ========================================
    print("\nHandling infinite and missing values in new features...")

    # Replace infinite values with NaN
    df_fe = df_fe.replace([np.inf, -np.inf], np.nan)

    # Fill NaN in new features with median
    for feature in features_created:
        if feature in df_fe.columns:
            missing_count = df_fe[feature].isnull().sum()
            if missing_count > 0:
                median_val = df_fe[feature].median()
                df_fe[feature].fillna(median_val, inplace=True)
                print(f"  - {feature}: filled {missing_count} NaN with median")

    # ========================================
    # Summary
    # ========================================
    print(f"\nTotal new features created: {len(features_created)}")
    print(f"Final dataset shape: {df_fe.shape}")

    return df_fe, features_created


def save_engineered_data(train, test, val, suffix='_fe'):
    """Save feature-engineered datasets."""

    print_section_header("SAVING FEATURE-ENGINEERED DATA", "-")

    train.to_csv(get_processed_data_path(f'train{suffix}.csv'), index=False)
    print(f"  - Train set saved: {get_processed_data_path(f'train{suffix}.csv')}")

    test.to_csv(get_processed_data_path(f'test{suffix}.csv'), index=False)
    print(f"  - Test set saved: {get_processed_data_path(f'test{suffix}.csv')}")

    val.to_csv(get_processed_data_path(f'validation{suffix}.csv'), index=False)
    print(f"  - Validation set saved: {get_processed_data_path(f'validation{suffix}.csv')}")


def main():
    """Main feature engineering execution function."""

    print_section_header("FEATURE ENGINEERING - HABITALPIES PROJECT")

    # Load preprocessed data
    print_section_header("LOADING PREPROCESSED DATA", "-")

    train = pd.read_csv(get_processed_data_path('train.csv'))
    print(f"Train set loaded: {train.shape}")

    test = pd.read_csv(get_processed_data_path('test.csv'))
    print(f"Test set loaded: {test.shape}")

    val = pd.read_csv(get_processed_data_path('validation.csv'))
    print(f"Validation set loaded: {val.shape}")

    # Create engineered features for each set
    print_section_header("ENGINEERING FEATURES FOR TRAIN SET", "-")
    train_fe, features_list = create_engineered_features(train)

    print_section_header("ENGINEERING FEATURES FOR TEST SET", "-")
    test_fe, _ = create_engineered_features(test)

    print_section_header("ENGINEERING FEATURES FOR VALIDATION SET", "-")
    val_fe, _ = create_engineered_features(val)

    # Ensure all sets have the same features
    print_section_header("ALIGNING FEATURES ACROSS SETS", "-")

    train_cols = set(train_fe.columns)
    test_cols = set(test_fe.columns)
    val_cols = set(val_fe.columns)

    common_cols = train_cols & test_cols & val_cols

    print(f"Train columns: {len(train_cols)}")
    print(f"Test columns: {len(test_cols)}")
    print(f"Validation columns: {len(val_cols)}")
    print(f"Common columns: {len(common_cols)}")

    # Keep only common columns
    train_fe = train_fe[sorted(common_cols)]
    test_fe = test_fe[sorted(common_cols)]
    val_fe = val_fe[sorted(common_cols)]

    print(f"\nAligned shapes:")
    print(f"  - Train: {train_fe.shape}")
    print(f"  - Test: {test_fe.shape}")
    print(f"  - Validation: {val_fe.shape}")

    # Save engineered data
    save_engineered_data(train_fe, test_fe, val_fe)

    print_section_header("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print(f"Feature-engineered data saved to: data/processed/")
    print(f"\nNew features created: {len(features_list)}")
    print("Features list:", features_list)


if __name__ == '__main__':
    main()
