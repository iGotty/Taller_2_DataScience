"""
04 - Model Training
HabitAlpes Apartment Price Prediction

This script trains multiple regression models and selects the best one.
Uses train set for training, test set for model selection.
"""

import numpy as np
import pandas as pd
import joblib
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from utils import (
    print_section_header, get_processed_data_path, get_model_path,
    print_model_metrics, save_results, create_results_dataframe
)
import warnings
warnings.filterwarnings('ignore')


def load_datasets():
    """Load train, test, and validation datasets."""

    print_section_header("LOADING DATASETS", "-")

    # Try to load feature-engineered datasets first
    try:
        train = pd.read_csv(get_processed_data_path('train_fe.csv'))
        test = pd.read_csv(get_processed_data_path('test_fe.csv'))
        val = pd.read_csv(get_processed_data_path('validation_fe.csv'))
        print("Loaded feature-engineered datasets")
    except FileNotFoundError:
        print("Feature-engineered datasets not found, loading preprocessed datasets")
        train = pd.read_csv(get_processed_data_path('train.csv'))
        test = pd.read_csv(get_processed_data_path('test.csv'))
        val = pd.read_csv(get_processed_data_path('validation.csv'))

    print(f"  - Train: {train.shape}")
    print(f"  - Test: {test.shape}")
    print(f"  - Validation: {val.shape}")

    return train, test, val


def prepare_data(train, test, val, target_col='precio_venta'):
    """Prepare X and y for training."""

    print_section_header("PREPARING DATA", "-")

    # Separate features and target
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]

    print(f"Train features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")
    print(f"Validation features: {X_val.shape}")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)

    # Convert back to DataFrame to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

    # Save scaler
    joblib.dump(scaler, get_model_path('scaler.pkl'))
    print(f"Scaler saved: {get_model_path('scaler.pkl')}")

    return (X_train, X_train_scaled, y_train,
            X_test, X_test_scaled, y_test,
            X_val, X_val_scaled, y_val)


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train Linear Regression baseline model."""

    print_section_header("TRAINING LINEAR REGRESSION", "-")

    start_time = time.time()

    model = LinearRegression()
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = print_model_metrics(y_train, y_pred_train, "Linear Regression (Train)")
    test_metrics = print_model_metrics(y_test, y_pred_test, "Linear Regression (Test)")

    print(f"\nTraining time: {training_time:.2f} seconds")

    return model, {**test_metrics, 'training_time': training_time}


def train_ridge_regression(X_train, y_train, X_test, y_test):
    """Train Ridge Regression with hyperparameter tuning."""

    print_section_header("TRAINING RIDGE REGRESSION", "-")

    start_time = time.time()

    # Hyperparameter grid
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }

    ridge = Ridge(random_state=42)
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    training_time = time.time() - start_time

    print(f"Best parameters: {grid_search.best_params_}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = print_model_metrics(y_train, y_pred_train, "Ridge Regression (Train)")
    test_metrics = print_model_metrics(y_test, y_pred_test, "Ridge Regression (Test)")

    print(f"\nTraining time: {training_time:.2f} seconds")

    return model, {**test_metrics, 'training_time': training_time}


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning."""

    print_section_header("TRAINING RANDOM FOREST", "-")

    start_time = time.time()

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Use RandomizedSearchCV for faster training
    random_search = RandomizedSearchCV(
        rf, param_grid, n_iter=10, cv=3,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_
    training_time = time.time() - start_time

    print(f"Best parameters: {random_search.best_params_}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = print_model_metrics(y_train, y_pred_train, "Random Forest (Train)")
    test_metrics = print_model_metrics(y_test, y_pred_test, "Random Forest (Test)")

    print(f"\nTraining time: {training_time:.2f} seconds")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

    return model, {**test_metrics, 'training_time': training_time}


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with hyperparameter tuning."""

    print_section_header("TRAINING GRADIENT BOOSTING", "-")

    start_time = time.time()

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }

    gb = GradientBoostingRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        gb, param_grid, n_iter=8, cv=3,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_
    training_time = time.time() - start_time

    print(f"Best parameters: {random_search.best_params_}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = print_model_metrics(y_train, y_pred_train, "Gradient Boosting (Train)")
    test_metrics = print_model_metrics(y_test, y_pred_test, "Gradient Boosting (Test)")

    print(f"\nTraining time: {training_time:.2f} seconds")

    return model, {**test_metrics, 'training_time': training_time}


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning."""

    print_section_header("TRAINING XGBOOST", "-")

    start_time = time.time()

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=12, cv=3,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_
    training_time = time.time() - start_time

    print(f"Best parameters: {random_search.best_params_}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = print_model_metrics(y_train, y_pred_train, "XGBoost (Train)")
    test_metrics = print_model_metrics(y_test, y_pred_test, "XGBoost (Test)")

    print(f"\nTraining time: {training_time:.2f} seconds")

    return model, {**test_metrics, 'training_time': training_time}


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with hyperparameter tuning."""

    print_section_header("TRAINING LIGHTGBM", "-")

    start_time = time.time()

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50],
        'subsample': [0.8, 1.0]
    }

    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

    random_search = RandomizedSearchCV(
        lgb_model, param_grid, n_iter=12, cv=3,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    model = random_search.best_estimator_
    training_time = time.time() - start_time

    print(f"Best parameters: {random_search.best_params_}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = print_model_metrics(y_train, y_pred_train, "LightGBM (Train)")
    test_metrics = print_model_metrics(y_test, y_pred_test, "LightGBM (Test)")

    print(f"\nTraining time: {training_time:.2f} seconds")

    return model, {**test_metrics, 'training_time': training_time}


def select_best_model(results):
    """Select the best model based on test metrics."""

    print_section_header("MODEL SELECTION", "-")

    results_df = create_results_dataframe(results)
    print("\nModel Comparison:")
    print(results_df.to_string())

    # Select based on lowest MAE (or you can use R2, RMSE, etc.)
    best_model_name = results_df['MAE'].idxmin()
    best_metrics = results_df.loc[best_model_name]

    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*80}")
    print(f"MAE: ${best_metrics['MAE']:,.0f}")
    print(f"RMSE: ${best_metrics['RMSE']:,.0f}")
    print(f"R²: {best_metrics['R2']:.4f}")
    print(f"MAPE: {best_metrics['MAPE']:.2f}%")
    print(f"Training time: {best_metrics['training_time']:.2f} seconds")
    print(f"{'='*80}")

    return best_model_name, results_df


def main():
    """Main modeling execution function."""

    print_section_header("MODEL TRAINING - HABITALPIES PROJECT")

    # Load datasets
    train, test, val = load_datasets()

    # Prepare data
    (X_train, X_train_scaled, y_train,
     X_test, X_test_scaled, y_test,
     X_val, X_val_scaled, y_val) = prepare_data(train, test, val)

    # Dictionary to store models and results
    models = {}
    results = {}

    # ============================================================
    # Train models (using scaled data for linear models,
    # original data for tree-based models)
    # ============================================================

    # 1. Linear Regression (baseline)
    model_lr, metrics_lr = train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    models['Linear Regression'] = model_lr
    results['Linear Regression'] = metrics_lr

    # 2. Ridge Regression
    model_ridge, metrics_ridge = train_ridge_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    models['Ridge Regression'] = model_ridge
    results['Ridge Regression'] = metrics_ridge

    # 3. Random Forest (use original data, tree models don't need scaling)
    model_rf, metrics_rf = train_random_forest(X_train, y_train, X_test, y_test)
    models['Random Forest'] = model_rf
    results['Random Forest'] = metrics_rf

    # 4. Gradient Boosting
    model_gb, metrics_gb = train_gradient_boosting(X_train, y_train, X_test, y_test)
    models['Gradient Boosting'] = model_gb
    results['Gradient Boosting'] = metrics_gb

    # 5. XGBoost
    model_xgb, metrics_xgb = train_xgboost(X_train, y_train, X_test, y_test)
    models['XGBoost'] = model_xgb
    results['XGBoost'] = metrics_xgb

    # 6. LightGBM
    model_lgb, metrics_lgb = train_lightgbm(X_train, y_train, X_test, y_test)
    models['LightGBM'] = model_lgb
    results['LightGBM'] = metrics_lgb

    # ============================================================
    # Select best model
    # ============================================================
    best_model_name, results_df = select_best_model(results)

    # ============================================================
    # Save results and best model
    # ============================================================
    print_section_header("SAVING MODELS AND RESULTS", "-")

    # Save all models
    for model_name, model in models.items():
        model_filename = model_name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, get_model_path(model_filename))
        print(f"  - Saved: {model_filename}")

    # Save results
    save_results(results, 'model_comparison.csv')

    # Mark the best model
    with open(get_model_path('best_model.txt'), 'w') as f:
        f.write(f"{best_model_name}\n")
        f.write(f"MAE: ${results[best_model_name]['MAE']:,.0f}\n")
        f.write(f"RMSE: ${results[best_model_name]['RMSE']:,.0f}\n")
        f.write(f"R²: {results[best_model_name]['R2']:.4f}\n")
        f.write(f"MAPE: {results[best_model_name]['MAPE']:.2f}%\n")

    print(f"\nBest model info saved: {get_model_path('best_model.txt')}")

    print_section_header("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"All models saved to: models/")
    print(f"Results saved to: data/results/model_comparison.csv")


if __name__ == '__main__':
    main()
