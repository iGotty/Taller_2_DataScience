"""
05 - Model Evaluation
HabitAlpes Apartment Price Prediction

This script performs comprehensive quantitative evaluation of the best model
using the validation dataset.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import (
    print_section_header, get_processed_data_path, get_model_path,
    save_figure, save_results, format_cop
)
import warnings
warnings.filterwarnings('ignore')


def load_best_model():
    """Load the best model from training."""

    print_section_header("LOADING BEST MODEL", "-")

    # Read best model name
    with open(get_model_path('best_model.txt'), 'r') as f:
        lines = f.readlines()
        best_model_name = lines[0].strip()

    print(f"Best model: {best_model_name}")

    # Load model
    model_filename = best_model_name.lower().replace(' ', '_') + '.pkl'
    model = joblib.load(get_model_path(model_filename))

    print(f"Model loaded: {model_filename}")

    # Load scaler
    scaler = joblib.load(get_model_path('scaler.pkl'))
    print("Scaler loaded")

    return model, scaler, best_model_name


def load_validation_data():
    """Load validation dataset."""

    print_section_header("LOADING VALIDATION DATA", "-")

    # Try feature-engineered data first
    try:
        val = pd.read_csv(get_processed_data_path('validation_fe.csv'))
        print("Loaded feature-engineered validation set")
    except FileNotFoundError:
        val = pd.read_csv(get_processed_data_path('validation.csv'))
        print("Loaded preprocessed validation set")

    print(f"Validation set shape: {val.shape}")

    return val


def calculate_detailed_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""

    print_section_header("CALCULATING METRICS", "-")

    # Standard regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Additional metrics
    median_ae = np.median(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))

    # Percentage within thresholds
    threshold_10m = np.sum(np.abs(y_true - y_pred) <= 10_000_000) / len(y_true) * 100
    threshold_20m = np.sum(np.abs(y_true - y_pred) <= 20_000_000) / len(y_true) * 100
    threshold_30m = np.sum(np.abs(y_true - y_pred) <= 30_000_000) / len(y_true) * 100

    # Underestimation vs overestimation
    underestimations = np.sum(y_pred < y_true)
    overestimations = np.sum(y_pred > y_true)
    perfect = np.sum(y_pred == y_true)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Median_AE': median_ae,
        'Max_Error': max_error,
        'Within_10M_%': threshold_10m,
        'Within_20M_%': threshold_20m,
        'Within_30M_%': threshold_30m,
        'Underestimations': underestimations,
        'Overestimations': overestimations,
        'Perfect_Predictions': perfect
    }

    # Print metrics
    print("\nRegression Metrics:")
    print("-" * 50)
    print(f"MAE (Mean Absolute Error):        {format_cop(mae)}")
    print(f"RMSE (Root Mean Squared Error):   {format_cop(rmse)}")
    print(f"R² Score:                          {r2:.4f}")
    print(f"MAPE (Mean Absolute % Error):      {mape:.2f}%")
    print(f"Median Absolute Error:             {format_cop(median_ae)}")
    print(f"Maximum Error:                     {format_cop(max_error)}")

    print("\nBusiness-Relevant Metrics:")
    print("-" * 50)
    print(f"Predictions within ±10M COP:       {threshold_10m:.2f}%")
    print(f"Predictions within ±20M COP:       {threshold_20m:.2f}%")
    print(f"Predictions within ±30M COP:       {threshold_30m:.2f}%")

    print("\nPrediction Distribution:")
    print("-" * 50)
    print(f"Underestimations (pred < true):    {underestimations:,} ({underestimations/len(y_true)*100:.2f}%)")
    print(f"Overestimations (pred > true):     {overestimations:,} ({overestimations/len(y_true)*100:.2f}%)")
    print(f"Perfect predictions:                {perfect:,} ({perfect/len(y_true)*100:.2f}%)")

    return metrics


def plot_actual_vs_predicted(y_true, y_pred):
    """Plot actual vs predicted values."""

    print_section_header("PLOTTING ACTUAL VS PREDICTED", "-")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price (COP)')
    axes[0].set_ylabel('Predicted Price (COP)')
    axes[0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='plain')

    # Hexbin plot for density
    axes[1].hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
    axes[1].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Price (COP)')
    axes[1].set_ylabel('Predicted Price (COP)')
    axes[1].set_title('Actual vs Predicted (Density)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].ticklabel_format(style='plain')

    plt.tight_layout()
    save_figure('15_actual_vs_predicted.png')
    plt.close()


def plot_residuals(y_true, y_pred):
    """Plot residual analysis."""

    print_section_header("PLOTTING RESIDUALS", "-")

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Price (COP)')
    axes[0, 0].set_ylabel('Residuals (COP)')
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='plain')

    # Residuals distribution
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals (COP)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals vs Actual
    axes[1, 1].scatter(y_true, residuals, alpha=0.5, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Actual Price (COP)')
    axes[1, 1].set_ylabel('Residuals (COP)')
    axes[1, 1].set_title('Residuals vs Actual', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='plain')

    plt.tight_layout()
    save_figure('16_residual_analysis.png')
    plt.close()


def plot_error_distribution(y_true, y_pred):
    """Plot error distribution by price ranges."""

    print_section_header("PLOTTING ERROR DISTRIBUTION", "-")

    # Calculate errors
    errors = np.abs(y_true - y_pred)
    percentage_errors = (errors / y_true) * 100

    # Create price bins
    price_bins = pd.qcut(y_true, q=10, duplicates='drop')

    # Calculate mean error per bin
    error_by_bin = pd.DataFrame({
        'price_bin': price_bins,
        'error': errors,
        'pct_error': percentage_errors
    })

    mean_error_by_bin = error_by_bin.groupby('price_bin').agg({
        'error': 'mean',
        'pct_error': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute error by price range
    x_labels = [f"{int(interval.left/1e6)}-{int(interval.right/1e6)}M"
                for interval in mean_error_by_bin['price_bin']]

    axes[0].bar(range(len(mean_error_by_bin)), mean_error_by_bin['error'], alpha=0.7)
    axes[0].set_xticks(range(len(mean_error_by_bin)))
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].set_xlabel('Price Range (COP)')
    axes[0].set_ylabel('Mean Absolute Error (COP)')
    axes[0].set_title('Error by Price Range', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].ticklabel_format(style='plain', axis='y')

    # Percentage error by price range
    axes[1].bar(range(len(mean_error_by_bin)), mean_error_by_bin['pct_error'], alpha=0.7, color='coral')
    axes[1].set_xticks(range(len(mean_error_by_bin)))
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[1].set_xlabel('Price Range (COP)')
    axes[1].set_ylabel('Mean Percentage Error (%)')
    axes[1].set_title('Percentage Error by Price Range', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure('17_error_by_price_range.png')
    plt.close()


def analyze_extreme_errors(y_true, y_pred, X, top_n=20):
    """Analyze cases with largest errors."""

    print_section_header("ANALYZING EXTREME ERRORS", "-")

    # Calculate errors
    errors = np.abs(y_true - y_pred)

    # Get top N errors
    error_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': errors,
        'pct_error': (errors / y_true) * 100
    })

    error_df = error_df.sort_values('error', ascending=False)

    print(f"\nTop {top_n} largest errors:")
    print(error_df.head(top_n).to_string())

    # Save to CSV
    error_df.to_csv(get_processed_data_path('../results/error_analysis.csv'), index=False)

    return error_df


def main():
    """Main evaluation execution function."""

    print_section_header("MODEL EVALUATION - HABITALPIES PROJECT")

    # Load model
    model, scaler, model_name = load_best_model()

    # Load validation data
    val = load_validation_data()

    # Prepare data
    X_val = val.drop(columns=['precio_venta'])
    y_val = val['precio_venta']

    # Check if model needs scaling
    needs_scaling = model_name in ['Linear Regression', 'Ridge Regression', 'Lasso']

    if needs_scaling:
        print(f"\nScaling features for {model_name}...")
        X_val_processed = scaler.transform(X_val)
    else:
        print(f"\nUsing original features for {model_name}...")
        X_val_processed = X_val

    # Make predictions
    print_section_header("MAKING PREDICTIONS", "-")
    y_pred = model.predict(X_val_processed)
    print(f"Generated {len(y_pred):,} predictions")

    # Calculate metrics
    metrics = calculate_detailed_metrics(y_val, y_pred)

    # Visualizations
    plot_actual_vs_predicted(y_val, y_pred)
    plot_residuals(y_val, y_pred)
    plot_error_distribution(y_val, y_pred)

    # Error analysis
    error_analysis = analyze_extreme_errors(y_val, y_pred, X_val)

    # Save evaluation results
    print_section_header("SAVING EVALUATION RESULTS", "-")

    eval_results = pd.DataFrame([metrics])
    eval_results.to_csv(get_processed_data_path('../results/validation_metrics.csv'), index=False)
    print(f"Validation metrics saved")

    # Create summary report
    summary = f"""
=============================================================================
MODEL EVALUATION SUMMARY - {model_name}
=============================================================================

VALIDATION SET PERFORMANCE:
  - Dataset size: {len(y_val):,} apartments
  - MAE: {format_cop(metrics['MAE'])}
  - RMSE: {format_cop(metrics['RMSE'])}
  - R²: {metrics['R2']:.4f}
  - MAPE: {metrics['MAPE']:.2f}%

BUSINESS METRICS:
  - Predictions within ±10M: {metrics['Within_10M_%']:.2f}%
  - Predictions within ±20M: {metrics['Within_20M_%']:.2f}%
  - Predictions within ±30M: {metrics['Within_30M_%']:.2f}%

PREDICTION DISTRIBUTION:
  - Underestimations: {metrics['Underestimations']:,} ({metrics['Underestimations']/len(y_val)*100:.2f}%)
  - Overestimations: {metrics['Overestimations']:,} ({metrics['Overestimations']/len(y_val)*100:.2f}%)

MODEL QUALITY ASSESSMENT:
  - R² of {metrics['R2']:.4f} indicates the model explains {metrics['R2']*100:.2f}% of variance
  - MAPE of {metrics['MAPE']:.2f}% shows average percentage error
  - {metrics['Within_20M_%']:.2f}% of predictions are within ±20M COP threshold

=============================================================================
"""

    print(summary)

    with open(get_processed_data_path('../results/evaluation_summary.txt'), 'w') as f:
        f.write(summary)

    print_section_header("EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"All results saved to: data/results/")
    print(f"All figures saved to: reports/figures/")


if __name__ == '__main__':
    main()
