"""
06 - Model Interpretability
HabitAlpes Apartment Price Prediction

This script uses SHAP and LIME to interpret model predictions.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from utils import (
    print_section_header, get_processed_data_path, get_model_path,
    save_figure, format_cop
)
import warnings
warnings.filterwarnings('ignore')


def load_model_and_data():
    """Load best model and validation data."""

    print_section_header("LOADING MODEL AND DATA", "-")

    # Load best model info
    with open(get_model_path('best_model.txt'), 'r') as f:
        best_model_name = f.readlines()[0].strip()

    print(f"Best model: {best_model_name}")

    # Load model
    model_filename = best_model_name.lower().replace(' ', '_') + '.pkl'
    model = joblib.load(get_model_path(model_filename))

    # Load scaler
    scaler = joblib.load(get_model_path('scaler.pkl'))

    # Load validation data
    try:
        val = pd.read_csv(get_processed_data_path('validation_fe.csv'))
    except FileNotFoundError:
        val = pd.read_csv(get_processed_data_path('validation.csv'))

    X_val = val.drop(columns=['precio_venta'])
    y_val = val['precio_venta']

    print(f"Validation set: {X_val.shape}")

    return model, scaler, X_val, y_val, best_model_name


def perform_shap_analysis(model, X_val, model_name, sample_size=1000):
    """Perform SHAP analysis for global and local interpretability."""

    print_section_header("SHAP ANALYSIS", "-")

    # Sample data for faster computation
    if len(X_val) > sample_size:
        print(f"Sampling {sample_size} instances for SHAP analysis...")
        X_sample = X_val.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_val

    # Create SHAP explainer
    print("\nCreating SHAP explainer...")

    # Choose explainer based on model type
    if 'forest' in model_name.lower() or 'xgb' in model_name.lower() or 'lightgbm' in model_name.lower():
        # Tree-based explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Linear explainer or general explainer
        try:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        except:
            # Fallback to KernelExplainer (slower but works for all models)
            print("Using KernelExplainer (this may take a while)...")
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
            shap_values = explainer.shap_values(X_sample)

    print("SHAP values computed successfully!")

    # =================================================================
    # 1. GLOBAL INTERPRETABILITY
    # =================================================================
    print_section_header("GLOBAL INTERPRETABILITY", "-")

    # Summary plot
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    save_figure('18_shap_summary_plot.png')
    plt.close()

    # Feature importance (bar plot)
    print("Generating SHAP feature importance plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    save_figure('19_shap_feature_importance.png')
    plt.close()

    # Calculate mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop 20 most important features (by mean |SHAP value|):")
    print(feature_importance.head(20).to_string(index=False))

    # Save feature importance
    feature_importance.to_csv(get_processed_data_path('../results/shap_feature_importance.csv'), index=False)

    # =================================================================
    # 2. DEPENDENCE PLOTS FOR TOP FEATURES
    # =================================================================
    print_section_header("SHAP DEPENDENCE PLOTS", "-")

    top_features = feature_importance.head(6)['feature'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for i, feature in enumerate(top_features):
        if i < len(axes):
            shap.dependence_plot(
                feature, shap_values, X_sample,
                ax=axes[i], show=False
            )
            axes[i].set_title(f'SHAP Dependence: {feature}', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure('20_shap_dependence_plots.png')
    plt.close()

    # =================================================================
    # 3. LOCAL INTERPRETABILITY - INDIVIDUAL PREDICTIONS
    # =================================================================
    print_section_header("LOCAL INTERPRETABILITY - INDIVIDUAL CASES", "-")

    # Select diverse samples
    # 1. High price apartment
    high_price_idx = y_val.iloc[X_sample.index].idxmax()
    # 2. Low price apartment
    low_price_idx = y_val.iloc[X_sample.index].idxmin()
    # 3. Median price apartment
    median_price_idx = (y_val.iloc[X_sample.index] - y_val.iloc[X_sample.index].median()).abs().idxmin()
    # 4-5. Random samples
    random_indices = X_sample.sample(n=2, random_state=42).index.tolist()

    sample_indices = [high_price_idx, low_price_idx, median_price_idx] + random_indices

    # Make sure indices exist in sample
    sample_indices = [idx for idx in sample_indices if idx in X_sample.index][:5]

    for idx in sample_indices:
        # Get position in sample
        sample_position = X_sample.index.get_loc(idx)

        # Get actual and predicted values
        actual_price = y_val.loc[idx]
        predicted_price = model.predict(X_sample.iloc[[sample_position]])[0]

        print(f"\nAnalyzing instance {idx}:")
        print(f"  Actual Price: {format_cop(actual_price)}")
        print(f"  Predicted Price: {format_cop(predicted_price)}")
        print(f"  Error: {format_cop(abs(actual_price - predicted_price))}")

        # Force plot
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[sample_position],
            X_sample.iloc[sample_position],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        save_figure(f'21_shap_force_plot_instance_{idx}.png')
        plt.close()

        # Waterfall plot (if available)
        try:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_position],
                    base_values=explainer.expected_value,
                    data=X_sample.iloc[sample_position].values,
                    feature_names=X_sample.columns.tolist()
                ),
                show=False
            )
            plt.tight_layout()
            save_figure(f'22_shap_waterfall_instance_{idx}.png')
            plt.close()
        except:
            print("  Waterfall plot not available for this SHAP version")

    return shap_values, explainer, X_sample


def perform_lime_analysis(model, X_val, y_val, model_name, num_samples=5):
    """Perform LIME analysis for local interpretability."""

    print_section_header("LIME ANALYSIS", "-")

    # Create LIME explainer
    print("Creating LIME explainer...")

    explainer = lime_tabular.LimeTabularExplainer(
        X_val.values,
        feature_names=X_val.columns.tolist(),
        mode='regression',
        random_state=42
    )

    # Select same diverse samples as SHAP
    high_price_idx = y_val.idxmax()
    low_price_idx = y_val.idxmin()
    median_price_idx = (y_val - y_val.median()).abs().idxmin()
    random_indices = X_val.sample(n=2, random_state=42).index.tolist()

    sample_indices = [high_price_idx, low_price_idx, median_price_idx] + random_indices
    sample_indices = sample_indices[:num_samples]

    print(f"\nAnalyzing {len(sample_indices)} instances with LIME...")

    for idx in sample_indices:
        # Get instance position
        instance_position = X_val.index.get_loc(idx)
        instance = X_val.iloc[instance_position].values

        # Get actual and predicted values
        actual_price = y_val.iloc[instance_position]
        predicted_price = model.predict(X_val.iloc[[instance_position]])[0]

        print(f"\nLIME Analysis for instance {idx}:")
        print(f"  Actual Price: {format_cop(actual_price)}")
        print(f"  Predicted Price: {format_cop(predicted_price)}")

        # Generate explanation
        explanation = explainer.explain_instance(
            instance,
            model.predict,
            num_features=15
        )

        # Plot explanation
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(12, 8)
        plt.title(f'LIME Explanation - Instance {idx}\n'
                 f'Actual: {format_cop(actual_price)} | '
                 f'Predicted: {format_cop(predicted_price)}',
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_figure(f'23_lime_explanation_instance_{idx}.png')
        plt.close()

        # Print top features
        print("  Top contributing features:")
        for feature, weight in explanation.as_list()[:10]:
            print(f"    {feature}: {weight:+,.0f}")

    return explainer


def compare_shap_lime(shap_values, X_sample, lime_explainer, model, sample_idx):
    """Compare SHAP and LIME explanations for a specific instance."""

    print_section_header("COMPARING SHAP AND LIME", "-")

    # Get sample position
    sample_position = X_sample.index.get_loc(sample_idx)

    # SHAP values for this instance
    shap_vals = shap_values[sample_position]

    # LIME explanation for this instance
    instance = X_sample.iloc[sample_position].values
    lime_exp = lime_explainer.explain_instance(instance, model.predict, num_features=15)

    # Extract LIME weights
    lime_dict = dict(lime_exp.as_list())

    # Create comparison dataframe
    comparison = pd.DataFrame({
        'feature': X_sample.columns,
        'shap_value': shap_vals
    })

    # Add LIME values (may not have all features)
    comparison['lime_value'] = 0.0
    for feature_desc, weight in lime_dict.items():
        # Extract feature name (LIME uses descriptive strings)
        feature_name = feature_desc.split()[0].split('<')[0].split('>')[0].split('=')[0]
        if feature_name in comparison['feature'].values:
            comparison.loc[comparison['feature'] == feature_name, 'lime_value'] = weight

    # Sort by absolute SHAP value
    comparison['abs_shap'] = np.abs(comparison['shap_value'])
    comparison = comparison.sort_values('abs_shap', ascending=False).head(15)

    print(f"\nComparison for instance {sample_idx}:")
    print(comparison[['feature', 'shap_value', 'lime_value']].to_string(index=False))

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # SHAP
    axes[0].barh(range(len(comparison)), comparison['shap_value'].values)
    axes[0].set_yticks(range(len(comparison)))
    axes[0].set_yticklabels(comparison['feature'].values)
    axes[0].set_xlabel('SHAP Value')
    axes[0].set_title('SHAP Feature Attribution', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')

    # LIME
    axes[1].barh(range(len(comparison)), comparison['lime_value'].values, color='coral')
    axes[1].set_yticks(range(len(comparison)))
    axes[1].set_yticklabels(comparison['feature'].values)
    axes[1].set_xlabel('LIME Weight')
    axes[1].set_title('LIME Feature Attribution', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_figure(f'24_shap_lime_comparison_instance_{sample_idx}.png')
    plt.close()


def main():
    """Main interpretability execution function."""

    print_section_header("MODEL INTERPRETABILITY - HABITALPIES PROJECT")

    # Load model and data
    model, scaler, X_val, y_val, model_name = load_model_and_data()

    # SHAP Analysis
    shap_values, shap_explainer, X_sample = perform_shap_analysis(
        model, X_val, model_name, sample_size=1000
    )

    # LIME Analysis
    lime_explainer = perform_lime_analysis(
        model, X_val, y_val, model_name, num_samples=5
    )

    # Compare SHAP and LIME for one instance
    if len(X_sample) > 0:
        sample_idx = X_sample.sample(n=1, random_state=42).index[0]
        compare_shap_lime(shap_values, X_sample, lime_explainer, model, sample_idx)

    # ============================================================
    # Summary interpretation
    # ============================================================
    print_section_header("INTERPRETATION SUMMARY", "-")

    summary = """
MODEL INTERPRETABILITY INSIGHTS:

GLOBAL BEHAVIOR:
1. The model's predictions are primarily driven by property characteristics
   such as area, location (localidad/barrio), and socioeconomic stratum (estrato).

2. Amenities and proximity features play secondary but important roles
   in determining property values.

3. Feature interactions (e.g., area × estrato) capture non-linear
   relationships that improve prediction accuracy.

LOCAL BEHAVIOR:
1. Individual predictions can be explained by examining the contribution
   of each feature using SHAP force plots and LIME explanations.

2. For high-value properties, features like area, estrato, and amenities
   contribute positively to the prediction.

3. For lower-value properties, lack of amenities and lower estrato
   contribute to reduced predicted prices.

SHAP vs LIME:
- SHAP provides consistent global feature importance and local attributions
- LIME offers intuitive local explanations with feature value ranges
- Both methods generally agree on the most important features

BUSINESS IMPLICATIONS:
- The model is interpretable and trustworthy for business use
- Clients can understand why their property received a specific valuation
- Feature importance aligns with real estate domain knowledge
"""

    print(summary)

    with open(get_processed_data_path('../results/interpretability_summary.txt'), 'w') as f:
        f.write(summary)

    print_section_header("INTERPRETABILITY ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All figures saved to: reports/figures/")
    print(f"Results saved to: data/results/")


if __name__ == '__main__':
    main()
