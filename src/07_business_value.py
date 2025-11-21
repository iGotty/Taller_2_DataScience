"""
07 - Business Value Analysis
HabitAlpes Apartment Price Prediction

This script calculates ROI, break-even point, and business value metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    print_section_header, get_processed_data_path,
    save_figure, format_cop
)
import warnings
warnings.filterwarnings('ignore')


def load_evaluation_metrics():
    """Load evaluation metrics from validation."""

    print_section_header("LOADING EVALUATION METRICS", "-")

    metrics = pd.read_csv(get_processed_data_path('../results/validation_metrics.csv'))

    print("Validation Metrics:")
    print(metrics.to_string(index=False))

    return metrics.iloc[0].to_dict()


def define_business_parameters():
    """Define business parameters for value calculation."""

    print_section_header("BUSINESS PARAMETERS", "-")

    params = {
        # Expert costs
        'hourly_rate': 9_500,  # COP per hour
        'hours_without_ml': 6,  # Hours per valuation without ML
        'hours_with_ml': 1,     # Hours per valuation with ML

        # Capacity
        'monthly_capacity': 500,  # Max apartments per month

        # Error thresholds
        'underestimation_threshold': 20_000_000,  # COP

        # Development costs (estimates)
        'dev_hours': 160,  # Hours to develop the model (1 month, 1 person)
        'dev_hourly_rate': 25_000,  # COP per hour for data scientist

        # Deployment costs (estimates)
        'infrastructure_monthly': 200_000,  # Server costs per month
        'maintenance_hours_monthly': 20,  # Hours per month for maintenance
        'maintenance_hourly_rate': 25_000,  # COP per hour

        # Assumptions
        'adoption_rate': 0.8,  # 80% of valuations will use ML
        'manual_review_cost_multiplier': 1.5  # Cost multiplier for manual reviews triggered by errors
    }

    print("\nBusiness Parameters:")
    print("-" * 70)
    print(f"Expert hourly rate:               {format_cop(params['hourly_rate'])}")
    print(f"Hours without ML:                 {params['hours_without_ml']} hours")
    print(f"Hours with ML:                    {params['hours_with_ml']} hour(s)")
    print(f"Monthly capacity:                 {params['monthly_capacity']} apartments")
    print(f"Underestimation threshold:        {format_cop(params['underestimation_threshold'])}")
    print(f"\nDevelopment costs:")
    print(f"  Development hours:              {params['dev_hours']}")
    print(f"  Dev hourly rate:                {format_cop(params['dev_hourly_rate'])}")
    print(f"  Total dev cost:                 {format_cop(params['dev_hours'] * params['dev_hourly_rate'])}")
    print(f"\nOperational costs (monthly):")
    print(f"  Infrastructure:                 {format_cop(params['infrastructure_monthly'])}")
    print(f"  Maintenance hours:              {params['maintenance_hours_monthly']}")
    print(f"  Maintenance cost:               {format_cop(params['maintenance_hours_monthly'] * params['maintenance_hourly_rate'])}")
    print(f"  Total monthly operational:      {format_cop(params['infrastructure_monthly'] + params['maintenance_hours_monthly'] * params['maintenance_hourly_rate'])}")
    print(f"\nAssumptions:")
    print(f"  Adoption rate:                  {params['adoption_rate']*100:.0f}%")

    return params


def calculate_current_costs(params):
    """Calculate current costs without ML."""

    print_section_header("CURRENT COSTS (WITHOUT ML)", "-")

    cost_per_valuation = params['hourly_rate'] * params['hours_without_ml']
    monthly_cost = cost_per_valuation * params['monthly_capacity']
    annual_cost = monthly_cost * 12

    current_costs = {
        'cost_per_valuation': cost_per_valuation,
        'monthly_cost': monthly_cost,
        'annual_cost': annual_cost
    }

    print(f"Cost per valuation:   {format_cop(cost_per_valuation)}")
    print(f"Monthly cost:         {format_cop(monthly_cost)}")
    print(f"Annual cost:          {format_cop(annual_cost)}")

    return current_costs


def calculate_ml_costs(params, metrics):
    """Calculate costs with ML including error costs."""

    print_section_header("COSTS WITH ML", "-")

    # Base cost per valuation (expert review time)
    base_cost_per_valuation = params['hourly_rate'] * params['hours_with_ml']

    # Error analysis
    underestimation_rate = metrics['Underestimations'] / (
        metrics['Underestimations'] + metrics['Overestimations'] + metrics['Perfect_Predictions']
    )

    # Assume that underestimations > threshold trigger manual review
    # Estimate percentage of underestimations that exceed threshold
    # Using MAPE and distribution assumptions
    severe_underestimation_rate = underestimation_rate * 0.5  # Conservative estimate: 50% of underestimations are severe

    # Cost of severe underestimation (triggers full manual review)
    manual_review_cost = params['hourly_rate'] * params['hours_without_ml'] * params['manual_review_cost_multiplier']

    # Average cost per valuation with ML (including error costs)
    avg_cost_per_valuation = (
        base_cost_per_valuation +
        (severe_underestimation_rate * manual_review_cost)
    )

    # Monthly operational costs
    monthly_operational = (
        params['infrastructure_monthly'] +
        (params['maintenance_hours_monthly'] * params['maintenance_hourly_rate'])
    )

    # Monthly valuation costs
    valuations_per_month = params['monthly_capacity'] * params['adoption_rate']
    monthly_valuation_cost = avg_cost_per_valuation * valuations_per_month

    # Total monthly cost with ML
    total_monthly_cost = monthly_valuation_cost + monthly_operational

    # Annual cost
    annual_cost = total_monthly_cost * 12

    ml_costs = {
        'base_cost_per_valuation': base_cost_per_valuation,
        'underestimation_rate': underestimation_rate,
        'severe_underestimation_rate': severe_underestimation_rate,
        'manual_review_cost': manual_review_cost,
        'avg_cost_per_valuation': avg_cost_per_valuation,
        'monthly_operational': monthly_operational,
        'valuations_per_month': valuations_per_month,
        'monthly_valuation_cost': monthly_valuation_cost,
        'total_monthly_cost': total_monthly_cost,
        'annual_cost': annual_cost
    }

    print(f"Base cost per valuation (ML):          {format_cop(base_cost_per_valuation)}")
    print(f"\nError Analysis:")
    print(f"  Underestimation rate:                {underestimation_rate*100:.2f}%")
    print(f"  Severe underestimation rate:         {severe_underestimation_rate*100:.2f}%")
    print(f"  Manual review cost (when triggered): {format_cop(manual_review_cost)}")
    print(f"\nAverage cost per valuation (ML + errors): {format_cop(avg_cost_per_valuation)}")
    print(f"\nMonthly costs:")
    print(f"  Operational (infra + maintenance):   {format_cop(monthly_operational)}")
    print(f"  Valuations ({valuations_per_month:.0f} apartments):    {format_cop(monthly_valuation_cost)}")
    print(f"  Total monthly cost:                  {format_cop(total_monthly_cost)}")
    print(f"\nAnnual cost:                           {format_cop(annual_cost)}")

    return ml_costs


def calculate_savings_and_roi(params, current_costs, ml_costs):
    """Calculate savings and ROI."""

    print_section_header("SAVINGS AND ROI ANALYSIS", "-")

    # Development cost (one-time)
    development_cost = params['dev_hours'] * params['dev_hourly_rate']

    # Monthly savings (current - ML)
    valuations_per_month = ml_costs['valuations_per_month']
    current_monthly_for_ml_volume = (
        current_costs['cost_per_valuation'] * valuations_per_month
    )
    monthly_savings = current_monthly_for_ml_volume - ml_costs['total_monthly_cost']

    # Annual savings
    annual_savings = monthly_savings * 12

    # Break-even point (months)
    if monthly_savings > 0:
        break_even_months = development_cost / monthly_savings
    else:
        break_even_months = float('inf')

    # ROI calculations
    # Year 1
    year1_investment = development_cost + (ml_costs['monthly_operational'] * 12)
    year1_savings = annual_savings
    year1_net = year1_savings - year1_investment
    year1_roi = (year1_net / year1_investment) * 100 if year1_investment > 0 else 0

    # Year 2 (no development cost)
    year2_investment = ml_costs['monthly_operational'] * 12
    year2_savings = annual_savings
    year2_net = year2_savings - year2_investment
    year2_roi = (year2_net / year2_investment) * 100 if year2_investment > 0 else 0

    # 3 Year Total
    total_3year_investment = development_cost + (ml_costs['monthly_operational'] * 36)
    total_3year_savings = annual_savings * 3
    total_3year_net = total_3year_savings - total_3year_investment
    total_3year_roi = (total_3year_net / total_3year_investment) * 100 if total_3year_investment > 0 else 0

    roi_metrics = {
        'development_cost': development_cost,
        'monthly_savings': monthly_savings,
        'annual_savings': annual_savings,
        'break_even_months': break_even_months,
        'year1_investment': year1_investment,
        'year1_savings': year1_savings,
        'year1_net': year1_net,
        'year1_roi': year1_roi,
        'year2_investment': year2_investment,
        'year2_savings': year2_savings,
        'year2_net': year2_net,
        'year2_roi': year2_roi,
        'total_3year_investment': total_3year_investment,
        'total_3year_savings': total_3year_savings,
        'total_3year_net': total_3year_net,
        'total_3year_roi': total_3year_roi
    }

    print(f"Development cost (one-time):      {format_cop(development_cost)}")
    print(f"\nMonthly savings:                  {format_cop(monthly_savings)}")
    print(f"Annual savings:                   {format_cop(annual_savings)}")
    print(f"\nBreak-even point:                 {break_even_months:.1f} months")
    print(f"\n" + "="*70)
    print("ROI ANALYSIS")
    print("="*70)
    print(f"\nYear 1:")
    print(f"  Investment:                     {format_cop(year1_investment)}")
    print(f"  Savings:                        {format_cop(year1_savings)}")
    print(f"  Net:                            {format_cop(year1_net)}")
    print(f"  ROI:                            {year1_roi:+.2f}%")
    print(f"\nYear 2:")
    print(f"  Investment:                     {format_cop(year2_investment)}")
    print(f"  Savings:                        {format_cop(year2_savings)}")
    print(f"  Net:                            {format_cop(year2_net)}")
    print(f"  ROI:                            {year2_roi:+.2f}%")
    print(f"\n3-Year Total:")
    print(f"  Total Investment:               {format_cop(total_3year_investment)}")
    print(f"  Total Savings:                  {format_cop(total_3year_savings)}")
    print(f"  Total Net:                      {format_cop(total_3year_net)}")
    print(f"  Total ROI:                      {total_3year_roi:+.2f}%")

    return roi_metrics


def visualize_business_value(current_costs, ml_costs, roi_metrics):
    """Create visualizations for business value."""

    print_section_header("CREATING BUSINESS VALUE VISUALIZATIONS", "-")

    # =================================================================
    # 1. Cost Comparison
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per valuation cost
    categories = ['Without ML', 'With ML\n(base)', 'With ML\n(avg w/ errors)']
    costs = [
        current_costs['cost_per_valuation'],
        ml_costs['base_cost_per_valuation'],
        ml_costs['avg_cost_per_valuation']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    axes[0].bar(categories, costs, color=colors, alpha=0.8)
    axes[0].set_ylabel('Cost (COP)')
    axes[0].set_title('Cost per Valuation Comparison', fontsize=14, fontweight='bold')
    axes[0].ticklabel_format(style='plain', axis='y')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (cat, cost) in enumerate(zip(categories, costs)):
        axes[0].text(i, cost, f'{format_cop(cost)}', ha='center', va='bottom', fontsize=9)

    # Monthly cost comparison
    monthly_categories = ['Without ML', 'With ML']
    valuations = ml_costs['valuations_per_month']
    monthly_costs = [
        current_costs['cost_per_valuation'] * valuations,
        ml_costs['total_monthly_cost']
    ]

    axes[1].bar(monthly_categories, monthly_costs, color=['#FF6B6B', '#45B7D1'], alpha=0.8)
    axes[1].set_ylabel('Cost (COP)')
    axes[1].set_title(f'Monthly Cost Comparison ({valuations:.0f} valuations)', fontsize=14, fontweight='bold')
    axes[1].ticklabel_format(style='plain', axis='y')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (cat, cost) in enumerate(zip(monthly_categories, monthly_costs)):
        axes[1].text(i, cost, f'{format_cop(cost)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_figure('25_cost_comparison.png')
    plt.close()

    # =================================================================
    # 2. Break-even Analysis
    # =================================================================
    months = np.arange(0, 25)
    cumulative_savings = roi_metrics['monthly_savings'] * months
    cumulative_investment = roi_metrics['development_cost'] + (ml_costs['monthly_operational'] * months)
    cumulative_net = cumulative_savings - cumulative_investment

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(months, cumulative_savings, label='Cumulative Savings', linewidth=2.5, color='green')
    ax.plot(months, cumulative_investment, label='Cumulative Investment', linewidth=2.5, color='red')
    ax.plot(months, cumulative_net, label='Cumulative Net Benefit', linewidth=2.5, color='blue', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Mark break-even point
    be_months = roi_metrics['break_even_months']
    if be_months < 25:
        ax.axvline(x=be_months, color='orange', linestyle=':', linewidth=2)
        ax.plot(be_months, 0, 'o', markersize=12, color='orange', label=f'Break-even ({be_months:.1f} months)')

    ax.set_xlabel('Months', fontsize=12)
    ax.set_ylabel('Amount (COP)', fontsize=12)
    ax.set_title('Break-even Analysis', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    save_figure('26_breakeven_analysis.png')
    plt.close()

    # =================================================================
    # 3. ROI Over Time
    # =================================================================
    years = [1, 2, 3]
    roi_values = [
        roi_metrics['year1_roi'],
        roi_metrics['year2_roi'],
        roi_metrics['total_3year_roi']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(years, roi_values, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('ROI (%)', fontsize=12)
    ax.set_title('Return on Investment by Year', fontsize=14, fontweight='bold')
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, roi in zip(bars, roi_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{roi:+.1f}%', ha='center', va='bottom' if roi > 0 else 'top', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure('27_roi_by_year.png')
    plt.close()

    # =================================================================
    # 4. Sensitivity Analysis
    # =================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    adoption_rates = np.arange(0.2, 1.01, 0.1)
    monthly_savings_by_adoption = []

    for rate in adoption_rates:
        valuations = ml_costs['monthly_capacity'] * rate
        current_cost = current_costs['cost_per_valuation'] * valuations
        ml_cost = ml_costs['total_monthly_cost'] * (rate / ml_costs['adoption_rate'])
        savings = current_cost - ml_cost
        monthly_savings_by_adoption.append(savings)

    ax.plot(adoption_rates * 100, monthly_savings_by_adoption, linewidth=2.5, marker='o', markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Adoption Rate (%)', fontsize=12)
    ax.set_ylabel('Monthly Savings (COP)', fontsize=12)
    ax.set_title('Sensitivity Analysis: Savings vs Adoption Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    save_figure('28_sensitivity_analysis.png')
    plt.close()


def main():
    """Main business value analysis execution."""

    print_section_header("BUSINESS VALUE ANALYSIS - HABITALPIES PROJECT")

    # Load metrics
    metrics = load_evaluation_metrics()

    # Define parameters
    params = define_business_parameters()

    # Calculate costs
    current_costs = calculate_current_costs(params)
    ml_costs = calculate_ml_costs(params, metrics)

    # Calculate ROI
    roi_metrics = calculate_savings_and_roi(params, current_costs, ml_costs)

    # Visualizations
    visualize_business_value(current_costs, ml_costs, roi_metrics)

    # =================================================================
    # Save comprehensive report
    # =================================================================
    print_section_header("GENERATING BUSINESS VALUE REPORT", "-")

    report = f"""
{'='*80}
BUSINESS VALUE ANALYSIS REPORT
HabitAlpes - Apartment Price Prediction Model
{'='*80}

EXECUTIVE SUMMARY
-----------------

The implementation of the Machine Learning model for apartment price prediction
demonstrates strong business value for HabitAlpes with significant cost savings
and rapid return on investment.

KEY FINDINGS:
-------------

1. COST REDUCTION
   - Current cost per valuation:        {format_cop(current_costs['cost_per_valuation'])}
   - ML cost per valuation:              {format_cop(ml_costs['avg_cost_per_valuation'])}
   - Savings per valuation:              {format_cop(current_costs['cost_per_valuation'] - ml_costs['avg_cost_per_valuation'])}
   - Reduction:                          {((current_costs['cost_per_valuation'] - ml_costs['avg_cost_per_valuation']) / current_costs['cost_per_valuation'] * 100):.1f}%

2. FINANCIAL IMPACT
   - Monthly savings:                    {format_cop(roi_metrics['monthly_savings'])}
   - Annual savings:                     {format_cop(roi_metrics['annual_savings'])}
   - Break-even point:                   {roi_metrics['break_even_months']:.1f} months

3. RETURN ON INVESTMENT
   - Year 1 ROI:                         {roi_metrics['year1_roi']:+.1f}%
   - Year 2 ROI:                         {roi_metrics['year2_roi']:+.1f}%
   - 3-Year Total ROI:                   {roi_metrics['total_3year_roi']:+.1f}%

4. INVESTMENT BREAKDOWN
   - Development cost (one-time):        {format_cop(roi_metrics['development_cost'])}
   - Monthly operational cost:           {format_cop(ml_costs['monthly_operational'])}
   - Annual operational cost:            {format_cop(ml_costs['monthly_operational'] * 12)}

DETAILED ANALYSIS:
------------------

MODEL PERFORMANCE IMPACT:
  - Model R²:                            {metrics['R2']:.4f}
  - Mean Absolute Error:                 {format_cop(metrics['MAE'])}
  - Predictions within ±20M:             {metrics['Within_20M_%']:.1f}%
  - Underestimation rate:                {ml_costs['underestimation_rate']*100:.2f}%
  - Severe underestimations:             {ml_costs['severe_underestimation_rate']*100:.2f}%

ERROR COSTS:
  - Manual review cost (when triggered): {format_cop(ml_costs['manual_review_cost'])}
  - Expected error cost per valuation:   {format_cop(ml_costs['avg_cost_per_valuation'] - ml_costs['base_cost_per_valuation'])}

SCALABILITY:
  - Current monthly capacity:            {params['monthly_capacity']} apartments
  - ML adoption rate:                    {params['adoption_rate']*100:.0f}%
  - Monthly ML valuations:               {ml_costs['valuations_per_month']:.0f} apartments

3-YEAR PROJECTION:
  - Total investment:                    {format_cop(roi_metrics['total_3year_investment'])}
  - Total savings:                       {format_cop(roi_metrics['total_3year_savings'])}
  - Total net benefit:                   {format_cop(roi_metrics['total_3year_net'])}

RECOMMENDATIONS:
----------------

1. IMMEDIATE IMPLEMENTATION:
   With a break-even point of {roi_metrics['break_even_months']:.1f} months and strong ROI,
   immediate deployment is recommended.

2. GRADUAL ROLLOUT:
   Start with {params['adoption_rate']*100:.0f}% adoption rate and monitor performance.
   Scale up as confidence in the model increases.

3. CONTINUOUS MONITORING:
   Track underestimation rates and adjust the model retraining schedule
   to maintain accuracy and minimize manual review costs.

4. EXPERT TRAINING:
   Train experts to effectively use ML predictions, reducing review time
   from 6 hours to the target 1 hour.

5. MODEL UPDATES:
   Plan for quarterly model retraining to maintain accuracy as the market evolves.
   Budget {format_cop(params['maintenance_hours_monthly'] * params['maintenance_hourly_rate'])} monthly for maintenance.

RISK MITIGATION:
----------------

1. Quality Assurance:
   - Implement random sampling of ML valuations for quality checks
   - Track actual vs predicted prices for deployed valuations

2. Client Trust:
   - Provide transparency through SHAP/LIME explanations
   - Offer manual review option for high-value properties

3. Market Changes:
   - Monitor model performance metrics monthly
   - Retrain model when MAPE exceeds acceptable threshold

CONCLUSION:
-----------

The ML model delivers substantial value to HabitAlpes with:
- {((current_costs['cost_per_valuation'] - ml_costs['avg_cost_per_valuation']) / current_costs['cost_per_valuation'] * 100):.1f}% cost reduction per valuation
- {format_cop(roi_metrics['annual_savings'])} annual savings
- {roi_metrics['break_even_months']:.1f} month break-even period
- {roi_metrics['total_3year_roi']:+.1f}% ROI over 3 years

The business case is compelling and warrants immediate implementation.

{'='*80}
END OF REPORT
{'='*80}
"""

    print(report)

    # Save report
    with open(get_processed_data_path('../results/business_value_report.txt'), 'w') as f:
        f.write(report)

    # Save metrics to CSV
    all_metrics = {
        **params,
        **current_costs,
        **ml_costs,
        **roi_metrics
    }

    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(get_processed_data_path('../results/business_value_metrics.csv'), index=False)

    print_section_header("BUSINESS VALUE ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Report saved to: data/results/business_value_report.txt")
    print(f"Metrics saved to: data/results/business_value_metrics.csv")
    print(f"Visualizations saved to: reports/figures/")


if __name__ == '__main__':
    main()
