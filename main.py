"""
Utility-Based Demand Estimation Pipeline
Main execution script for the 5-step demand estimation process
"""

import os
import pandas as pd
from library import UtilityDemandEstimator
from config import Config


def load_input_data(data_dir):
    """Load input data from CSV files"""
    
    print("Loading input data...")
    
    # Load required input files with proper index handling
    trips_to = pd.read_csv(os.path.join(data_dir, 'trips_to_zones.csv'), index_col=0)
    trips_from = pd.read_csv(os.path.join(data_dir, 'trips_from_zones.csv'), index_col=0)
    activity_participation = pd.read_csv(os.path.join(data_dir, 'activity_participation.csv'), index_col=0)
    travel_times = pd.read_csv(os.path.join(data_dir, 'travel_times.csv'), index_col=0).values
    
    # Package input data
    input_data = {
        'trips_to_zones': trips_to,
        'trips_from_zones': trips_from,
        'activity_participation': activity_participation,
        'travel_times': travel_times,
        'zones': list(trips_to.columns),
        'activities': list(activity_participation.columns)
    }
    
    print(f"✓ Loaded {len(input_data['zones'])} zones, {len(input_data['activities'])} activities")
    print(f"✓ Time periods: {len(trips_to)} hours")
    
    return input_data


def save_results(results, output_dir):
    """Save estimation results to CSV files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save activity-specific trip patterns
    patterns_dir = os.path.join(output_dir, 'trip_patterns')
    os.makedirs(patterns_dir, exist_ok=True)
    
    for pattern_name, pattern_df in results['trip_patterns'].items():
        pattern_df.to_csv(os.path.join(patterns_dir, f'{pattern_name}.csv'))
    
    # Save calibrated parameters
    results['parameters'].to_csv(os.path.join(output_dir, 'calibrated_parameters.csv'), index=False)
    
    # Save model diagnostics
    with open(os.path.join(output_dir, 'estimation_summary.txt'), 'w') as f:
        f.write(results['summary'])
    
    print(f"✓ Results saved to: {output_dir}")


def main():
    """Run complete utility-based demand estimation pipeline"""
    
    print("=" * 60)
    print("UTILITY-BASED DEMAND ESTIMATION PIPELINE")
    print("=" * 60)
    
    # Configuration
    config = Config()
    
    # Load input data
    input_data = load_input_data(config.DATA_DIR)
    
    # Initialize estimator
    estimator = UtilityDemandEstimator(config)
    
    # Run 5-step pipeline
    print("\nRunning estimation pipeline...")
    results = estimator.estimate_demand(input_data)
    
    # Save results
    save_results(results, config.OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"📊 Generated {len(results['trip_patterns'])} activity-specific trip patterns")
    print(f"📋 Calibrated {len(results['parameters'])} utility parameters")
    print(f"📁 Results saved to: {config.OUTPUT_DIR}")
    
    return results


if __name__ == "__main__":
    results = main()