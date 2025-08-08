# Integration Guide: Utility-Based Demand Estimation

## Package Structure

```
utility_demand_estimation/
├── main.py                    # Complete pipeline runner
├── library.py                # Core estimation functionality  
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── example_data.py           # Sample data generator
├── README.md                 # Full API documentation
├── INTEGRATION_GUIDE.md      # This file
└── data/                     # Input data directory
    ├── trips_to_zones.csv
    ├── trips_from_zones.csv
    ├── activity_participation.csv
    └── travel_times.csv
```

## For Integration Teams

### What This Code Does

This package implements the utility maximization approach for macroscopic demand estimation from the research manuscript. It takes aggregate trip data and estimates activity-specific travel patterns using behavioral utility theory and MCMC calibration.

**Key Features:**
- Minimal data requirements (only aggregate trip counts)
- Behaviorally consistent results using utility theory
- Activity-specific demand patterns (work, shopping, leisure)
- Real-time capable (no individual microsimulation)
- MCMC parameter calibration for model fitting

### Quick Integration Test

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate example data:**
   ```bash
   python example_data.py
   ```

3. **Run full pipeline:**
   ```bash
   python main.py
   ```

4. **Check outputs:**
   ```
   outputs/
   ├── trip_patterns/
   │   ├── work_to.csv, work_from.csv
   │   ├── shop_to.csv, shop_from.csv  
   │   ├── leisure_to.csv, leisure_from.csv
   │   └── total_to.csv, total_from.csv
   ├── calibrated_parameters.csv
   └── estimation_summary.txt
   ```

### Integration Options

#### Option 1: Direct Python Integration
```python
from library import UtilityDemandEstimator
from config import Config

# Initialize
estimator = UtilityDemandEstimator(Config())

# Your data in required format
input_data = {
    'trips_to_zones': your_trips_to_df,
    'trips_from_zones': your_trips_from_df,
    'activity_participation': your_participation_df,
    'travel_times': your_travel_times_array,
    'zones': your_zone_list,
    'activities': ['work', 'shop', 'leisure']
}

# Run estimation
results = estimator.estimate_demand(input_data)

# Use results
activity_patterns = results['trip_patterns']
parameters = results['parameters']
```

#### Option 2: REST API Wrapper
You can wrap the core functionality in REST endpoints:

```python
from flask import Flask, request, jsonify
from library import UtilityDemandEstimator
from config import Config

app = Flask(__name__)
estimator = UtilityDemandEstimator(Config())

@app.route('/estimate_demand', methods=['POST'])
def estimate_demand():
    input_data = request.json
    results = estimator.estimate_demand(input_data)
    return jsonify(results)
```

#### Option 3: Command Line Interface
```python
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    # Load data, run estimation, save results
    # ... implementation
```

### Key Integration Points

#### Input Data Requirements
- **Spatial**: Zone-based (any number of zones)
- **Temporal**: Hourly trip counts (24 hours)
- **Activities**: Work, shopping, leisure (expandable)
- **Format**: CSV files or pandas DataFrames

#### Output Data Provided
- **Activity-specific OD matrices**: Hourly trips by activity type
- **Total OD matrices**: Aggregated across all activities  
- **Utility parameters**: Behavioral insights for policy analysis
- **Model diagnostics**: Calibration quality metrics

#### Configuration Flexibility
- MCMC iterations (quality vs. speed tradeoff)
- Parameter step sizes (convergence tuning)
- Activity constraints (time windows, durations)
- Initial parameter values

### Performance Characteristics

- **Computational complexity**: O(zones²) for choice generation
- **Memory usage**: Moderate (stores choice alternatives)
- **Runtime**: Minutes to hours depending on zones and MCMC iterations
- **Scalability**: Suitable for city-level analyses (hundreds of zones)

### Data Quality Requirements

- **Trip counts**: Should be consistent (TO ≈ FROM over full day)
- **Activity participation**: Should reflect realistic population levels
- **Travel times**: Can be time-averaged or activity-specific
- **Missing data**: Model handles zeros but not missing values

### Common Integration Scenarios

#### 1. Transportation Planning Model
```python
# Use as dynamic OD matrix generator
od_matrices = results['trip_patterns']
work_od = od_matrices['work_to']  # Work trip arrivals
total_od = od_matrices['total_to']  # All trip arrivals
```

#### 2. Real-time Demand Estimation
```python
# Quick estimation with fewer MCMC iterations
config.MCMC_ITERATIONS = 100
estimator = UtilityDemandEstimator(config)
results = estimator.estimate_demand(current_data)
```

#### 3. Policy Analysis
```python
# Use calibrated parameters for scenario analysis
params = results['parameters']
work_peak_time = params[
    (params['activity'] == 'work') & 
    (params['utility_type'] == 'u_activity') & 
    (params['parameter'] == 'alpha')
]['value'].iloc[0]
```

### Error Handling

The code includes validation for:
- Data dimension mismatches
- Parameter bound violations  
- Numerical instabilities
- Empty or invalid input data

### Support and Customization

#### Extending to New Activities
1. Add activity to `config.ACTIVITY_CONSTRAINTS`
2. Add initial parameters to `config.INITIAL_UTILITY_PARAMETERS`
3. Update input data to include new activity participation

#### Modifying Utility Functions
- Utility evaluation is in `library._evaluate_utility()`
- Currently uses Ettema-Timmermans formulation
- Can be replaced with alternative utility functions

#### Adjusting Choice Generation
- Alternative generation is in `library._generate_alternatives()`
- Can modify time constraints, duration options
- Can add mode choice or other dimensions

### Testing and Validation

The package includes:
- Example data generator for testing
- Built-in diagnostics and validation
- Conservation checks (trips out = trips back)
- Parameter bound enforcement
- Likelihood monitoring during calibration

### Deployment Considerations

- **Dependencies**: Only numpy, pandas, scipy (standard packages)
- **Platform**: Pure Python (cross-platform compatible)
- **Scaling**: Can run on single machine or distributed
- **Integration**: Flexible APIs for different use cases