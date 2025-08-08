# Utility-Based Demand Estimation API

A Python implementation of the utility maximization approach for macroscopic activity-specific demand estimation. This model estimates dynamic origin-destination matrices by activity type using utility theory and MCMC parameter calibration.

## Overview

The model predicts individual travel behavior components including activity type, departure/arrival times, destination zones, and trip patterns using utility maximization principles. It requires minimal aggregate data and provides behaviorally consistent results suitable for macroscopic traffic modeling applications.

## Quick Start

### Installation

```python
pip install -r requirements.txt
```

### Basic Usage

```python
from library import UtilityDemandEstimator
from config import Config
import pandas as pd

# Initialize
config = Config()
estimator = UtilityDemandEstimator(config)

# Load your data (see Data Format section)
input_data = {
    'trips_to_zones': pd.read_csv('data/trips_to_zones.csv'),
    'trips_from_zones': pd.read_csv('data/trips_from_zones.csv'),
    'activity_participation': pd.read_csv('data/activity_participation.csv', index_col=0),
    'travel_times': pd.read_csv('data/travel_times.csv', index_col=0).values,
    'zones': ['A', 'B', 'C', 'D'],  # Your zone names
    'activities': ['work', 'shop', 'leisure']  # Activity types
}

# Run estimation
results = estimator.estimate_demand(input_data)
```

## API Reference

### Main Class: `UtilityDemandEstimator`

#### Constructor
```python
estimator = UtilityDemandEstimator(config)
```

**Parameters:**
- `config`: Configuration object with model settings

#### Primary Method: `estimate_demand()`

```python
results = estimator.estimate_demand(input_data)
```

**Input Data Structure:**
The `input_data` dictionary must contain:

- **trips_to_zones**: DataFrame (24 rows × N zones)
  - Hourly trip counts arriving at each destination zone
  - Index: Hour strings ('00:00', '01:00', ..., '23:00')
  - Columns: Zone names

- **trips_from_zones**: DataFrame (24 rows × N zones)  
  - Hourly trip counts departing from each origin zone
  - Same format as trips_to_zones

- **activity_participation**: DataFrame (N zones × M activities)
  - Total daily participants for each activity type by home zone
  - Index: Zone names
  - Columns: Activity types ('work', 'shop', 'leisure')

- **travel_times**: NumPy array (N zones × N zones)
  - Zone-to-zone travel times in minutes

- **zones**: List of zone names (strings)

- **activities**: List of activity types (strings)

**Output Structure:**
Returns dictionary with:

- **trip_patterns**: Dictionary of DataFrames
  - Keys: 'work_to', 'work_from', 'shop_to', 'shop_from', 'leisure_to', 'leisure_from', 'total_to', 'total_from'
  - Values: DataFrames (24 hours × zones) with predicted trip counts

- **parameters**: DataFrame 
  - Calibrated utility function parameters
  - Columns: 'activity', 'utility_type', 'parameter', 'value'

- **summary**: String with calibration diagnostics and model performance

## Data Format Examples

### trips_to_zones.csv
```csv
,A,B,C,D
00:00,5,3,2,1
01:00,2,1,1,0
...
23:00,8,5,4,2
```

### trips_from_zones.csv
```csv
,A,B,C,D
00:00,3,2,1,1
01:00,1,1,0,0
...
23:00,6,4,3,2
```

### activity_participation.csv
```csv
,work,shop,leisure
A,120,80,60
B,100,70,50
C,90,60,45
D,80,50,40
```

### travel_times.csv
```csv
,A,B,C,D
A,0,15,25,35
B,15,0,20,30
C,25,20,0,15
D,35,30,15,0
```

## Configuration

### Basic Configuration
```python
from config import Config

config = Config()

# Adjust MCMC settings
config.MCMC_ITERATIONS = 2000
config.STEP_SIZES['U_max'] = 1.5

# Update data directories
config.DATA_DIR = 'my_data'
config.OUTPUT_DIR = 'my_results'
```

### Key Configuration Parameters

- **MCMC_ITERATIONS**: Number of calibration iterations (default: 1000)
- **STEP_SIZES**: Parameter step sizes for MCMC proposals
- **ACTIVITY_CONSTRAINTS**: Time and duration constraints by activity type
- **INITIAL_UTILITY_PARAMETERS**: Starting values for utility function parameters

## Model Components

### 1. Utility Functions
The model uses Ettema-Timmermans utility functions with parameters:
- **U_max**: Maximum utility value
- **alpha**: Time location parameter (minutes from midnight)
- **beta**: Steepness parameter
- **gamma**: Skewness parameter  
- **tau**: Duration dependency (0 for work, 1 for shop/leisure)

### 2. Choice Model
Generates feasible alternatives with activity-specific constraints:
- **Work**: 6-10 AM departures, 4-10 hour durations
- **Shopping**: 8 AM-8 PM departures, 0.5-2 hour durations
- **Leisure**: 10 AM-7 PM departures, 1-4 hour durations

### 3. Trip Generation
Distributes activity participants across alternatives using multinomial logit probabilities, generating separate patterns for:
- Activity trips (people going TO perform activities)
- Return trips (people coming back FROM activities)

### 4. MCMC Calibration
Calibrates utility parameters against observed trip data using Metropolis-Hastings sampling.

## Output Interpretation

### Trip Patterns
- **work_to**: People arriving at zones to work
- **work_from**: People leaving zones after work
- **shop_to**: People arriving at zones to shop
- **shop_from**: People leaving zones after shopping
- **leisure_to**: People arriving at zones for leisure
- **leisure_from**: People leaving zones after leisure
- **total_to**: Sum of all activity arrivals
- **total_from**: Sum of all activity departures

### Parameters
Calibrated utility function parameters provide behavioral insights:
- Peak activity times (alpha parameters)
- Activity attractiveness (U_max parameters)
- Temporal flexibility (beta, gamma parameters)

## Integration Notes

### For Transportation Models
- Use `trip_patterns['total_to']` and `total_from` as dynamic OD matrices
- Activity-specific patterns enable targeted policy analysis
- Parameters provide behavioral foundations for scenario analysis

### For Real-time Applications
- Model runs efficiently with aggregate data
- No individual-level simulation required
- Suitable for operational traffic management

### Data Requirements
- Standard traffic counting data (hourly trip matrices)
- Basic activity participation surveys
- Zone-to-zone travel time estimates
- No detailed individual travel diaries needed

## Error Handling

The model includes validation for:
- Data dimension consistency
- Parameter bound constraints
- Numerical stability in utility calculations
- MCMC convergence monitoring

## Performance Considerations

- Computational complexity scales with number of zones squared
- MCMC iterations affect calibration quality vs. runtime
- Memory usage depends on number of choice alternatives generated
- Larger step sizes improve mixing but may reduce acceptance rates

## Support

For integration support or questions about model implementation, refer to the original research paper or contact the development team.# arianes_tv_api
