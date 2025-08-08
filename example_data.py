"""
Example data generator for utility-based demand estimation
Creates sample CSV files in the required format
"""

import pandas as pd
import numpy as np
import os


def create_example_data(output_dir='data'):
    """Create example input data files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define zones and activities
    zones = ['A', 'B', 'C', 'D']
    activities = ['work', 'shop', 'leisure']
    hours = [f"{h:02d}:00" for h in range(24)]
    
    # 1. Create trips_to_zones.csv (hourly arrivals)
    np.random.seed(42)
    trips_to_data = []
    
    for hour in range(24):
        row = {}
        for zone in zones:
            # Peak hours (7-9 AM, 12-1 PM, 5-7 PM) have more trips
            if hour in [7, 8, 12, 17, 18]:
                base_trips = np.random.poisson(25)
            elif hour in [9, 10, 11, 13, 14, 15, 16]:
                base_trips = np.random.poisson(15)
            else:
                base_trips = np.random.poisson(5)
            
            row[zone] = max(0, base_trips)
        trips_to_data.append(row)
    
    trips_to = pd.DataFrame(trips_to_data, index=hours)
    trips_to.to_csv(os.path.join(output_dir, 'trips_to_zones.csv'))
    
    # 2. Create trips_from_zones.csv (hourly departures)
    # Similar pattern but shifted later in the day
    trips_from_data = []
    
    for hour in range(24):
        row = {}
        for zone in zones:
            # Peak departure hours (8-10 AM, 1-2 PM, 6-8 PM)
            if hour in [8, 9, 13, 18, 19]:
                base_trips = np.random.poisson(25)
            elif hour in [10, 11, 12, 14, 15, 16, 17]:
                base_trips = np.random.poisson(15)
            else:
                base_trips = np.random.poisson(5)
            
            row[zone] = max(0, base_trips)
        trips_from_data.append(row)
    
    trips_from = pd.DataFrame(trips_from_data, index=hours)
    trips_from.to_csv(os.path.join(output_dir, 'trips_from_zones.csv'))
    
    # 3. Create activity_participation.csv
    # Total daily participants by zone and activity
    participation_data = {
        'work': [150, 120, 100, 80],     # Zone A has most workers
        'shop': [80, 70, 60, 50],        # Shopping participation
        'leisure': [60, 50, 45, 40]      # Leisure participation
    }
    
    activity_participation = pd.DataFrame(participation_data, index=zones)
    activity_participation.to_csv(os.path.join(output_dir, 'activity_participation.csv'))
    
    # 4. Create travel_times.csv (symmetric matrix)
    # Travel times in minutes between zones
    travel_times_data = {
        'A': [0, 15, 25, 35],   # From A to A,B,C,D
        'B': [15, 0, 20, 30],   # From B to A,B,C,D  
        'C': [25, 20, 0, 15],   # From C to A,B,C,D
        'D': [35, 30, 15, 0]    # From D to A,B,C,D
    }
    
    travel_times = pd.DataFrame(travel_times_data, index=zones)
    travel_times.to_csv(os.path.join(output_dir, 'travel_times.csv'))
    
    print(f"✓ Created example data files in '{output_dir}/':")
    print(f"  - trips_to_zones.csv: {trips_to.shape}")
    print(f"  - trips_from_zones.csv: {trips_from.shape}")
    print(f"  - activity_participation.csv: {activity_participation.shape}")
    print(f"  - travel_times.csv: {travel_times.shape}")
    print()
    print("Data summary:")
    print(f"  Zones: {zones}")
    print(f"  Activities: {activities}")
    print(f"  Total daily trips TO: {trips_to.sum().sum()}")
    print(f"  Total daily trips FROM: {trips_from.sum().sum()}")
    print(f"  Total activity participants: {activity_participation.sum().sum()}")
    
    return {
        'trips_to_zones': trips_to,
        'trips_from_zones': trips_from,
        'activity_participation': activity_participation,
        'travel_times': travel_times.values,
        'zones': zones,
        'activities': activities
    }


if __name__ == "__main__":
    # Create example data when run directly
    example_data = create_example_data()
    print("\nExample data created successfully!")
    print("You can now run: python main.py")