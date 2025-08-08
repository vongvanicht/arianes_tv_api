"""
Configuration settings for utility-based demand estimation
"""

import os


class Config:
    """Configuration class for demand estimation pipeline"""
    
    # Data directories
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs'
    
    # MCMC calibration settings
    MCMC_ITERATIONS = 1000
    BURN_IN_PERIOD = 300
    
    # Parameter step sizes for MCMC proposals
    STEP_SIZES = {
        'U_max': 1.0,      # Maximum utility parameter
        'alpha': 30.0,     # Time location parameter (minutes)
        'beta': 0.002,     # Steepness parameter
        'gamma': 0.1       # Skewness parameter
        # 'tau' is fixed (0.0 for work, 1.0 for shop/leisure)
    }
    
    # Activity-specific constraints
    ACTIVITY_CONSTRAINTS = {
        'work': {
            'min_departure': 6 * 60,    # 6:00 AM in minutes
            'max_departure': 10 * 60,   # 10:00 AM in minutes
            'durations': [240, 480, 600],  # 4h, 8h, 10h in minutes
            'departure_step': 30        # Every 30 minutes
        },
        'shop': {
            'min_departure': 8 * 60,    # 8:00 AM in minutes
            'max_departure': 20 * 60,   # 8:00 PM in minutes
            'durations': [30, 60, 120], # 0.5h, 1h, 2h in minutes
            'departure_step': 60        # Every hour
        },
        'leisure': {
            'min_departure': 10 * 60,   # 10:00 AM in minutes
            'max_departure': 19 * 60,   # 7:00 PM in minutes
            'durations': [60, 120, 180, 240], # 1h, 2h, 3h, 4h in minutes
            'departure_step': 60        # Every hour
        }
    }
    
    # Utility function parameters (initial values)
    INITIAL_UTILITY_PARAMETERS = {
        'work': {
            'u_before': {'U_max': 8.0, 'alpha': 480, 'beta': 0.010, 'gamma': 1.0, 'tau': 0.0},
            'u_activity': {'U_max': 15.0, 'alpha': 540, 'beta': 0.008, 'gamma': 1.2, 'tau': 0.0},
            'u_after': {'U_max': 6.0, 'alpha': 990, 'beta': 0.012, 'gamma': 0.8, 'tau': 0.0}
        },
        'shop': {
            'u_before': {'U_max': 5.0, 'alpha': 60, 'beta': 0.015, 'gamma': 1.0, 'tau': 1.0},
            'u_activity': {'U_max': 10.0, 'alpha': 72, 'beta': 0.020, 'gamma': 1.5, 'tau': 1.0},
            'u_after': {'U_max': 4.0, 'alpha': 84, 'beta': 0.015, 'gamma': 1.0, 'tau': 1.0}
        },
        'leisure': {
            'u_before': {'U_max': 4.0, 'alpha': 66, 'beta': 0.012, 'gamma': 1.0, 'tau': 1.0},
            'u_activity': {'U_max': 12.0, 'alpha': 78, 'beta': 0.010, 'gamma': 1.3, 'tau': 1.0},
            'u_after': {'U_max': 3.0, 'alpha': 90, 'beta': 0.015, 'gamma': 0.9, 'tau': 1.0}
        }
    }
    
    # Model settings
    TIME_STEP = 10  # Minutes (utility function discretization)
    MAX_END_TIME = 22 * 60  # 10:00 PM in minutes
    TRAVEL_COST_PER_MINUTE = 0.5  # Utility cost per minute of travel
    
    # Numerical settings
    UTILITY_CLIP_RANGE = (-700, 700)  # For numerical stability in logit
    
    def __init__(self):
        """Initialize configuration and create directories if needed"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def update_mcmc_settings(self, iterations=None, step_sizes=None):
        """Update MCMC settings"""
        if iterations:
            self.MCMC_ITERATIONS = iterations
        if step_sizes:
            self.STEP_SIZES.update(step_sizes)
    
    def get_activity_list(self):
        """Get list of supported activities"""
        return list(self.INITIAL_UTILITY_PARAMETERS.keys())