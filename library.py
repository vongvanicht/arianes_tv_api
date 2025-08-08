"""
Utility-Based Demand Estimation Library
Core functionality for activity-specific demand estimation using utility maximization
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


class UtilityDemandEstimator:
    """Main class for utility-based demand estimation"""
    
    def __init__(self, config):
        self.config = config
        self.parameters = None
        self.utility_functions = None
        
    def estimate_demand(self, input_data):
        """
        Run complete demand estimation pipeline
        
        Args:
            input_data: Dict with keys:
                - trips_to_zones: DataFrame (24 hours x zones)
                - trips_from_zones: DataFrame (24 hours x zones) 
                - activity_participation: DataFrame (zones x activities)
                - travel_times: Array (zones x zones)
                - zones: List of zone names
                - activities: List of activity names
                
        Returns:
            Dict with estimation results
        """
        
        print("Step 1: Data preparation...")
        processed_data = self._prepare_data(input_data)
        
        print("Step 2: Creating utility model...")
        utility_model = self._create_utility_model(processed_data)
        
        print("Step 3: Generating choice alternatives...")
        choice_model = self._create_choice_model(processed_data, utility_model)
        
        print("Step 4: Generating trip patterns...")
        trip_patterns = self._generate_trip_patterns(processed_data, choice_model)
        
        print("Step 5: MCMC parameter calibration...")
        calibrated_results = self._calibrate_parameters(processed_data, initial_trip_patterns=trip_patterns)
        
        # Package final results
        results = {
            'trip_patterns': calibrated_results['final_trip_patterns'],
            'parameters': calibrated_results['final_parameters'],
            'summary': calibrated_results['summary']
        }
        
        return results
    
    def _prepare_data(self, input_data):
        """Step 1: Prepare and validate input data"""
        
        # Create travel times dict (same matrix for all activities for now)
        travel_times = {}
        for activity in input_data['activities']:
            travel_times[activity] = input_data['travel_times'].copy()
        
        processed_data = {
            'trips_to_zones': input_data['trips_to_zones'],
            'trips_from_zones': input_data['trips_from_zones'],
            'activity_participation': input_data['activity_participation'],
            'travel_times': travel_times,
            'zones': input_data['zones'],
            'activities': input_data['activities']
        }
        
        return processed_data
    
    def _create_utility_model(self, data):
        """Step 2: Create utility functions for each activity"""
        
        # Initialize utility parameters (Ettema-Timmermans formulation)
        parameters = {
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
        
        # Create utility functions  
        time_points = np.arange(0, 1440, 10)  # Every 10 minutes
        utility_functions = self._create_utility_functions(parameters, time_points)
        
        utility_model = {
            'parameters': parameters,
            'utility_functions': utility_functions,
            'time_points': time_points,
            'zones': data['zones'],
            'activities': data['activities']
        }
        
        return utility_model
    
    def _create_utility_functions(self, parameters, time_points):
        """Create lookup tables for utility functions"""
        
        utility_functions = {}
        
        for activity in parameters.keys():
            utility_functions[activity] = {}
            
            for utility_type in ['u_before', 'u_activity', 'u_after']:
                params = parameters[activity][utility_type]
                
                utility_values = []
                for t in time_points:
                    ts = t  # Simplified: start time = current time
                    utility = self._evaluate_utility(t, ts, params)
                    utility_values.append(utility)
                
                utility_functions[activity][utility_type] = {
                    'time_points': time_points,
                    'utility_values': np.array(utility_values)
                }
        
        return utility_functions
    
    def _evaluate_utility(self, t, ts, params):
        """Evaluate Ettema-Timmermans utility function"""
        
        U_max = params['U_max']
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        tau = params['tau']
        
        # Ettema-Timmermans formula
        arg = beta * (t - (alpha + tau * ts))
        numerator = gamma * beta * U_max
        denominator = np.exp(arg) * np.power(1 + np.exp(-arg), gamma + 1)
        
        return numerator / denominator
    
    def _create_choice_model(self, data, utility_model):
        """Step 3: Generate choice alternatives and probabilities"""
        
        # Generate feasible alternatives
        alternatives = self._generate_alternatives(data['zones'], data['activities'], data['travel_times'])
        
        # Calculate choice probabilities
        probabilities = self._calculate_choice_probabilities(alternatives, utility_model, data['travel_times'])
        
        choice_model = {
            'alternatives': alternatives,
            'probabilities': probabilities,
            'zones': data['zones'],
            'activities': data['activities']
        }
        
        return choice_model
    
    def _generate_alternatives(self, zones, activities, travel_times):
        """Generate feasible choice alternatives with activity-specific constraints"""
        
        # Activity-specific constraints
        constraints = {
            'work': {'min_dep': 6*60, 'max_dep': 10*60, 'durations': [240, 480, 600], 'step': 30},
            'shop': {'min_dep': 8*60, 'max_dep': 20*60, 'durations': [30, 60, 120], 'step': 60},
            'leisure': {'min_dep': 10*60, 'max_dep': 19*60, 'durations': [60, 120, 180, 240], 'step': 60}
        }
        
        alternatives = []
        
        for orig_idx, origin in enumerate(zones):
            for dest_idx, dest in enumerate(zones):
                for activity in activities:
                    
                    travel_time = travel_times[activity][orig_idx, dest_idx]
                    constraint = constraints[activity]
                    
                    for dep_time in range(constraint['min_dep'], constraint['max_dep'], constraint['step']):
                        start_time = dep_time + travel_time
                        
                        for duration in constraint['durations']:
                            end_time = start_time + duration
                            
                            if end_time <= 22 * 60:  # Must end by 10 PM
                                alternatives.append({
                                    'origin': origin,
                                    'origin_idx': orig_idx,
                                    'destination': dest,
                                    'dest_idx': dest_idx,
                                    'activity': activity,
                                    'departure_time': dep_time,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'duration': duration
                                })
        
        return alternatives
    
    def _calculate_choice_probabilities(self, alternatives, utility_model, travel_times):
        """Calculate multinomial logit choice probabilities"""
        
        utilities = []
        
        for alt in alternatives:
            utility = self._calculate_total_utility(
                alt['origin_idx'], alt['dest_idx'], alt['activity'],
                alt['departure_time'], alt['end_time'],
                utility_model, travel_times
            )
            utilities.append(utility)
        
        utilities = np.array(utilities)
        
        # Multinomial logit with numerical stability
        max_utility = np.max(utilities)
        scaled_utilities = utilities - max_utility
        scaled_utilities = np.clip(scaled_utilities, -700, 700)
        
        exp_utilities = np.exp(scaled_utilities)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        return probabilities
    
    def _calculate_total_utility(self, origin, destination, activity, departure_time, end_time, utility_model, travel_times):
        """Calculate total utility with proper time integration"""
        
        travel_time = travel_times[activity][origin, destination]
        start_time = departure_time + travel_time
        
        # Get utility functions
        u_before = utility_model['utility_functions'][activity]['u_before']
        u_activity = utility_model['utility_functions'][activity]['u_activity']
        u_after = utility_model['utility_functions'][activity]['u_after']
        
        time_points = utility_model['time_points']
        
        def integrate_utility(util_func, period_start, period_end, time_step=10):
            """Integrate utility over time period"""
            if period_end <= period_start:
                return 0
            
            period_start = max(0, int(period_start))
            period_end = min(1440, int(period_end))
            
            total_utility = 0
            for t in range(period_start, period_end, time_step):
                closest_idx = np.argmin(np.abs(time_points - t))
                utility_at_t = util_func['utility_values'][closest_idx]
                total_utility += utility_at_t * time_step
            
            return total_utility
        
        # Calculate integrated utilities
        u_before_total = integrate_utility(u_before, 0, departure_time)
        travel_cost = 0.5 * travel_time  # Travel disutility
        u_activity_total = integrate_utility(u_activity, start_time, end_time)
        u_after_total = integrate_utility(u_after, end_time, 1440)
        
        total_utility = u_before_total - travel_cost + u_activity_total + u_after_total
        
        return total_utility
    
    def _generate_trip_patterns(self, data, choice_model):
        """Step 4: Generate activity-specific trip patterns"""
        
        alternatives = choice_model['alternatives']
        probabilities = choice_model['probabilities']
        zones = data['zones']
        activities = data['activities']
        activity_participation = data['activity_participation']
        
        # Initialize trip matrices
        activity_trips_to = np.zeros((len(zones), len(activities), 24))
        return_trips_from = np.zeros((len(zones), len(activities), 24))
        
        # Distribute participants across alternatives
        for home_zone_idx, home_zone in enumerate(zones):
            for activity_idx, activity in enumerate(activities):
                
                total_participants = activity_participation.loc[home_zone, activity]
                
                # Find relevant alternatives
                relevant_alts = []
                relevant_probs = []
                
                for i, alt in enumerate(alternatives):
                    if alt['origin'] == home_zone and alt['activity'] == activity:
                        relevant_alts.append(alt)
                        relevant_probs.append(probabilities[i])
                
                if len(relevant_probs) > 0:
                    relevant_probs = np.array(relevant_probs)
                    
                    # Distribute participants
                    for alt, prob in zip(relevant_alts, relevant_probs):
                        participants = total_participants * prob
                        dest_zone_idx = zones.index(alt['destination'])
                        
                        # Activity trips (arrival time)
                        arrival_hour = alt['start_time'] // 60
                        if arrival_hour < 24:
                            activity_trips_to[dest_zone_idx, activity_idx, arrival_hour] += participants
                        
                        # Return trips (departure time after activity)
                        return_hour = alt['end_time'] // 60
                        if return_hour < 24:
                            return_trips_from[dest_zone_idx, activity_idx, return_hour] += participants
        
        # Create trip pattern DataFrames
        time_index = [f"{h:02d}:00" for h in range(24)]
        trip_patterns = {}
        
        # Activity-specific patterns
        for activity_idx, activity in enumerate(activities):
            # TO patterns
            activity_data = activity_trips_to[:, activity_idx, :].T
            trip_patterns[f'{activity}_to'] = pd.DataFrame(activity_data, index=time_index, columns=zones)
            
            # FROM patterns
            return_data = return_trips_from[:, activity_idx, :].T
            trip_patterns[f'{activity}_from'] = pd.DataFrame(return_data, index=time_index, columns=zones)
        
        # Total patterns
        total_to = np.sum(activity_trips_to, axis=1).T
        trip_patterns['total_to'] = pd.DataFrame(total_to, index=time_index, columns=zones)
        
        total_from = np.sum(return_trips_from, axis=1).T
        trip_patterns['total_from'] = pd.DataFrame(total_from, index=time_index, columns=zones)
        
        return trip_patterns
    
    def _calibrate_parameters(self, data, initial_trip_patterns):
        """Step 5: MCMC parameter calibration"""
        
        observed_to = data['trips_to_zones'].values
        observed_from = data['trips_from_zones'].values
        
        # MCMC settings
        n_iterations = self.config.MCMC_ITERATIONS
        step_sizes = self.config.STEP_SIZES
        
        # Initialize with default parameters
        current_params = self._create_utility_model(data)['parameters']
        
        # Calculate initial likelihood
        pred_to, pred_from = self._run_pipeline_with_parameters(current_params, data)
        current_likelihood = self._calculate_likelihood(pred_to, observed_to) + self._calculate_likelihood(pred_from, observed_from)
        
        # MCMC chain storage
        best_params = deepcopy(current_params)
        best_likelihood = current_likelihood
        accepted_count = 0
        
        print(f"Running MCMC calibration ({n_iterations} iterations)...")
        
        # MCMC loop
        for iteration in range(n_iterations):
            
            # Propose new parameters
            proposed_params = self._propose_new_parameters(current_params, step_sizes)
            
            try:
                # Calculate likelihood with proposed parameters
                pred_to, pred_from = self._run_pipeline_with_parameters(proposed_params, data)
                proposed_likelihood = self._calculate_likelihood(pred_to, observed_to) + self._calculate_likelihood(pred_from, observed_from)
                
                # Metropolis-Hastings acceptance
                acceptance_ratio = min(1.0, np.exp(proposed_likelihood - current_likelihood))
                
                if np.random.random() < acceptance_ratio:
                    current_params = proposed_params
                    current_likelihood = proposed_likelihood
                    accepted_count += 1
                    
                    if current_likelihood > best_likelihood:
                        best_params = deepcopy(current_params)
                        best_likelihood = current_likelihood
                
            except:
                # Reject problematic parameters
                pass
            
            # Progress reporting
            if (iteration + 1) % (n_iterations // 10) == 0:
                progress = (iteration + 1) / n_iterations * 100
                acceptance_rate = accepted_count / (iteration + 1)
                print(f"  {progress:3.0f}% | Likelihood: {current_likelihood:8.2f} | Acceptance: {acceptance_rate:.2f}")
        
        # Generate final results with best parameters
        print("Generating final trip patterns with calibrated parameters...")
        final_trip_patterns = self._run_full_pipeline_with_parameters(best_params, data)
        
        # Convert parameters to DataFrame
        param_data = []
        for activity in best_params.keys():
            for utility_type in best_params[activity].keys():
                for param_name, value in best_params[activity][utility_type].items():
                    param_data.append({
                        'activity': activity,
                        'utility_type': utility_type,
                        'parameter': param_name,
                        'value': value
                    })
        
        final_parameters = pd.DataFrame(param_data)
        
        # Create summary
        final_acceptance = accepted_count / n_iterations
        improvement = best_likelihood - self._calculate_likelihood(initial_trip_patterns['total_to'].values, observed_to) - self._calculate_likelihood(initial_trip_patterns['total_from'].values, observed_from)
        
        summary = f"""MCMC Calibration Results
========================

Iterations: {n_iterations}
Acceptance rate: {final_acceptance:.2%}
Final likelihood: {best_likelihood:.2f}
Likelihood improvement: {improvement:.2f}

Total predicted trips TO: {final_trip_patterns['total_to'].sum().sum():.0f}
Total observed trips TO: {np.sum(observed_to):.0f}
Prediction error TO: {abs(final_trip_patterns['total_to'].sum().sum() - np.sum(observed_to)) / np.sum(observed_to):.1%}

Total predicted trips FROM: {final_trip_patterns['total_from'].sum().sum():.0f}
Total observed trips FROM: {np.sum(observed_from):.0f}
Prediction error FROM: {abs(final_trip_patterns['total_from'].sum().sum() - np.sum(observed_from)) / np.sum(observed_from):.1%}
"""
        
        results = {
            'final_parameters': final_parameters,
            'final_trip_patterns': final_trip_patterns,
            'summary': summary
        }
        
        return results
    
    def _run_pipeline_with_parameters(self, parameters, data):
        """Run steps 2-4 with given parameters (for MCMC) - returns numpy arrays"""
        
        # Create utility model with new parameters
        time_points = np.arange(0, 1440, 10)
        utility_functions = self._create_utility_functions(parameters, time_points)
        
        utility_model = {
            'parameters': parameters,
            'utility_functions': utility_functions,
            'time_points': time_points,
            'zones': data['zones'],
            'activities': data['activities']
        }
        
        # Generate choice model
        choice_model = self._create_choice_model(data, utility_model)
        
        # Generate trip patterns
        trip_patterns = self._generate_trip_patterns(data, choice_model)
        
        return trip_patterns['total_to'].values, trip_patterns['total_from'].values
    
    def _run_full_pipeline_with_parameters(self, parameters, data):
        """Run steps 2-4 with given parameters - returns full trip patterns"""
        
        # Create utility model with new parameters
        time_points = np.arange(0, 1440, 10)
        utility_functions = self._create_utility_functions(parameters, time_points)
        
        utility_model = {
            'parameters': parameters,
            'utility_functions': utility_functions,
            'time_points': time_points,
            'zones': data['zones'],
            'activities': data['activities']
        }
        
        # Generate choice model
        choice_model = self._create_choice_model(data, utility_model)
        
        # Generate trip patterns (returns full dictionary with all activities)
        trip_patterns = self._generate_trip_patterns(data, choice_model)
        
        return trip_patterns
    
    def _calculate_likelihood(self, predicted, observed):
        """Calculate log-likelihood (sum of squared errors)"""
        residuals = predicted - observed
        return -0.5 * np.sum(residuals ** 2)
    
    def _propose_new_parameters(self, current_params, step_sizes):
        """Propose new parameters for MCMC"""
        
        new_params = deepcopy(current_params)
        
        for activity in new_params.keys():
            for utility_type in new_params[activity].keys():
                for param_name in new_params[activity][utility_type].keys():
                    
                    if param_name in step_sizes:
                        noise = np.random.normal(0, step_sizes[param_name])
                        new_params[activity][utility_type][param_name] += noise
                        
                        # Apply bounds
                        if param_name == 'U_max':
                            new_params[activity][utility_type][param_name] = max(1.0, new_params[activity][utility_type][param_name])
                        elif param_name == 'alpha':
                            new_params[activity][utility_type][param_name] = max(0, min(1440, new_params[activity][utility_type][param_name]))
                        elif param_name == 'beta':
                            new_params[activity][utility_type][param_name] = max(0.001, min(0.1, new_params[activity][utility_type][param_name]))
                        elif param_name == 'gamma':
                            new_params[activity][utility_type][param_name] = max(0.1, min(3.0, new_params[activity][utility_type][param_name]))
        
        return new_params