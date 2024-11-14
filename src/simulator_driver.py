import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging
import random
import json
from pathlib import Path

from .types import (
    Vehicle, Request, Match, ServiceArea, 
    SimulationMetrics, SimulationException
)
from .utils import (
    calculate_distance, get_route, 
    convert_to_datetime, setup_logging
)
from .data_loader import DataLoader
from .matching_policy import PoolingMatchingPolicy
from .dqn_policy import DQNDispatchPolicy, State, Action

class SimulationConfig:
    def __init__(self, config_dict: Dict):
        """Initialize simulation configuration"""
        self.data_path = config_dict['data_path']
        self.output_dir = config_dict.get('output_dir', 'simulation_results')
        self.num_vehicles = config_dict['num_vehicles']
        self.max_capacity = config_dict['max_capacity']
        self.timestep = config_dict['timestep']  # seconds
        self.max_time_window = config_dict.get('max_time_window', 15)  # minutes
        self.max_pickup_distance = config_dict.get('max_pickup_distance', 2000)  # meters
        self.max_detour_ratio = config_dict.get('max_detour_ratio', 1.5)
        
        # DQN configuration
        self.state_dim = config_dict.get('state_dim', 128)
        self.action_dim = config_dict.get('action_dim', 100)
        self.batch_size = config_dict.get('batch_size', 32)
        self.learning_rate = config_dict.get('learning_rate', 0.001)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

class SimulatorDriver:
    def __init__(self, config: SimulationConfig):
        """Initialize simulation driver"""
        self.config = config
        self.current_step = 0

        
        self.service_area = ServiceArea()
        
        # Initialize components
        self.data_loader = DataLoader(
            data_path=config.data_path,
        )
        
        self.matching_policy = PoolingMatchingPolicy(
            max_wait_time=300,  # 5 minutes
            max_detour_ratio=config.max_detour_ratio,
            max_pickup_distance=config.max_pickup_distance,
            max_time_window=config.max_time_window,
            service_area=self.service_area
        )
        
        self.dispatch_policy = DQNDispatchPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            learning_rate=config.learning_rate,
            service_area=self.service_area
        )
        
        # Get simulation time boundaries
        self.start_time, self.end_time = self.data_loader.get_time_boundaries()
        self.current_time = self.start_time
        
        # Initialize vehicles and requests
        self.vehicles = self._initialize_vehicles()
        self.active_requests = []
        self.completed_requests = []
        
        # Initialize metrics
        self.metrics = SimulationMetrics()
        self.step_metrics = []

    def _initialize_vehicles(self) -> List[Vehicle]:
        """Initialize vehicles based on initial trip locations"""
        initial_trips = self.data_loader.get_active_trips(self.start_time)
        vehicles = []
        
        # Use initial trip locations
        for i in range(min(self.config.num_vehicles, len(initial_trips))):
            trip = initial_trips[i]
            vehicles.append(Vehicle(
                id=f'v_{i}',
                lat=trip.pickup_lat,
                lon=trip.pickup_lng,
                status='IDLE',
                capacity=self.config.max_capacity,
                current_passengers=0
            ))
        
        # Add remaining vehicles in high-demand areas
        if len(vehicles) < self.config.num_vehicles:
            demand_matrix = self.data_loader.get_demand_matrix(self.start_time)
            remaining = self._create_vehicles_by_demand(
                self.config.num_vehicles - len(vehicles),
                demand_matrix,
                start_idx=len(vehicles)
            )
            vehicles.extend(remaining)
        
        return vehicles

    def _create_vehicles_by_demand(self, count: int, demand_matrix: np.ndarray,
                                start_idx: int) -> List[Vehicle]:
        """Create vehicles distributed according to demand"""
        vehicles = []
        grid_size = len(demand_matrix)
        
        # Convert demand to probabilities
        total_demand = demand_matrix.sum()
        if total_demand > 0:
            probs = demand_matrix / total_demand
        else:
            # If no demand, use uniform distribution
            probs = np.ones_like(demand_matrix) / (grid_size * grid_size)
        
        # Sample grid cells based on demand
        flat_probs = probs.flatten()
        chosen_cells = np.random.choice(
            len(flat_probs),
            size=count,
            p=flat_probs
        )
        
        # Convert cell indices to coordinates
        lat_step = (self.service_area.max_lat - self.service_area.min_lat) / grid_size
        lon_step = (self.service_area.max_lon - self.service_area.min_lon) / grid_size
        
        for i, cell_idx in enumerate(chosen_cells):
            row = cell_idx // grid_size
            col = cell_idx % grid_size
            
            # Add random offset within cell
            lat = (self.service_area.min_lat + row * lat_step + 
                  random.uniform(0, lat_step))
            lon = (self.service_area.min_lon + col * lon_step + 
                  random.uniform(0, lon_step))
            
            vehicles.append(Vehicle(
                id=f'v_{start_idx + i}',
                lat=lat,
                lon=lon,
                status='IDLE',
                capacity=self.config.max_capacity,
                current_passengers=0
            ))
        
        return vehicles

    def _generate_requests(self) -> List[Request]:
        """Generate new requests from trip data"""
        active_trips = self.data_loader.get_active_trips(
            self.current_time,
            self.config.max_time_window
        )
        
        requests = []
        for trip in active_trips:
            request = Request(
                id=trip.trip_id,
                origin_lat=trip.pickup_lat,
                origin_lon=trip.pickup_lng,
                dest_lat=trip.dropoff_lat,
                dest_lon=trip.dropoff_lng,
                pickup_time=convert_to_datetime(
                    trip.week_id,
                    trip.date_id,
                    trip.time_id
                ),
                time_id=trip.time_id,
                week_id=trip.week_id,
                date_id=trip.date_id,
                trip_distance=trip.trip_distance,
                straight_line_distance=trip.straight_line_distance
            )
            requests.append(request)
        
        return requests

    def _get_vehicle_state(self, vehicle: Vehicle) -> State:
        """Get current state for DQN policy"""
        nearby_vehicles = [
            (v.lat, v.lon) for v in self.vehicles
            if v.id != vehicle.id and
            calculate_distance(vehicle.lat, vehicle.lon, v.lat, v.lon) <= 2
        ]
        
        demand_matrix = self.data_loader.get_demand_matrix(
            self.current_time,
            grid_size=10
        )
        
        return State(
            vehicle_location=(vehicle.lat, vehicle.lon),
            demand_matrix=demand_matrix,
            nearby_vehicles=nearby_vehicles,
            time_of_day=self.current_time.hour,
            current_time=self.current_time
        )

    def _execute_vehicle_action(self, vehicle: Vehicle, action: Action) -> None:
        """Execute repositioning action for vehicle"""
        old_location = (vehicle.lat, vehicle.lon)
        target_location = action.target_location
        
        route = get_route([
            {'lat': old_location[0], 'lon': old_location[1]},
            {'lat': target_location[0], 'lon': target_location[1]}
        ], self.service_area)
        
        if route:
            # Update vehicle location
            vehicle.lat = target_location[0]
            vehicle.lon = target_location[1]
            vehicle.route_history.append((vehicle.lat, vehicle.lon))
            
            # Calculate reward based on new position
            new_state = self._get_vehicle_state(vehicle)
            reward = self._calculate_repositioning_reward(
                old_location, target_location, route, new_state
            )
            
            # Store experience
            self.dispatch_policy.remember(
                state=self._get_vehicle_state(vehicle),
                action=action,
                reward=reward,
                next_state=new_state,
                done=False
            )

    def _calculate_repositioning_reward(self, old_location: Tuple[float, float],
                                    new_location: Tuple[float, float],
                                    route: Dict, new_state: State) -> float:
        """Calculate reward for repositioning action"""
        # Distance penalty
        distance = route['trip']['summary']['length']
        distance_penalty = -0.1 * distance  # Small penalty for movement
        
        # Demand reward
        demand_matrix = new_state.demand_matrix
        lat_idx = int((new_location[0] - self.service_area.min_lat) /
                     (self.service_area.max_lat - self.service_area.min_lat) * 9)
        lon_idx = int((new_location[1] - self.service_area.min_lon) /
                     (self.service_area.max_lon - self.service_area.min_lon) * 9)
        demand_reward = 5.0 * demand_matrix[lat_idx, lon_idx]
        
        # Vehicle density penalty
        density_penalty = -2.0 * len(new_state.nearby_vehicles)
        
        return distance_penalty + demand_reward + density_penalty

    def step(self) -> Dict:
        """Execute one simulation step"""
        step_metrics = {
            'timestamp': self.current_time.isoformat(),
            'active_vehicles': len(self.vehicles),
            'active_requests': len(self.active_requests)
        }
        
        try:
            # Generate new requests
            logging.info(f"Starting simulation step {self.current_step}.")
            logging.info(f"Number of active requests: {len(self.active_requests)}")
            logging.info(f"Number of available vehicles: {len([v for v in self.vehicles if v.is_available])}")
            new_requests = self._generate_requests()
            self.active_requests.extend(new_requests)
            step_metrics['new_requests'] = len(new_requests)
            
            # Match vehicles to requests
            matches = self.matching_policy.match(self.vehicles, self.active_requests)
            step_metrics['matches'] = len(matches)
            
            
            # Update vehicle states and metrics based on matches
            self._process_matches(matches)
            
            # Reposition idle vehicles
            idle_vehicles = [v for v in self.vehicles if v.status == 'IDLE']
            step_metrics['idle_vehicles'] = len(idle_vehicles)
            
            for vehicle in idle_vehicles:
                state = self._get_vehicle_state(vehicle)
                action = self.dispatch_policy.get_action(state)
                self._execute_vehicle_action(vehicle, action)
            
            # Train dispatch policy
            if len(self.dispatch_policy.memory) >= self.config.batch_size:
                loss = self.dispatch_policy.train()
                step_metrics['training_loss'] = loss
            
            # Update time
            self.current_time += timedelta(seconds=self.config.timestep)
            
            # Save step metrics
            self.step_metrics.append(step_metrics)
            self.current_step += 1  # Increment the step count

            
            return step_metrics
            
        except Exception as e:
            logging.error(f"Error in simulation step: {e}")
            raise SimulationException(f"Step failed: {e}")

    def _process_matches(self, matches: List[Match]) -> None:
        """Process matches and update metrics"""
        for match in matches:
            # Update vehicle
            vehicle = next(v for v in self.vehicles if v.id == match.get('vehicle_id'))
            vehicle.status = 'BUSY'
            vehicle.current_passengers += len(match.get('request_ids', []))  # Using `get` to avoid missing key errors
            
            # Move vehicle to last dropoff
            last_point = match.get('route', {}).get('trip', {}).get('locations', [])[-1]
            if last_point:
                vehicle.lat = last_point.get('lat')
                vehicle.lon = last_point.get('lon')
                vehicle.route_history.append((vehicle.lat, vehicle.lon))
            
            # Update metrics
            self.metrics.total_distance += match.get('total_distance', 0)
            self.metrics.matched_requests += len(match.get('request_ids', []))
            if len(match.get('request_ids', [])) > 1:
                self.metrics.pooled_rides += 1
            
            # Remove matched requests
            matched_ids = set(match.get('request_ids', []))
            self.active_requests = [r for r in self.active_requests if r.id not in matched_ids]

    def run(self) -> Dict:
        """Run full simulation"""
        logging.info("Starting simulation...")
        
        try:
            while self.current_time < self.end_time:
                step_metrics = self.step()
                
                if len(self.step_metrics) % 100 == 0:
                    self._log_progress()
            
            # Save final results
            self._save_results()
            
            return self.metrics.to_dict()
            
        except Exception as e:
            logging.error(f"Simulation failed: {e}")
            raise

    def _log_progress(self) -> None:
        """Log current simulation progress"""
        progress = (self.current_time - self.start_time) / (self.end_time - self.start_time)
        metrics = self.metrics.to_dict()
        
        logging.info(
            f"Progress: {progress:.1%} | "
            f"Matched Requests: {metrics['matched_requests']} | "
            f"Pooling Rate: {metrics['pooling_rate']:.2%} | "
            f"Vehicle Utilization: {metrics['vehicle_utilization']:.2%}"
        )

def _save_results(self) -> None:
        """Save complete simulation results"""
        results = {
            'config': {
                'num_vehicles': self.config.num_vehicles,
                'max_capacity': self.config.max_capacity,
                'timestep': self.config.timestep,
                'max_time_window': self.config.max_time_window,
                'max_pickup_distance': self.config.max_pickup_distance,
                'max_detour_ratio': self.config.max_detour_ratio
            },
            'metrics': self.metrics.to_dict(),
            'time_series': {
                'timestamps': [m['timestamp'] for m in self.step_metrics],
                'active_vehicles': [m['active_vehicles'] for m in self.step_metrics],
                'active_requests': [m['active_requests'] for m in self.step_metrics],
                'matches': [m.get('matches', 0) for m in self.step_metrics],
                'idle_vehicles': [m.get('idle_vehicles', 0) for m in self.step_metrics]
            },
            'matching_policy_stats': self.matching_policy.stats,
            'dqn_policy_stats': {
                'training_loss': self.dispatch_policy.stats['training_loss'],
                'average_reward': np.mean(self.dispatch_policy.stats['average_reward']),
                'final_epsilon': self.dispatch_policy.epsilon,
                'q_values': {
                    'mean': np.mean([q['mean'] for q in self.dispatch_policy.stats['q_values']]),
                    'max': np.max([q['max'] for q in self.dispatch_policy.stats['q_values']]),
                    'min': np.min([q['min'] for q in self.dispatch_policy.stats['q_values']])
                }
            },
            'vehicle_stats': self._get_vehicle_stats(),
            'request_stats': self._get_request_stats()
        }
        
        # Save main results
        results_path = os.path.join(self.config.output_dir, 'simulation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save DQN model if training was successful
        if len(self.dispatch_policy.stats['training_loss']) > 0:
            model_path = os.path.join(self.config.output_dir, 'dqn_model')
            self.dispatch_policy.save_model(model_path)
        
        logging.info(f"Results saved to {self.config.output_dir}")
        
def _get_vehicle_stats(self) -> Dict:
        """Calculate vehicle-specific statistics"""
        return {
            'route_lengths': [len(v.route_history) for v in self.vehicles],
            'final_locations': [
                {'lat': v.lat, 'lon': v.lon} for v in self.vehicles
            ],
            'utilization': [
                len([r for r in v.route_history if r != v.route_history[0]]) / 
                max(1, len(v.route_history))
                for v in self.vehicles
            ]
        }
        
def _get_request_stats(self) -> Dict:
        """Calculate request-specific statistics"""
        return {
            'completion_rate': len(self.completed_requests) / 
                             max(1, self.metrics.total_requests),
            'average_wait_time': self.metrics.total_wait_time / 
                               max(1, len(self.completed_requests)),
            'distance_distribution': {
                'mean': np.mean([r.trip_distance for r in self.completed_requests]),
                'std': np.std([r.trip_distance for r in self.completed_requests]),
                'min': min([r.trip_distance for r in self.completed_requests]),
                'max': max([r.trip_distance for r in self.completed_requests])
            } if self.completed_requests else {}
        }

def get_current_status(self) -> Dict:
        """Get current simulation status"""
        return {
            'time': self.current_time.isoformat(),
            'active_vehicles': len(self.vehicles),
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'metrics': self.metrics.to_dict(),
            'last_step': self.step_metrics[-1] if self.step_metrics else None
        }
    
def reset(self) -> None:
        """Reset simulation to initial state"""
        self.current_time = self.start_time
        self.vehicles = self._initialize_vehicles()
        self.active_requests = []
        self.completed_requests = []
        self.metrics = SimulationMetrics()
        self.step_metrics = []
        
        # Reset policy states
        self.dispatch_policy.epsilon = self.config.epsilon
        self.matching_policy.stats = defaultdict(int)
        
        logging.info("Simulation reset to initial state")