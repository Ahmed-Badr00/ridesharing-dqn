from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict
import random
from pathlib import Path
import json

from .types import (
    Vehicle, Request, Match, ServiceArea, 
    SimulationMetrics, SimulationException, State, Action
)
from .utils import (
    calculate_distance, get_route, 
    convert_to_datetime, setup_logging
)
from .data_loader import DataLoader
from .matching_policy import PoolingMatchingPolicy
from .dqn_policy import DQNDispatchPolicy

class SimulatorDriver:
    def __init__(self, config):
        """Initialize simulation driver"""
        self.config = config
        self.config.validate()  # Validate configuration parameters
        
        self.current_step = 0
        self.service_area = ServiceArea()
        
        # Initialize components
        self.data_loader = DataLoader(config.data_path)
        
        self.matching_policy = PoolingMatchingPolicy(
            max_wait_time=config.max_wait_time,
            max_detour_ratio=config.max_detour_ratio,
            max_pickup_distance=config.max_pickup_distance,
            max_time_window=config.max_time_window,
            min_sharing_distance=config.min_sharing_distance,
            service_area=self.service_area
        )
        
        self.dispatch_policy = DQNDispatchPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            memory_size=config.memory_size,
            batch_size=config.batch_size,
            service_area=self.service_area
        )
        
        # Initialize simulation state
        self.start_time, self.end_time = self.data_loader.get_time_boundaries()
        self.current_time = self.start_time
        self.vehicles = self._initialize_vehicles()
        self.active_requests = []
        self.completed_requests = []
        self.metrics = SimulationMetrics()
        self.step_metrics = []

    def _initialize_vehicles(self) -> List[Vehicle]:
            """Initialize vehicles based on demand distribution"""
            initial_trips = self.data_loader.get_active_trips(self.start_time)
            vehicles = []
            # Use initial trip locations for first batch of vehicles
            for i in range(min(self.config.num_vehicles, len(initial_trips))):
                trip = initial_trips[i]
                vehicles.append(Vehicle(
                    id=f'v_{i}',
                    lat=trip.origin_lat,  # Changed from pickup_lat
                    lon=trip.origin_lon,  # Changed from pickup_lng
                    status='IDLE',
                    capacity=self.config.max_capacity,
                    current_passengers=0
                ))
            
            # Distribute remaining vehicles based on demand
            if len(vehicles) < self.config.num_vehicles:
                demand_matrix = self.data_loader.get_demand_matrix(self.start_time)
                remaining = self._distribute_vehicles_by_demand(
                    self.config.num_vehicles - len(vehicles),
                    demand_matrix,
                    start_idx=len(vehicles)
                )
                vehicles.extend(remaining)
            
            logging.info(f"Initialized {len(vehicles)} vehicles")
            return vehicles


    def _distribute_vehicles_by_demand(self, count: int, 
                                     demand_matrix: np.ndarray,
                                     start_idx: int) -> List[Vehicle]:
        """Distribute vehicles according to demand patterns"""
        vehicles = []
        total_demand = demand_matrix.sum()
        
        if total_demand > 0:
            probs = demand_matrix / total_demand
        else:
            probs = np.ones_like(demand_matrix) / demand_matrix.size
            
        grid_size = len(demand_matrix)
        lat_step = (self.service_area.max_lat - self.service_area.min_lat) / grid_size
        lon_step = (self.service_area.max_lon - self.service_area.min_lon) / grid_size
        
        flat_probs = probs.flatten()
        chosen_cells = np.random.choice(len(flat_probs), size=count, p=flat_probs)
        
        for i, cell_idx in enumerate(chosen_cells):
            row = cell_idx // grid_size
            col = cell_idx % grid_size
            
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
        return self.data_loader.get_active_trips(
            self.current_time,
            self.config.max_time_window,
        )

    def _cleanup_expired_requests(self) -> None:
        """Clean up expired requests"""
        current_time = self.current_time
        valid_requests = []
        expired = []
        
        for request in self.active_requests:
            time_diff = (current_time - request.pickup_time).total_seconds() / 60
            
            if -self.config.max_time_window <= time_diff <= self.config.max_time_window:
                valid_requests.append(request)
            else:
                expired.append(request)
        
        removed_count = len(expired)
        if removed_count > 0:
            logging.info(f"Cleaned up {removed_count} expired requests. "
                        f"Active requests: {len(self.active_requests)} -> {len(valid_requests)}")
        
        self.active_requests = valid_requests
        self.metrics.cancelled_requests += len(expired)

    def step(self) -> Dict:
        """Execute one simulation step"""
        step_metrics = {
            'timestamp': self.current_time.isoformat(),
            'initial_active_requests': len(self.active_requests)
        }
        
        try:
            # 1. Clean up expired requests
            self._cleanup_expired_requests()
            
            # 2. Generate new requests with limiting
            new_requests = self._generate_requests()
            if len(new_requests) > self.config.max_new_requests_per_step:
                logging.warning(f"Limiting new requests from {len(new_requests)} "
                              f"to {self.config.max_new_requests_per_step}")
                new_requests = new_requests[:self.config.max_new_requests_per_step]
            
            self.active_requests.extend(new_requests)
            step_metrics['new_requests'] = len(new_requests)
            logging.debug(f"Generated requests: {len(new_requests)}")

            
            # 3. Enforce maximum active requests limit
            if len(self.active_requests) > self.config.max_active_requests:
                logging.warning(f"Too many active requests ({len(self.active_requests)}). "
                              f"Limiting to {self.config.max_active_requests}")
                self.active_requests = self.active_requests[:self.config.max_active_requests]
            
            # 4. Log step info
            logging.info(f"Starting simulation step {self.current_step}.")
            logging.info(f"Number of active requests: {len(self.active_requests)}")
            logging.info(f"Number of available vehicles: "
                        f"{len([v for v in self.vehicles if v.is_available])}")
            
            # 5. Process matches
            available_vehicles = [v for v in self.vehicles if v.is_available()]
            matches = []
            
            if available_vehicles and self.active_requests:
                matches = self.matching_policy.match(available_vehicles, self.active_requests)
            
            step_metrics['matches'] = len(matches)
            
            # 6. Process matches and update states
            if matches:
                self._process_matches(matches)
                
                # Remove matched requests
                matched_request_ids = {req_id for match in matches 
                                    for req_id in match['request_ids']}
                self.active_requests = [req for req in self.active_requests 
                                      if req.id not in matched_request_ids]
            
            # 7. Reposition idle vehicles
            idle_vehicles = [v for v in self.vehicles if v.status == 'IDLE']
            if idle_vehicles:
                self._reposition_vehicles(idle_vehicles)
            step_metrics['idle_vehicles'] = len(idle_vehicles)
            
            # 8. Train dispatch policy
            if (len(self.dispatch_policy.memory) >= self.config.batch_size and 
                self.current_step % 10 == 0):
                loss = self.dispatch_policy.train()
                step_metrics['training_loss'] = loss
            
            # 9. Update time and metrics
            self.current_time += timedelta(seconds=self.config.timestep)
            step_metrics['final_active_requests'] = len(self.active_requests)
            self.step_metrics.append(step_metrics)
            self.current_step += 1
            
            return step_metrics
            
        except Exception as e:
            logging.error(f"Error in simulation step: {e}")
            raise SimulationException(f"Step failed: {e}")

    def _process_matches(self, matches: List[Dict]) -> None:
        """Process matches efficiently"""
        vehicle_updates = defaultdict(list)
        
        # Group updates by vehicle
        for match in matches:
            vehicle_id = match['vehicle_id']
            vehicle_updates[vehicle_id].append(match)
        
        # Update vehicles and metrics
        for vehicle_id, vehicle_matches in vehicle_updates.items():
            vehicle = next(v for v in self.vehicles if v.id == vehicle_id)
            
            total_passengers = sum(len(m['request_ids']) for m in vehicle_matches)
            total_distance = sum(m['total_distance'] for m in vehicle_matches)
            
            # Update vehicle state
            vehicle.status = 'BUSY'
            vehicle.current_passengers += total_passengers
            
            # Update location to last dropoff
            last_match = vehicle_matches[-1]
            if last_location := last_match['route']['trip']['locations'][-1]:
                vehicle.lat = last_location['lat']
                vehicle.lon = last_location['lon']
                vehicle.route_history.append((vehicle.lat, vehicle.lon))
            
            # Update metrics
            self.metrics.total_distance += total_distance
            self.metrics.matched_requests += total_passengers
            if any(len(m['request_ids']) > 1 for m in vehicle_matches):
                self.metrics.pooled_rides += 1

    def _reposition_vehicles(self, idle_vehicles: List[Vehicle]) -> None:
        """Reposition idle vehicles efficiently"""
        batch_size = 20
        for i in range(0, len(idle_vehicles), batch_size):
            batch = idle_vehicles[i:i+batch_size]
            states = [(v, self._get_vehicle_state(v)) for v in batch]
            
            for vehicle, state in states:
                action = self.dispatch_policy.get_action(state)
                self._execute_vehicle_action(vehicle, action, state)

    def _get_vehicle_state(self, vehicle: Vehicle) -> State:
        """Get current state for vehicle"""
        nearby_vehicles = [
            (v.lat, v.lon) for v in self.vehicles
            if v.id != vehicle.id and
            calculate_distance(vehicle.lat, vehicle.lon, v.lat, v.lon) <= 2
        ]
        
        return State(
            vehicle_location=(vehicle.lat, vehicle.lon),
            demand_matrix=self.data_loader.get_demand_matrix(
                self.current_time,
                grid_size=10
            ),
            nearby_vehicles=nearby_vehicles,
            time_of_day=self.current_time.hour,
            current_time=self.current_time
        )

    def _execute_vehicle_action(self, vehicle: Vehicle, 
                              action: Action, 
                              state: State) -> None:
        """Execute repositioning action"""
        route = get_route([
            {'lat': vehicle.lat, 'lon': vehicle.lon},
            {'lat': action.target_location[0], 'lon': action.target_location[1]}
        ])
        
        if route:
            # Update vehicle location
            vehicle.lat = action.target_location[0]
            vehicle.lon = action.target_location[1]
            vehicle.route_history.append((vehicle.lat, vehicle.lon))
            
            # Calculate reward and store experience
            new_state = self._get_vehicle_state(vehicle)
            reward = self._calculate_repositioning_reward(
                (vehicle.lat, vehicle.lon),
                action.target_location,
                route,
                new_state
            )
            
            self.dispatch_policy.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=new_state,
                done=False
            )

    def _calculate_repositioning_reward(
        self,
        old_location: Tuple[float, float],
        new_location: Tuple[float, float],
        route: Dict,
        new_state: State
    ) -> float:
        """Calculate reward for repositioning"""
        distance = route['trip']['summary']['length']
        distance_penalty = -0.1 * distance
        
        demand_matrix = new_state.demand_matrix
        grid_size = len(demand_matrix)
        lat_idx = int((new_location[0] - self.service_area.min_lat) /
                     (self.service_area.max_lat - self.service_area.min_lat) * 
                     (grid_size - 1))
        lon_idx = int((new_location[1] - self.service_area.min_lon) /
                     (self.service_area.max_lon - self.service_area.min_lon) * 
                     (grid_size - 1))
        demand_reward = 5.0 * demand_matrix[lat_idx, lon_idx]
        
        density_penalty = -2.0 * len(new_state.nearby_vehicles)
        
        return distance_penalty + demand_reward + density_penalty

    def run(self) -> Dict:
        """Run full simulation"""
        logging.info("Starting simulation...")
        
        try:
            while self.current_time < self.end_time:
                step_metrics = self.step()
                
                # Log progress periodically
                if self.current_step % 100 == 0:
                    self._log_progress()
                
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
                f"Active Requests: {len(self.active_requests)}"
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
                    'max_detour_ratio': self.config.max_detour_ratio,
                    'min_sharing_distance': self.config.min_sharing_distance,
                    'max_new_requests_per_step': self.config.max_new_requests_per_step,
                    'max_active_requests': self.config.max_active_requests
                },
                'metrics': self.metrics.to_dict(),
                'time_series': {
                    'timestamps': [m['timestamp'] for m in self.step_metrics],
                    'active_vehicles': [m.get('available_vehicles', 0) for m in self.step_metrics],
                    'active_requests': [m.get('final_active_requests', 0) for m in self.step_metrics],
                    'matches': [m.get('matches', 0) for m in self.step_metrics],
                    'idle_vehicles': [m.get('idle_vehicles', 0) for m in self.step_metrics],
                    'new_requests': [m.get('new_requests', 0) for m in self.step_metrics]
                },
                'matching_policy_stats': self.matching_policy.stats,
                'vehicle_stats': self._get_vehicle_stats(),
                'request_stats': self._get_request_stats(),
                'performance_metrics': self._get_performance_metrics()
            }
            
            # Save main results
            results_path = Path(self.config.output_dir) / 'simulation_results.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save DQN model if training was successful
            if self.dispatch_policy.stats.get('training_loss'):
                model_path = Path(self.config.output_dir) / 'models' / 'dqn_model'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.dispatch_policy.save_model(str(model_path))
            
            logging.info(f"Results saved to {self.config.output_dir}")
            
    def _get_vehicle_stats(self) -> Dict:
            """Calculate vehicle-specific statistics"""
            stats = {
                'route_lengths': [],
                'utilization_rates': [],
                'final_locations': [],
                'passenger_counts': []
            }
            
            for vehicle in self.vehicles:
                # Calculate route length
                route_length = len(vehicle.route_history)
                
                # Calculate utilization (ratio of positions with passengers)
                busy_positions = len([pos for pos in vehicle.route_history[1:]
                                if vehicle.status == 'BUSY'])
                utilization = busy_positions / max(1, route_length - 1)
                
                stats['route_lengths'].append(route_length)
                stats['utilization_rates'].append(utilization)
                stats['final_locations'].append({
                    'lat': vehicle.lat,
                    'lon': vehicle.lon
                })
                stats['passenger_counts'].append(vehicle.current_passengers)
            
            return {
                'average_route_length': np.mean(stats['route_lengths']),
                'average_utilization': np.mean(stats['utilization_rates']),
                'total_passenger_count': sum(stats['passenger_counts']),
                'final_locations': stats['final_locations'],
                'utilization_distribution': {
                    'min': float(np.min(stats['utilization_rates'])),
                    'max': float(np.max(stats['utilization_rates'])),
                    'mean': float(np.mean(stats['utilization_rates'])),
                    'std': float(np.std(stats['utilization_rates']))
                }
            }
            
    def _get_request_stats(self) -> Dict:
            """Calculate request-specific statistics"""
            if not self.completed_requests and not self.active_requests:
                return {}
                
            all_requests = self.completed_requests + self.active_requests
            total_requests = len(all_requests)
            
            completion_rate = len(self.completed_requests) / max(1, total_requests)
            
            # Calculate trip distances
            trip_distances = [r.trip_distance for r in all_requests]
            
            return {
                'total_requests': total_requests,
                'completion_rate': completion_rate,
                'distance_distribution': {
                    'mean': float(np.mean(trip_distances)),
                    'std': float(np.std(trip_distances)),
                    'min': float(np.min(trip_distances)),
                    'max': float(np.max(trip_distances))
                } if trip_distances else {}
            }

    def _get_performance_metrics(self) -> Dict:
            """Calculate simulation performance metrics"""
            total_time = (self.current_time - self.start_time).total_seconds()
            steps_per_second = self.current_step / max(1, total_time)
            
            return {
                'total_simulation_time': total_time,
                'total_steps': self.current_step,
                'steps_per_second': steps_per_second,
                'average_requests_per_step': np.mean([m.get('new_requests', 0) 
                                                    for m in self.step_metrics]),
                'average_matches_per_step': np.mean([m.get('matches', 0) 
                                                for m in self.step_metrics])
                # 'peak_active_requests': max(m.get('final_active_requests', 0) 
                #                         for m in self.step_metrics)
            }

    def get_current_status(self) -> Dict:
            """Get current simulation status"""
            return {
                'time': self.current_time.isoformat(),
                'step': self.current_step,
                'active_vehicles': len([v for v in self.vehicles if v.status != 'IDLE']),
                'idle_vehicles': len([v for v in self.vehicles if v.status == 'IDLE']),
                'active_requests': len(self.active_requests),
                'completed_requests': len(self.completed_requests),
                'metrics': self.metrics.to_dict(),
                'last_step': self.step_metrics[-1] if self.step_metrics else None
            }

    def reset(self) -> None:
            """Reset simulation to initial state"""
            # Reset time
            self.current_time = self.start_time
            self.current_step = 0
            
            # Reset vehicles
            self.vehicles = self._initialize_vehicles()
            
            # Reset requests
            self.active_requests = []
            self.completed_requests = []
            
            # Reset metrics
            self.metrics = SimulationMetrics()
            self.step_metrics = []
            
            # Reset policies
            self.matching_policy.stats = defaultdict(int)
            self.dispatch_policy.epsilon = self.config.epsilon
            
            logging.info("Simulation reset to initial state")
            
    def validate_state(self) -> bool:
            """Validate simulation state"""
            try:
                # Check time bounds
                if self.current_time < self.start_time or self.current_time > self.end_time:
                    logging.error("Current time out of bounds")
                    return False
                
                # Check vehicle constraints
                for vehicle in self.vehicles:
                    if vehicle.current_passengers > vehicle.capacity:
                        logging.error(f"Vehicle {vehicle.id} exceeds capacity")
                        return False
                    if not ServiceArea.is_valid_location(vehicle.lat, vehicle.lon):
                        logging.error(f"Vehicle {vehicle.id} outside service area")
                        return False
                
                # Check request constraints
                for request in self.active_requests:
                    if (self.current_time - request.pickup_time).total_seconds() > \
                    self.config.max_time_window * 60:
                        logging.error(f"Request {request.id} exceeds time window")
                        return False
                
                return True
                
            except Exception as e:
                logging.error(f"State validation failed: {e}")
                return False