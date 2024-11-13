import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from valhalla import Actor
from matching_policy import PoolingMatchingPolicy, Vehicle, Request
from dqn_policy import DQNDispatchPolicy, State, Action
from data_loader import DataLoader, TripData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class SimulationConfig:
    data_path: str
    num_vehicles: int
    timestep: int  # seconds
    max_capacity: int
    state_dim: int
    action_dim: int
    batch_size: int = 32
    learning_rate: float = 0.001
    training_epochs: int = 100

class SimulatorDriver:
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Initialize data loader
        self.data_loader = DataLoader(config.data_path)
        
        # Get simulation time boundaries from data
        self.start_time, self.end_time = self.data_loader.get_time_boundaries()
        self.current_time = self.start_time
        
        # Get service area boundaries from data
        self.service_area = self.data_loader.get_service_area()
        
        # Initialize policies
        self.matching_policy = PoolingMatchingPolicy(
            average_trip_length=self.data_loader.get_average_trip_length(),
            average_trip_duration=self.data_loader.get_average_trip_duration()
        )
        
        self.dispatch_policy = DQNDispatchPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            learning_rate=config.learning_rate
        )
        
        # Initialize vehicles based on initial trip locations
        self.vehicles = self._initialize_vehicles()
        
        # Request tracking
        self.active_requests = []
        self.completed_requests = []
        
        # Performance metrics
        self.metrics = {
            'matches': 0,
            'total_wait_time': 0,
            'total_travel_time': 0,
            'total_distance': 0,
            'requests_served': 0,
            'requests_rejected': 0,
            'vehicle_utilization': [],
            'pooling_rate': []
        }
        
        # Initialize Valhalla client
        self.valhalla = Actor('valhalla.json')

    def _initialize_vehicles(self) -> List[Vehicle]:
        """Initialize vehicles based on initial trip locations from data"""
        initial_trips = self.data_loader.get_active_trips(self.start_time)
        vehicles = []
        
        # Use initial trip locations to position vehicles
        for i in range(min(self.config.num_vehicles, len(initial_trips))):
            trip = initial_trips[i]
            vehicles.append(Vehicle(
                id=f'v_{i}',
                lat=trip.lats[0],
                lon=trip.lngs[0],
                status='IDLE',
                capacity=self.config.max_capacity,
                current_passengers=0
            ))
            
        # If we need more vehicles, distribute them randomly
        if len(vehicles) < self.config.num_vehicles:
            remaining = self.config.num_vehicles - len(vehicles)
            vehicles.extend(self._create_random_vehicles(remaining))
            
        return vehicles

    def _create_random_vehicles(self, count: int) -> List[Vehicle]:
        """Create vehicles with random positions within service area"""
        vehicles = []
        for i in range(count):
            lat = np.random.uniform(
                self.service_area['min_lat'],
                self.service_area['max_lat']
            )
            lon = np.random.uniform(
                self.service_area['min_lon'],
                self.service_area['max_lon']
            )
            
            vehicles.append(Vehicle(
                id=f'v_{len(self.vehicles) + i}',
                lat=lat,
                lon=lon,
                status='IDLE',
                capacity=self.config.max_capacity,
                current_passengers=0
            ))
            
        return vehicles

    def _generate_requests_from_data(self) -> List[Request]:
        """Generate requests based on actual trip data"""
        active_trips = self.data_loader.get_active_trips(self.current_time)
        requests = []
        
        for trip in active_trips:
            # Create request from trip data
            request = Request(
                id=trip.trip_id,
                origin_lat=trip.lats[0],
                origin_lon=trip.lngs[0],
                dest_lat=trip.lats[-1],
                dest_lon=trip.lngs[-1],
                pickup_time=int(self.current_time.timestamp())
            )
            requests.append(request)
            
        return requests

    def _get_current_state(self, vehicle: Vehicle) -> State:
        """Get current state for RL agent using actual demand data"""
        # Get nearby vehicles
        nearby_vehicles = [
            (v.lat, v.lon) for v in self.vehicles 
            if v.id != vehicle.id and 
            self._calculate_distance(vehicle, v) < 2000  # 2km radius
        ]
        
        # Get demand matrix from actual data
        demand_matrix = self.data_loader.get_demand_matrix(
            self.current_time,
            grid_size=10  # 10x10 grid
        )
        
        return State(
            vehicle_location=(vehicle.lat, vehicle.lon),
            nearby_vehicles=nearby_vehicles,
            demand_matrix=demand_matrix,
            time_of_day=self.current_time.hour
        )

    def _calculate_reward(self, vehicle: Vehicle, action: Action, 
                         old_state: State, new_state: State) -> float:
        """Calculate reward based on action outcomes"""
        reward = 0
        
        # Reward for picking up passengers
        reward += 10 * (vehicle.current_passengers - 0)  # Assuming 0 was previous passenger count
        
        # Penalty for empty movement
        if vehicle.current_passengers == 0:
            try:
                route = self.valhalla.route([
                    {'lat': old_state.vehicle_location[0], 'lon': old_state.vehicle_location[1]},
                    {'lat': new_state.vehicle_location[0], 'lon': new_state.vehicle_location[1]}
                ])
                distance = route['trip']['summary']['length'] * 1000  # meters
                reward -= distance * 0.001  # Small penalty per meter of empty movement
            except Exception as e:
                logging.error(f"Routing error in reward calculation: {e}")
        
        # Reward for moving to high demand areas
        old_demand = old_state.demand_matrix.sum()
        new_demand = new_state.demand_matrix.sum()
        reward += (new_demand - old_demand) * 5
        
        return reward

    def step(self):
        """Execute one simulation step"""
        # Generate new requests from actual trip data
        new_requests = self._generate_requests_from_data()
        self.active_requests.extend(new_requests)
        
        # Get matches from pooling algorithm
        matches = self.matching_policy.match(self.vehicles, self.active_requests)
        
        # Update vehicle states based on matches
        self._update_vehicle_states(matches)
        
        # Update metrics
        self._update_metrics(matches)
        
        # Get dispatch actions for idle vehicles
        idle_vehicles = [v for v in self.vehicles if v.current_passengers == 0]
        for vehicle in idle_vehicles:
            old_state = self._get_current_state(vehicle)
            action = self.dispatch_policy.get_action(old_state)
            
            # Execute action
            try:
                route = self.valhalla.route([
                    {'lat': vehicle.lat, 'lon': vehicle.lon},
                    {'lat': action.target_location[0], 'lon': action.target_location[1]}
                ])
                vehicle.lat = action.target_location[0]
                vehicle.lon = action.target_location[1]
                
                # Calculate reward
                new_state = self._get_current_state(vehicle)
                reward = self._calculate_reward(vehicle, action, old_state, new_state)
                
                # Store experience
                done = False
                self.dispatch_policy.remember(old_state, action, reward, new_state, done)
                
            except Exception as e:
                logging.error(f"Routing error during dispatch: {e}")
        
        # Train dispatch policy
        if len(self.dispatch_policy.memory) >= self.config.batch_size:
            self.dispatch_policy.train(batch_size=self.config.batch_size)
        
        # Update time
        self.current_time += timedelta(seconds=self.config.timestep)
        
        # Log metrics
        self._log_metrics()

    def _update_metrics(self, matches: List[Dict]):
        """Update simulation metrics based on current matches"""
        self.metrics['matches'] += len(matches)
        
        for match in matches:
            # Update distance and time metrics
            route = match['route']
            self.metrics['total_distance'] += route['trip']['summary']['length']
            self.metrics['total_travel_time'] += route['trip']['summary']['time']
            
            # Update request counts
            self.metrics['requests_served'] += len(match['request_ids'])
            
            # Calculate pooling rate
            pooled_rides = len([m for m in matches if len(m['request_ids']) > 1])
            self.metrics['pooling_rate'].append(pooled_rides / len(matches) if matches else 0)
            
            # Calculate vehicle utilization
            utilized_vehicles = len(set(m['vehicle_id'] for m in matches))
            self.metrics['vehicle_utilization'].append(
                utilized_vehicles / len(self.vehicles)
            )

    def _log_metrics(self):
        """Log current simulation metrics"""
        logging.info(
            f"Time: {self.current_time}, "
            f"Active Requests: {len(self.active_requests)}, "
            f"Completed Requests: {len(self.completed_requests)}, "
            f"Total Distance: {self.metrics['total_distance']:.2f}km, "
            f"Pooling Rate: {np.mean(self.metrics['pooling_rate']):.2%}, "
            f"Vehicle Utilization: {np.mean(self.metrics['vehicle_utilization']):.2%}"
        )

    def run(self):
        """Run full simulation"""
        logging.info("Starting simulation...")
        
        while self.current_time < self.end_time:
            self.step()
            
        logging.info("Simulation completed!")
        self._print_final_metrics()
        
    def _print_final_metrics(self):
        """Print final simulation metrics"""
        logging.info("\nFinal Metrics:")
        logging.info(f"Total Requests Served: {self.metrics['requests_served']}")
        logging.info(f"Total Distance: {self.metrics['total_distance']:.2f}km")
        logging.info(f"Average Pooling Rate: {np.mean(self.metrics['pooling_rate']):.2%}")
        logging.info(f"Average Vehicle Utilization: {np.mean(self.metrics['vehicle_utilization']):.2%}")
        logging.info(f"Total Matches: {self.metrics['matches']}")

if __name__ == "__main__":
    # Example configuration
    config = SimulationConfig(
        data_path="path/to/your/trip_data.jsonl",
        num_vehicles=100,
        timestep=60,  # 1 minute
        max_capacity=4,
        state_dim=124,  # Needs to match your state representation
        action_dim=100  # Needs to match your action space
    )
    
    simulator = SimulatorDriver(config)
    simulator.run()