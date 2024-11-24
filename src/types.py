from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np

@dataclass
class ServiceArea:
    """Geographic boundaries and constraints"""
    min_lat: float = 31.668  # Amman bounds
    max_lat: float = 32.171
    min_lon: float = 35.715
    max_lon: float = 36.25
    
    @staticmethod
    def is_valid_location(lat: float, lon: float) -> bool:
        """Check if location is within service area"""
        return (ServiceArea.min_lat <= lat <= ServiceArea.max_lat and 
                ServiceArea.min_lon <= lon <= ServiceArea.max_lon)

@dataclass
class TripData:
    """Raw trip data"""
    trip_id: str
    driver_id: str
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    trip_distance: float
    trip_duration: float
    week_id: int
    time_id: float
    date_id: int
    segment_id: int
    straight_line_distance: float
    average_speed: float

@dataclass
class Request:
    """Request for service"""
    id: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    pickup_time: datetime
    time_id: float
    week_id: int
    date_id: int
    trip_distance: float
    straight_line_distance: float
    creation_time: datetime = field(default_factory=datetime.now)
    status: str = 'PENDING'  # PENDING, MATCHED, COMPLETED, CANCELLED

@dataclass
class Vehicle:
    """Vehicle state"""
    id: str
    lat: float
    lon: float
    status: str  # IDLE, ASSIGNED, BUSY
    capacity: int
    current_passengers: int
    current_requests: List['Request'] = field(default_factory=list)
    route_history: List[Tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.route_history:
            self.route_history = [(self.lat, self.lon)]

    def is_available(self):
        """Determine if the vehicle is available for new requests."""
        return self.status == 'IDLE' and self.current_passengers < self.capacity

@dataclass
class Match:
    """Matching result between vehicle and requests"""
    vehicle_id: str
    request_ids: List[str]
    route: Dict
    pickup_times: List[datetime]
    distances: List[float]
    total_distance: float
    total_duration: float
    detour_ratio: float

@dataclass
class SimulationMetrics:
    """Metrics tracked during simulation"""
    total_requests: int = 0
    matched_requests: int = 0
    completed_requests: int = 0
    cancelled_requests: int = 0
    total_distance: float = 0
    total_wait_time: float = 0
    total_trip_time: float = 0
    total_empty_distance: float = 0
    pooled_rides: int = 0
    avg_occupancy: float = 0
    vehicle_utilization: float = 0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'total_requests': self.total_requests,
            'matched_requests': self.matched_requests,
            'completed_requests': self.completed_requests,
            'cancelled_requests': self.cancelled_requests,
            'total_distance': round(self.total_distance, 2),
            'total_wait_time': round(self.total_wait_time / 60, 2),  # minutes
            'total_trip_time': round(self.total_trip_time / 60, 2),  # minutes
            'total_empty_distance': round(self.total_empty_distance, 2),
            'pooled_rides': self.pooled_rides,
            'pooling_rate': round(self.pooled_rides / max(self.matched_requests, 1), 3),
            'avg_occupancy': round(self.avg_occupancy, 2),
            'vehicle_utilization': round(self.vehicle_utilization, 3)
        }
@dataclass
class State:
    """State representation for DQN"""
    vehicle_location: Tuple[float, float]
    demand_matrix: np.ndarray
    nearby_vehicles: List[Tuple[float, float]]
    time_of_day: int
    current_time: datetime

    def to_vector(self, state_dim: int) -> np.ndarray:
        """Convert state to vector representation"""
        try:
            # Location features
            loc_features = list(self.vehicle_location)
            
            # Demand features (ensure it's 2D and flatten)
            if len(self.demand_matrix.shape) != 2:
                raise ValueError(f"Demand matrix should be 2D, got shape {self.demand_matrix.shape}")
            demand_flat = self.demand_matrix.flatten()
            demand_features = [
                float(np.mean(demand_flat)),
                float(np.max(demand_flat)),
                float(np.sum(demand_flat))
            ]
            demand_features.extend(demand_flat.tolist())
            
            # Time features
            hour = float(self.time_of_day)
            time_features = [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                hour / 24
            ]
            
            # Vehicle density features
            if self.nearby_vehicles:
                nearby_x = [float(v[0]) for v in self.nearby_vehicles]
                nearby_y = [float(v[1]) for v in self.nearby_vehicles]
                density_features = [
                    float(len(self.nearby_vehicles)),
                    float(np.mean(nearby_x)),
                    float(np.mean(nearby_y)),
                    float(np.std(nearby_x)) if len(nearby_x) > 1 else 0.0,
                    float(np.std(nearby_y)) if len(nearby_y) > 1 else 0.0
                ]
            else:
                density_features = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Combine all features
            full_features = loc_features + demand_features + time_features + density_features
            
            # Convert to numpy array and ensure float32 type
            feature_array = np.array(full_features, dtype=np.float32)
            
            # Pad or trim to match state_dim
            if len(feature_array) < state_dim:
                padded = np.zeros(state_dim, dtype=np.float32)
                padded[:len(feature_array)] = feature_array
                return padded
            else:
                return feature_array[:state_dim]
                
        except Exception as e:
            logging.error(f"Error in State.to_vector: {e}")
            # Return zero vector in case of error
            return np.zeros(state_dim, dtype=np.float32)
        
@dataclass
class Action:
    """Action representation for DQN"""
    target_location: Tuple[float, float]
    expected_value: float = 0.0

class SimulationException(Exception):
    """Custom exception for simulation errors"""
    pass
