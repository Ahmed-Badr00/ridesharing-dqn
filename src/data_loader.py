import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from .types import TripData, ServiceArea
from .utils import convert_to_datetime, create_grid_index

class DataLoader:
    def __init__(self, data_path: str):
        """Initialize data loader"""
        self.data_path = data_path
        self.trips: Dict[str, TripData] = {}
        self.time_index: Dict[datetime, List[str]] = {}
        
        # Load and validate data
        self.load_data()
        self._build_time_index()

    def load_data(self) -> None:
        """Load trip data from JSON file"""
        try:
            with open(self.data_path, 'r') as f:
                trips_data = json.load(f)
                
            valid_trips = 0
            total_trips = 0
            
            for trip_dict in trips_data:
                total_trips += 1
                
                # Only load trips with segment_id = 3
                if trip_dict['segment_id'] != 3:
                    continue
                    
                # Validate coordinates
                if not ServiceArea.is_valid_location(trip_dict['pickup_lat'], trip_dict['pickup_lng']) or \
                not ServiceArea.is_valid_location(trip_dict['dropoff_lat'], trip_dict['dropoff_lng']):
                    continue
                
                trip_data = TripData(
                    trip_id=trip_dict['trip_id'],
                    driver_id=trip_dict['driver_id'],
                    pickup_lat=trip_dict['pickup_lat'],
                    pickup_lng=trip_dict['pickup_lng'],
                    dropoff_lat=trip_dict['dropoff_lat'],
                    dropoff_lng=trip_dict['dropoff_lng'],
                    average_speed=trip_dict.get('average_speed', 0),  # Ensure this field is provided
                    trip_distance=trip_dict['trip_distance'],
                    trip_duration=trip_dict['trip_duration'],
                    week_id=trip_dict['week_id'],
                    time_id=trip_dict['time_id'],
                    date_id=trip_dict['date_id'],
                    segment_id=trip_dict['segment_id'],
                    straight_line_distance=trip_dict['straight_line_distance']
                )
                
                self.trips[trip_data.trip_id] = trip_data
                valid_trips += 1
                
            logging.info(f"Loaded {valid_trips}/{total_trips} valid trips with segment_id 3")
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def _build_time_index(self) -> None:
        """Build index of trips by time for efficient lookup"""
        self.time_index.clear()
        
        for trip in self.trips.values():
            trip_time = convert_to_datetime(trip.week_id, trip.date_id, trip.time_id)
            
            # Round to nearest minute for indexing
            rounded_time = trip_time.replace(second=0, microsecond=0)
            
            if rounded_time not in self.time_index:
                self.time_index[rounded_time] = []
            self.time_index[rounded_time].append(trip.trip_id)

    def get_active_trips(self, current_time: datetime, time_window: int = 15) -> List[TripData]:
        """Get trips active within time window from current_time"""
        active_trips = []
        
        # Look through time window
        for minutes in range(-time_window, time_window + 1):
            check_time = current_time + timedelta(minutes=minutes)
            check_time = check_time.replace(second=0, microsecond=0)
            
            if check_time in self.time_index:
                for trip_id in self.time_index[check_time]:
                    active_trips.append(self.trips[trip_id])
        
        return active_trips

    def get_demand_matrix(self, current_time: datetime, grid_size: int = 10) -> np.ndarray:
        """Create demand matrix for current time"""
        active_trips = self.get_active_trips(current_time)
        demand = np.zeros((grid_size, grid_size))
        
        points = [{'lat': trip.pickup_lat, 'lon': trip.pickup_lng} 
                 for trip in active_trips]
        
        grid_index = create_grid_index(points, grid_size)
        
        # Convert grid index to demand matrix
        for (lat_idx, lon_idx), points in grid_index.items():
            demand[lat_idx, lon_idx] = len(points)
        
        return demand

    def get_time_boundaries(self) -> Tuple[datetime, datetime]:
        """Get the earliest and latest times in the dataset"""
        if not self.trips:
            raise ValueError("No trips loaded")
            
        min_time = None
        max_time = None
        
        for trip in self.trips.values():
            trip_time = convert_to_datetime(trip.week_id, trip.date_id, trip.time_id)
            
            if min_time is None or trip_time < min_time:
                min_time = trip_time
            if max_time is None or trip_time > max_time:
                max_time = trip_time
        
        return min_time, max_time

    def get_trip_stats(self) -> Dict:
        """Get statistical summary of trips"""
        if not self.trips:
            return {}
            
        distances = [trip.trip_distance for trip in self.trips.values()]
        durations = [trip.trip_duration for trip in self.trips.values()]
        speeds = [trip.average_speed for trip in self.trips.values()]
        
        return {
            'total_trips': len(self.trips),
            'distance_stats': {
                'min': min(distances),
                'max': max(distances),
                'mean': np.mean(distances),
                'std': np.std(distances)
            },
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'mean': np.mean(durations),
                'std': np.std(durations)
            },
            'speed_stats': {
                'min': min(speeds),
                'max': max(speeds),
                'mean': np.mean(speeds),
                'std': np.std(speeds)
            }
        }