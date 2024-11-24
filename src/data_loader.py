import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from .types import TripData, ServiceArea, Request
from .utils import convert_to_datetime, create_grid_index, calculate_distance

class DataLoader:
    def __init__(self, data_path: str):
        """Initialize data loader"""
        self.data_path = data_path
        self.trips: Dict[str, TripData] = {}
        self.time_index: Dict[datetime, List[str]] = {}
        self.demand_cache: Dict[Tuple[datetime, int], np.ndarray] = {}
        
        # Load and validate data
        self.load_data()
        self._build_time_index()
        
        # Log data summary
        self._log_data_summary()

    def load_data(self) -> None:
        """Load trip data from JSON file"""
        try:
            with open(self.data_path, 'r') as f:
                trips_data = json.load(f)
                
            valid_trips = 0
            total_trips = 0
            invalid_locations = 0
            invalid_segments = 0
            logging.debug(f"Loaded {len(self.trips)} trips. Time index contains {len(self.time_index)} unique timestamps.")

            
            for trip_dict in trips_data:
                total_trips += 1
                
                # Only load trips with segment_id = 3
                if trip_dict['segment_id'] != 3:
                    invalid_segments += 1
                    continue
                    
                # Validate coordinates
                if not ServiceArea.is_valid_location(trip_dict['pickup_lat'], trip_dict['pickup_lng']) or \
                   not ServiceArea.is_valid_location(trip_dict['dropoff_lat'], trip_dict['dropoff_lng']):
                    invalid_locations += 1
                    continue
                
                # Calculate straight line distance if not provided
                if 'straight_line_distance' not in trip_dict:
                    trip_dict['straight_line_distance'] = calculate_distance(
                        trip_dict['pickup_lat'], trip_dict['pickup_lng'],
                        trip_dict['dropoff_lat'], trip_dict['dropoff_lng']
                    )
                
                trip_data = TripData(
                    trip_id=trip_dict['trip_id'],
                    driver_id=trip_dict['driver_id'],
                    pickup_lat=trip_dict['pickup_lat'],
                    pickup_lng=trip_dict['pickup_lng'],
                    dropoff_lat=trip_dict['dropoff_lat'],
                    dropoff_lng=trip_dict['dropoff_lng'],
                    average_speed=trip_dict.get('average_speed', 0),
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
            
            logging.info(
                f"Data loading summary:\n"
                f"  Total trips: {total_trips}\n"
                f"  Valid trips: {valid_trips}\n"
                f"  Invalid locations: {invalid_locations}\n"
                f"  Invalid segments: {invalid_segments}"
            )
            
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
        
        logging.info(f"Built time index with {len(self.time_index)} unique timestamps")

    def get_active_trips(self, current_time: datetime, time_window: int = 15) -> List[Request]:
        """Get trips active within time window from current_time as Request objects"""
        active_trip_data = []
        
        # Look through time window
        for minutes in range(-time_window, time_window + 1):
            check_time = current_time + timedelta(minutes=minutes)
            check_time = check_time.replace(second=0, microsecond=0)
            
            if check_time in self.time_index:
                for trip_id in self.time_index[check_time]:
                    active_trip_data.append(self.trips[trip_id])
        
        # Convert TripData to Request objects
        active_requests = []
        for trip in active_trip_data:
            # Convert to datetime
            pickup_time = convert_to_datetime(trip.week_id, trip.date_id, trip.time_id)
            
            request = Request(
                id=trip.trip_id,
                origin_lat=trip.pickup_lat,
                origin_lon=trip.pickup_lng,
                dest_lat=trip.dropoff_lat,
                dest_lon=trip.dropoff_lng,
                pickup_time=pickup_time,
                time_id=trip.time_id,
                week_id=trip.week_id,
                date_id=trip.date_id,
                trip_distance=trip.trip_distance,
                straight_line_distance=trip.straight_line_distance,
                creation_time=current_time,
                status='PENDING'
            )
            active_requests.append(request)
        
        if active_requests:
            logging.debug(f"Found {len(active_requests)} active requests at {current_time}")
        
        return active_requests

    def get_demand_matrix(self, current_time: datetime, grid_size: int = 10) -> np.ndarray:
        """Create demand matrix for current time with caching"""
        cache_key = (current_time.replace(second=0, microsecond=0), grid_size)
        
        # Check cache first
        if cache_key in self.demand_cache:
            return self.demand_cache[cache_key]
        
        # Create new demand matrix
        demand = np.zeros((grid_size, grid_size))
        active_trips = self.get_active_trips(current_time)
        
        if not active_trips:
            return demand
        
        points = [{'lat': trip.origin_lat, 'lon': trip.origin_lon} 
                 for trip in active_trips]
        
        grid_index = create_grid_index(points, grid_size)
        
        # Convert grid index to demand matrix
        for (lat_idx, lon_idx), points in grid_index.items():
            demand[lat_idx, lon_idx] = len(points)
        
        # Cache the result
        self.demand_cache[cache_key] = demand
        
        # Clean cache if too large
        if len(self.demand_cache) > 1000:
            self.demand_cache.clear()
        
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
        
        logging.info(f"Time boundaries: {min_time} to {max_time}")
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
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances))
            },
            'duration_stats': {
                'min': float(np.min(durations)),
                'max': float(np.max(durations)),
                'mean': float(np.mean(durations)),
                'std': float(np.std(durations))
            },
            'speed_stats': {
                'min': float(np.min(speeds)),
                'max': float(np.max(speeds)),
                'mean': float(np.mean(speeds)),
                'std': float(np.std(speeds))
            }
        }

    def _log_data_summary(self) -> None:
        """Log summary statistics of loaded data"""
        stats = self.get_trip_stats()
        if not stats:
            logging.warning("No trip data available for summary")
            return
            
        logging.info(
            f"Data Summary:\n"
            f"  Total trips: {stats['total_trips']}\n"
            f"  Distance (km) - Mean: {stats['distance_stats']['mean']:.2f}, "
            f"Std: {stats['distance_stats']['std']:.2f}\n"
            f"  Duration (s) - Mean: {stats['duration_stats']['mean']:.2f}, "
            f"Std: {stats['duration_stats']['std']:.2f}\n"
            f"  Speed (km/h) - Mean: {stats['speed_stats']['mean']:.2f}, "
            f"Std: {stats['speed_stats']['std']:.2f}"
        )

    def validate_data(self) -> bool:
        """Validate loaded data"""
        try:
            if not self.trips:
                logging.error("No trips loaded")
                return False
                
            if not self.time_index:
                logging.error("Time index not built")
                return False
                
            # Check some trips for basic validity
            for trip in list(self.trips.values())[:10]:
                pickup_time = convert_to_datetime(trip.week_id, trip.date_id, trip.time_id)
                if not isinstance(pickup_time, datetime):
                    logging.error(f"Invalid datetime conversion for trip {trip.trip_id}")
                    return False
                    
                if not ServiceArea.is_valid_location(trip.pickup_lat, trip.pickup_lng):
                    logging.error(f"Invalid pickup location for trip {trip.trip_id}")
                    return False
                    
                if trip.trip_distance <= 0:
                    logging.error(f"Invalid trip distance for trip {trip.trip_id}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            return False