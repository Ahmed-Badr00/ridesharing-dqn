import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class TripData:
    trip_id: str
    driver_id: str
    time_gaps: List[float]
    dist_gaps: List[float]
    lats: List[float]
    lngs: List[float]
    total_distance: float
    total_time: float
    week_id: int
    time_id: float
    date_id: int
    segment_id: int

class DataLoader:
    def __init__(self, data_path: str):
        """Initialize data loader with path to data file"""
        self.data_path = data_path
        self.trips: Dict[str, TripData] = {}
        self.load_data()

    def load_data(self):
        """Load trip data from file"""
        with open(self.data_path, 'r') as f:
            for line in f:
                trip_dict = json.loads(line.strip())
                trip_data = TripData(
                    trip_id=trip_dict['trip_id'],
                    driver_id=trip_dict['driverID'],
                    time_gaps=trip_dict['time_gap'],
                    dist_gaps=trip_dict['dist_gap'],
                    lats=trip_dict['lats'],
                    lngs=trip_dict['lngs'],
                    total_distance=trip_dict['dist'],
                    total_time=trip_dict['trip_time'],
                    week_id=trip_dict['weekID'],
                    time_id=trip_dict['timeID'],
                    date_id=trip_dict['dateID'],
                    segment_id=trip_dict['segmentID']
                )
                self.trips[trip_data.trip_id] = trip_data

    def get_active_trips(self, current_time: datetime) -> List[TripData]:
        """Get trips that are active at the current time
        Based on timeID (hour of week)"""
        current_week_hour = current_time.weekday() * 24 + current_time.hour
        return [
            trip for trip in self.trips.values()
            if abs(trip.time_id - current_week_hour) < 1  # Within 1 hour window
        ]

    def get_trip_trajectory(self, trip_id: str) -> Optional[pd.DataFrame]:
        """Get complete trajectory for a trip"""
        if trip_id not in self.trips:
            return None
            
        trip = self.trips[trip_id]
        return pd.DataFrame({
            'time': trip.time_gaps,
            'distance': trip.dist_gaps,
            'lat': trip.lats,
            'lng': trip.lngs
        })

    def get_driver_trips(self, driver_id: str) -> List[TripData]:
        """Get all trips for a specific driver"""
        return [
            trip for trip in self.trips.values()
            if trip.driver_id == driver_id
        ]

    def get_demand_matrix(self, current_time: datetime, grid_size: int = 10) -> np.ndarray:
        """Create demand matrix based on trip starts in current time window"""
        active_trips = self.get_active_trips(current_time)
        demand = np.zeros((grid_size, grid_size))
        
        for trip in active_trips:
            # Use first point as pickup location
            lat, lng = trip.lats[0], trip.lngs[0]
            
            # Convert to grid coordinates
            x = int((lat - min(trip.lats)) / (max(trip.lats) - min(trip.lats)) * (grid_size-1))
            y = int((lng - min(trip.lngs)) / (max(trip.lngs) - min(trip.lngs)) * (grid_size-1))
            
            x = min(max(x, 0), grid_size-1)
            y = min(max(y, 0), grid_size-1)
            
            demand[x, y] += 1
            
        return demand

    def get_time_boundaries(self) -> tuple:
        """Get the earliest and latest times in the dataset"""
        min_week = min(trip.week_id for trip in self.trips.values())
        max_week = max(trip.week_id for trip in self.trips.values())
        min_time = min(trip.time_id for trip in self.trips.values())
        max_time = max(trip.time_id for trip in self.trips.values())
        
        base_date = datetime(2024, 1, 1)  # Example base date
        
        start_time = base_date + timedelta(weeks=min_week, hours=int(min_time))
        end_time = base_date + timedelta(weeks=max_week, hours=int(max_time))
        
        return start_time, end_time

    def get_service_area(self) -> Dict[str, float]:
        """Get the geographical boundaries of the service area"""
        all_lats = [lat for trip in self.trips.values() for lat in trip.lats]
        all_lngs = [lng for trip in self.trips.values() for lng in trip.lngs]
        
        return {
            'min_lat': min(all_lats),
            'max_lat': max(all_lats),
            'min_lon': min(all_lngs),
            'max_lon': max(all_lngs)
        }

    def get_average_trip_length(self) -> float:
        """Get average trip distance"""
        return np.mean([trip.total_distance for trip in self.trips.values()])

    def get_average_trip_duration(self) -> float:
        """Get average trip duration"""
        return np.mean([trip.total_time for trip in self.trips.values()])
