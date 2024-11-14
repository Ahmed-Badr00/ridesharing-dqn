import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)

class DataCleaner:
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize data cleaner with Amman's actual bounds"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Amman's actual bounds
        self.min_lon = 35.715  # Western bound
        self.max_lon = 36.25   # Eastern bound
        self.min_lat = 31.668  # Southern bound
        self.max_lat = 32.171  # Northern bound
        
        # Define major districts/areas in Amman for validation
        self.MAJOR_AREAS = [
            # City center and surroundings
            (31.952, 35.933),  # Downtown Amman
            (31.977, 35.843),  # West Amman
            (31.898, 35.898),  # South Amman
            (32.013, 35.911),  # North Amman
            (31.925, 35.945),  # East Amman
            # Major districts
            (31.957, 35.856),  # Abdali
            (31.982, 35.883),  # Shmeisani
            (31.942, 35.892),  # Jabal Amman
            (31.989, 35.864),  # Sweifieh
            (31.977, 35.913)   # Jubeiha
        ]

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Amman's bounds"""
        return (self.min_lat <= lat <= self.max_lat and 
                self.min_lon <= lon <= self.max_lon)

    def is_reasonable_trip(self, pickup_lat: float, pickup_lng: float, 
                         dropoff_lat: float, dropoff_lng: float,
                         duration: float, distance: float) -> bool:
        """
        Validate if the trip parameters are reasonable
        
        Args:
            pickup_lat, pickup_lng: Pickup coordinates
            dropoff_lat, dropoff_lng: Dropoff coordinates
            duration: Trip duration in seconds
            distance: Trip distance in kilometers
        
        Returns:
            bool: True if trip parameters seem reasonable
        """
        # Check if coordinates are within bounds
        if not (self.validate_coordinates(pickup_lat, pickup_lng) and 
                self.validate_coordinates(dropoff_lat, dropoff_lng)):
            return False
            
        # Calculate straight-line distance
        straight_dist = np.sqrt(
            (pickup_lat - dropoff_lat)**2 + 
            (pickup_lng - dropoff_lng)**2
        ) * 111  # Rough conversion to km
        
        # Validations:
        # 1. Trip shouldn't be too short or too long
        if not (0.5 <= distance <= 50):  # km
            return False
            
        # 2. Duration should be reasonable (5 min to 2 hours)
        if not (300 <= duration <= 7200):  # seconds
            return False
            
        # 3. Average speed should be reasonable (5 to 80 km/h)
        avg_speed = (distance * 3600) / duration  # km/h
        if not (5 <= avg_speed <= 80):
            return False
            
        # 4. Actual route shouldn't be too much longer than straight line
        if distance > straight_dist * 3:
            return False
            
        return True

    def clean_trip(self, trip_data: dict) -> dict:
        """Clean individual trip data"""
        try:
            # Extract key points
            pickup_lat = trip_data['lats'][0]
            pickup_lng = trip_data['lngs'][0]
            dropoff_lat = trip_data['lats'][-1]
            dropoff_lng = trip_data['lngs'][-1]
            
            # Validate trip parameters
            if not self.is_reasonable_trip(
                pickup_lat, pickup_lng,
                dropoff_lat, dropoff_lng,
                trip_data['trip_time'],
                trip_data['dist']
            ):
                return None
            
            # Calculate actual time and distance
            time_gaps = [t for t in trip_data['time_gap'] if t > 0]
            dist_gaps = [d for d in trip_data['dist_gap'] if d > 0]
            
            actual_time = sum(time_gaps)
            actual_dist = sum(dist_gaps)
            
            # Create cleaned trip data
            cleaned_trip = {
                'trip_id': trip_data['trip_id'],
                'driver_id': trip_data['driverID'],
                'pickup_lat': pickup_lat,
                'pickup_lng': pickup_lng,
                'dropoff_lat': dropoff_lat,
                'dropoff_lng': dropoff_lng,
                'trip_distance': actual_dist,
                'trip_duration': actual_time,
                'week_id': trip_data['weekID'],
                'time_id': trip_data['timeID'],
                'date_id': trip_data['dateID'],
                'segment_id': trip_data['segmentID'],
                # Add some derived metrics
                'straight_line_distance': np.sqrt(
                    (pickup_lat - dropoff_lat)**2 + 
                    (pickup_lng - dropoff_lng)**2
                ) * 111,  # km
                'average_speed': (actual_dist * 3600) / actual_time  # km/h
            }
            
            return cleaned_trip
            
        except Exception as e:
            logging.error(f"Error cleaning trip {trip_data.get('trip_id', 'unknown')}: {e}")
            return None

    def clean_data(self):
        """Clean all data files"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        cleaned_trips = []
        stats = {
            'total_trips': 0,
            'valid_trips': 0,
            'invalid_trips': 0,
            'invalid_coordinates': 0,
            'invalid_duration': 0,
            'invalid_distance': 0,
            'invalid_speed': 0,
            'districts': {area: 0 for area in self.MAJOR_AREAS},
            'hourly_distribution': {h: 0 for h in range(24)},
            'distance_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'total': 0
            },
            'duration_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'total': 0
            }
        }
        
        # Process each input file
        for filename in os.listdir(self.input_dir):
            if not filename.endswith('.json'):
                continue
                
            input_path = os.path.join(self.input_dir, filename)
            logging.info(f"Processing {filename}")
            
            with open(input_path, 'r') as f:
                for line in f:
                    try:
                        stats['total_trips'] += 1
                        trip_data = json.loads(line.strip())
                        cleaned_trip = self.clean_trip(trip_data)
                        
                        if cleaned_trip:
                            cleaned_trips.append(cleaned_trip)
                            stats['valid_trips'] += 1
                            
                            # Update statistics
                            hour = int(trip_data['timeID'] % 24)
                            stats['hourly_distribution'][hour] += 1
                            
                            stats['distance_stats']['min'] = min(
                                stats['distance_stats']['min'], 
                                cleaned_trip['trip_distance']
                            )
                            stats['distance_stats']['max'] = max(
                                stats['distance_stats']['max'], 
                                cleaned_trip['trip_distance']
                            )
                            stats['distance_stats']['total'] += cleaned_trip['trip_distance']
                            
                            stats['duration_stats']['min'] = min(
                                stats['duration_stats']['min'], 
                                cleaned_trip['trip_duration']
                            )
                            stats['duration_stats']['max'] = max(
                                stats['duration_stats']['max'], 
                                cleaned_trip['trip_duration']
                            )
                            stats['duration_stats']['total'] += cleaned_trip['trip_duration']
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON in {filename}: {e}")
                        continue
        
        # Calculate averages
        if stats['valid_trips'] > 0:
            stats['distance_stats']['avg'] = stats['distance_stats']['total'] / stats['valid_trips']
            stats['duration_stats']['avg'] = stats['duration_stats']['total'] / stats['valid_trips']
        
        stats['invalid_trips'] = stats['total_trips'] - stats['valid_trips']
        stats['valid_percentage'] = (stats['valid_trips'] / stats['total_trips'] * 100) if stats['total_trips'] > 0 else 0
        
        # Save cleaned data
        output_file = os.path.join(self.output_dir, 'cleaned_trips.json')
        with open(output_file, 'w') as f:
            json.dump(cleaned_trips, f, indent=2)
            
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'cleaning_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logging.info(f"Cleaning completed. {stats['valid_trips']}/{stats['total_trips']} trips valid ({stats['valid_percentage']:.2f}%)")
        
        return cleaned_trips

if __name__ == "__main__":
    cleaner = DataCleaner(
        input_dir="data/",
        output_dir="data/cleaned_data"
    )
    cleaned_trips = cleaner.clean_data()