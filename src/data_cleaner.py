import json
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

class DataExtractor:
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize data extractor"""
        self.input_dir = input_dir
        self.output_dir = output_dir

    def extract_trip_data(self, trip_data: dict) -> dict:
        """Extract key points from individual trip data without validation"""
        try:
            pickup_lat = trip_data['lats'][0]
            pickup_lng = trip_data['lngs'][0]
            dropoff_lat = trip_data['lats'][-1]
            dropoff_lng = trip_data['lngs'][-1]
            
            # Calculate time and distance without any validation checks
            time_gaps = [t for t in trip_data['time_gap'] if t > 0]
            dist_gaps = [d for d in trip_data['dist_gap'] if d > 0]
            
            actual_time = sum(time_gaps)
            actual_dist = sum(dist_gaps)
            
            extracted_trip = {
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
                'straight_line_distance': np.sqrt(
                    (pickup_lat - dropoff_lat)**2 + 
                    (pickup_lng - dropoff_lng)**2
                ) * 111,  # Rough conversion to km
                'average_speed': (actual_dist * 3600) / actual_time if actual_time > 0 else None  # km/h
            }
            
            return extracted_trip
            
        except Exception as e:
            logging.error(f"Error extracting trip {trip_data.get('trip_id', 'unknown')}: {e}")
            return None

    def extract_data(self):
        """Extract all data files without validation"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        extracted_trips = []
        stats = {
            'total_trips': 0,
            'processed_trips': 0,
            'hourly_distribution': {str(h): 0 for h in range(24)},  # Convert hour to string
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
                        extracted_trip = self.extract_trip_data(trip_data)
                        
                        if extracted_trip:
                            extracted_trips.append(extracted_trip)
                            stats['processed_trips'] += 1
                            
                            # Update statistics
                            hour = str(int(trip_data['timeID'] % 24))  # Convert to string
                            stats['hourly_distribution'][hour] += 1
                            
                            stats['distance_stats']['min'] = min(
                                stats['distance_stats']['min'], 
                                extracted_trip['trip_distance']
                            )
                            stats['distance_stats']['max'] = max(
                                stats['distance_stats']['max'], 
                                extracted_trip['trip_distance']
                            )
                            stats['distance_stats']['total'] += extracted_trip['trip_distance']
                            
                            stats['duration_stats']['min'] = min(
                                stats['duration_stats']['min'], 
                                extracted_trip['trip_duration']
                            )
                            stats['duration_stats']['max'] = max(
                                stats['duration_stats']['max'], 
                                extracted_trip['trip_duration']
                            )
                            stats['duration_stats']['total'] += extracted_trip['trip_duration']
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON in {filename}: {e}")
                        continue
        
        # Calculate averages
        if stats['processed_trips'] > 0:
            stats['distance_stats']['avg'] = stats['distance_stats']['total'] / stats['processed_trips']
            stats['duration_stats']['avg'] = stats['duration_stats']['total'] / stats['processed_trips']
        
        # Save extracted data
        output_file = os.path.join(self.output_dir, 'extracted_trips.json')
        with open(output_file, 'w') as f:
            json.dump(extracted_trips, f, indent=2)
            
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'extraction_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logging.info(f"Extraction completed. Processed {stats['processed_trips']} trips from {stats['total_trips']} total trips.")
        
        return extracted_trips

if __name__ == "__main__":
    extractor = DataExtractor(
        input_dir="data/",
        output_dir="data/extracted_data"
    )
    extracted_trips = extractor.extract_data()
