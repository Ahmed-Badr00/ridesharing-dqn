import json
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)

class DataExtractor:
    def __init__(self, input_file: str, output_file: str):
        """Initialize data extractor
        
        Args:
            input_file (str): Path to input JSON file
            output_file (str): Path to output JSON file
        """
        self.input_file = input_file
        self.output_file = output_file
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        """Extract data from single JSON file"""
        if not os.path.exists(self.input_file):
            logging.error(f"Input file {self.input_file} not found")
            return None
            
        extracted_trips = []
        stats = {
            'total_trips': 0,
            'processed_trips': 0,
            'hourly_distribution': {str(h): 0 for h in range(24)},
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
        
        logging.info(f"Processing {self.input_file}")
        
        try:
            with open(self.input_file, 'r') as f:
                for line in f:
                    try:
                        stats['total_trips'] += 1
                        trip_data = json.loads(line.strip())
                        extracted_trip = self.extract_trip_data(trip_data)
                        
                        if extracted_trip:
                            extracted_trips.append(extracted_trip)
                            stats['processed_trips'] += 1
                            
                            # Update statistics
                            hour = str(int(trip_data['timeID'] % 24))
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
                        logging.error(f"Error decoding JSON: {e}")
                        continue
                        
            # Calculate averages if we have processed any trips
            if stats['processed_trips'] > 0:
                stats['distance_stats']['avg'] = stats['distance_stats']['total'] / stats['processed_trips']
                stats['duration_stats']['avg'] = stats['duration_stats']['total'] / stats['processed_trips']
            
                # Reset min values if no trips were processed
                if stats['distance_stats']['min'] == float('inf'):
                    stats['distance_stats']['min'] = 0
                if stats['duration_stats']['min'] == float('inf'):
                    stats['duration_stats']['min'] = 0
            
            # Save extracted trips
            logging.info(f"Saving extracted trips to {self.output_file}")
            with open(self.output_file, 'w') as f:
                json.dump(extracted_trips, f, indent=2)
            
            # Save statistics to a separate file
            output_base = os.path.splitext(self.output_file)[0]
            stats_file = f"{output_base}_stats.json"
            logging.info(f"Saving statistics to {stats_file}")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logging.info(f"Extraction completed. Processed {stats['processed_trips']} trips from {stats['total_trips']} total trips.")
            
            return extracted_trips
            
        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return None

if __name__ == "__main__":
    # Example usage with explicit paths
    input_file = "data/Segmented_Trips_01_25.json"  # Replace with your input file path
    output_file = "extracted_trips_01_25.json"  # Replace with your desired output path
    
    extractor = DataExtractor(input_file, output_file)
    extracted_trips = extractor.extract_data()