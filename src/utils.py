import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from typing import List, Dict, Optional
from .types import ServiceArea

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance between two points in kilometers"""
    R = 6371  # Earth's radius in kilometers
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def get_route(locations: List[Dict], validate: bool = True) -> Optional[Dict]:
    """Get route between locations using routing service"""
    if validate:
        for loc in locations:
            if not ServiceArea.is_valid_location(loc['lat'], loc['lon']):
                logging.warning(f"Invalid location: {loc}")
                return None

    try:
        response = requests.post(
            'http://localhost:8002/route',
            json={
                'locations': locations,
                'costing': 'auto',
                'directions_options': {'units': 'kilometers'}
            },
            headers={'Content-Type': 'application/json'}
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        logging.error(f"Routing error: {e}")
        return None

def convert_to_datetime(week_id: int, date_id: int, time_id: float) -> datetime:
    """Convert week_id, date_id, and time_id to datetime"""
    base_date = datetime(2024, 1, 1)
    
    week_offset = timedelta(weeks=week_id)
    day_offset = timedelta(days=date_id - 1)
    hours = int(time_id)
    minutes = int((time_id % 1) * 60)
    
    return base_date + week_offset + day_offset + timedelta(hours=hours, minutes=minutes)

def setup_logging(filename: str = 'simulation.log', level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler()
        ]
    )

def create_grid_index(points: List[Dict], grid_size: int = 10) -> Dict:
    """Create spatial grid index for points"""
    grid = {}
    
    lat_min, lat_max = ServiceArea.min_lat, ServiceArea.max_lat
    lon_min, lon_max = ServiceArea.min_lon, ServiceArea.max_lon
    
    lat_step = (lat_max - lat_min) / grid_size
    lon_step = (lon_max - lon_min) / grid_size
    
    for point in points:
        lat_idx = int((point['lat'] - lat_min) / lat_step)
        lon_idx = int((point['lon'] - lon_min) / lon_step)
        
        # Ensure within bounds
        lat_idx = min(max(lat_idx, 0), grid_size-1)
        lon_idx = min(max(lon_idx, 0), grid_size-1)
        
        cell = (lat_idx, lon_idx)
        if cell not in grid:
            grid[cell] = []
        grid[cell].append(point)
    
    return grid