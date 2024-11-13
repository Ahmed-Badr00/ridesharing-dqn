import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from valhalla import Actor
import logging

@dataclass
class Vehicle:
    id: str
    lat: float
    lon: float
    status: str
    capacity: int
    current_passengers: int
    route_history: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.route_history is None:
            self.route_history = [(self.lat, self.lon)]

@dataclass
class Request:
    id: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    pickup_time: int
    time_gaps: Optional[List[float]] = None
    dist_gaps: Optional[List[float]] = None

class PoolingMatchingPolicy:
    def __init__(self, 
                max_wait_time: int = 300,
                max_detour_ratio: float = 1.5,
                max_pickup_distance: float = 2000,
                average_trip_length: float = None,
                average_trip_duration: float = None):
        """Initialize matching policy with constraints
        
        Args:
            max_wait_time: Maximum pickup waiting time in seconds
            max_detour_ratio: Maximum allowed trip time increase factor
            max_pickup_distance: Maximum pickup distance in meters
            average_trip_length: Average trip length from historical data
            average_trip_duration: Average trip duration from historical data
        """
        self.max_wait_time = max_wait_time
        self.max_detour_ratio = max_detour_ratio
        self.max_pickup_distance = max_pickup_distance
        self.avg_trip_length = average_trip_length or 5000  # 5km default
        self.avg_trip_duration = average_trip_duration or 900  # 15 min default
        
        # Initialize Valhalla routing
        self.valhalla = Actor('valhalla.json')
        
        # Cache for route calculations
        self.route_cache = {}
        
    def match(self, vehicles: List[Vehicle], requests: List[Request]) -> List[Dict]:
        """Match vehicles to requests considering pooling constraints"""
        matches = []
        unmatched_requests = requests.copy()
        
        # Sort requests by pickup time
        unmatched_requests.sort(key=lambda r: r.pickup_time)
        
        # Find available vehicles
        available_vehicles = [v for v in vehicles if v.current_passengers < v.capacity]
        
        # Create spatial index for vehicles
        vehicle_locations = self._create_spatial_index(available_vehicles)
        
        while unmatched_requests and available_vehicles:
            best_match = self._find_best_match(unmatched_requests[0], available_vehicles, vehicle_locations)
            
            if best_match:
                vehicle, matched_requests, route = best_match
                
                # Create match command
                match_command = {
                    'vehicle_id': vehicle.id,
                    'request_ids': [r.id for r in matched_requests],
                    'route': route
                }
                matches.append(match_command)
                
                # Update vehicle state
                vehicle.current_passengers += len(matched_requests)
                vehicle.lat = matched_requests[-1].dest_lat
                vehicle.lon = matched_requests[-1].dest_lon
                vehicle.route_history.append((vehicle.lat, vehicle.lon))
                
                # Remove matched requests
                for request in matched_requests:
                    unmatched_requests.remove(request)
                
                # Update available vehicles
                if vehicle.current_passengers >= vehicle.capacity:
                    available_vehicles.remove(vehicle)
                    del vehicle_locations[self._get_grid_cell(vehicle.lat, vehicle.lon)][vehicle.id]
                
            else:
                # No feasible match found for this request
                unmatched_requests.pop(0)
        
        return matches

    def _find_best_match(self, request: Request, vehicles: List[Vehicle], 
                        vehicle_locations: Dict) -> Optional[Tuple[Vehicle, List[Request], Dict]]:
        """Find best vehicle and potential shared rides for a request"""
        best_score = float('inf')
        best_match = None
        
        # Get nearby vehicles using grid cells
        nearby_vehicles = self._get_nearby_vehicles(request, vehicle_locations)
        
        for vehicle in nearby_vehicles:
            # Check basic feasibility
            if not self._is_feasible_pickup(vehicle, request):
                continue
                
            # Try matching just this request
            route = self._get_route([
                (vehicle.lat, vehicle.lon),
                (request.origin_lat, request.origin_lon),
                (request.dest_lat, request.dest_lon)
            ])
            
            if not route:
                continue
                
            # Calculate score (weighted sum of time and distance)
            score = self._calculate_match_score(route, [request])
            
            if score < best_score:
                best_score = score
                best_match = (vehicle, [request], route)
            
            # Try adding this request to vehicle's current passengers
            if vehicle.current_passengers > 0:
                combined_route = self._get_shared_route(vehicle, vehicle.route_history[-1], request)
                if combined_route:
                    shared_score = self._calculate_match_score(combined_route, [request])
                    if shared_score < best_score:
                        best_score = shared_score
                        best_match = (vehicle, [request], combined_route)
        
        return best_match

    def _is_feasible_pickup(self, vehicle: Vehicle, request: Request) -> bool:
        """Check if vehicle can feasibly pickup request"""
        # Check capacity
        if vehicle.current_passengers >= vehicle.capacity:
            return False
            
        try:
            # Calculate pickup time
            pickup_route = self.valhalla.route([
                {'lat': vehicle.lat, 'lon': vehicle.lon},
                {'lat': request.origin_lat, 'lon': request.origin_lon}
            ])
            
            pickup_time = pickup_route['trip']['summary']['time']
            pickup_distance = pickup_route['trip']['summary']['length'] * 1000  # Convert to meters
            
            # Check constraints
            return (pickup_time <= self.max_wait_time and 
                   pickup_distance <= self.max_pickup_distance)
                   
        except Exception as e:
            logging.error(f"Routing error in feasibility check: {e}")
            return False

    def _get_shared_route(self, vehicle: Vehicle, current_location: Tuple[float, float],
                         new_request: Request) -> Optional[Dict]:
        """Get optimal route for shared ride"""
        try:
            # Get all points to visit
            waypoints = [
                {'lat': current_location[0], 'lon': current_location[1]},
                {'lat': new_request.origin_lat, 'lon': new_request.origin_lon},
                {'lat': new_request.dest_lat, 'lon': new_request.dest_lon}
            ]
            
            # Add destinations of current passengers
            for lat, lon in vehicle.route_history[1:]:  # Skip current location
                waypoints.append({'lat': lat, 'lon': lon})
            
            # Try all possible permutations of waypoints
            best_route = None
            best_time = float('inf')
            
            for perm in self._get_feasible_permutations(waypoints):
                try:
                    route = self.valhalla.route(perm)
                    total_time = route['trip']['summary']['time']
                    
                    if total_time < best_time:
                        best_time = total_time
                        best_route = route
                        
                except Exception as e:
                    continue
            
            return best_route
                    
        except Exception as e:
            logging.error(f"Error in shared route calculation: {e}")
            return None

    def _calculate_match_score(self, route: Dict, requests: List[Request]) -> float:
        """Calculate score for a potential match"""
        total_time = route['trip']['summary']['time']
        total_distance = route['trip']['summary']['length'] * 1000  # Convert to meters
        
        # Normalize by average trip metrics
        time_score = total_time / self.avg_trip_duration
        distance_score = total_distance / self.avg_trip_length
        
        # Weighted sum (can be adjusted based on priorities)
        return 0.7 * time_score + 0.3 * distance_score

    def _create_spatial_index(self, vehicles: List[Vehicle]) -> Dict:
        """Create grid-based spatial index for vehicles"""
        index = defaultdict(dict)
        for vehicle in vehicles:
            cell = self._get_grid_cell(vehicle.lat, vehicle.lon)
            index[cell][vehicle.id] = vehicle
        return index

    def _get_grid_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert coordinates to grid cell"""
        # Simple grid with 0.01 degree cells (about 1km)
        return (int(lat * 100), int(lon * 100))

    def _get_nearby_vehicles(self, request: Request, 
                           vehicle_locations: Dict) -> List[Vehicle]:
        """Get vehicles in nearby grid cells"""
        request_cell = self._get_grid_cell(request.origin_lat, request.origin_lon)
        nearby_vehicles = []
        
        # Check surrounding cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (request_cell[0] + dx, request_cell[1] + dy)
                if cell in vehicle_locations:
                    nearby_vehicles.extend(vehicle_locations[cell].values())
                    
        return nearby_vehicles

    def _get_route(self, points: List[Tuple[float, float]]) -> Optional[Dict]:
        """Get route between points using Valhalla"""
        cache_key = tuple(points)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
            
        try:
            waypoints = [{'lat': lat, 'lon': lon} for lat, lon in points]
            route = self.valhalla.route(waypoints)
            self.route_cache[cache_key] = route
            return route
        except Exception as e:
            logging.error(f"Routing error: {e}")
            return None

    def _get_feasible_permutations(self, waypoints: List[Dict]) -> List[List[Dict]]:
        """Get feasible permutations of waypoints considering constraints"""
        # Simple implementation: only consider original order and reverse
        # Could be extended with more sophisticated permutation logic
        return [waypoints, waypoints[::-1]]