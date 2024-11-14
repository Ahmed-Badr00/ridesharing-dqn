from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from .types import Vehicle, Request, ServiceArea
from .utils import calculate_distance, get_route, create_grid_index

class PoolingMatchingPolicy:
    def __init__(
        self,
        max_wait_time: int = 300,      # 5 minutes in seconds
        max_detour_ratio: float = 1.5,
        max_pickup_distance: float = 2000,  # 2km
        max_time_window: int = 15,     # 15 minutes
        min_sharing_distance: float = 1.0,   # 1km minimum trip distance for sharing
        service_area: Optional[ServiceArea] = None

    ):
        self.max_wait_time = max_wait_time
    
        self.max_detour_ratio = max_detour_ratio
        self.max_pickup_distance = max_pickup_distance
        self.max_time_window = max_time_window
        self.min_sharing_distance = min_sharing_distance
        self.service_area = service_area  # Store service_area if needed

        # Stats tracking
        self.stats = defaultdict(int)

    def match(self, vehicles: List[Vehicle], requests: List[Request]) -> List[Dict]:
        """Match vehicles to requests considering pooling constraints"""
        if not vehicles or not requests:
            return []

        matches = []
        unmatched_requests = requests.copy()
        
        # Sort requests by pickup time and distance
        unmatched_requests.sort(key=lambda r: (r.pickup_time, r.trip_distance))
        
        # Find available vehicles
        available_vehicles = [v for v in vehicles if v.current_passengers < v.capacity]
        
        # Create spatial index for vehicles
        vehicle_grid = self._create_vehicle_grid(available_vehicles)
        
        while unmatched_requests and available_vehicles:
            request = unmatched_requests[0]
            best_match = self._find_best_match(request, available_vehicles, vehicle_grid)
            
            if best_match:
                matches.append(best_match)
                self._update_after_match(best_match, unmatched_requests, 
                                      available_vehicles, vehicle_grid)
            else:
                unmatched_requests.pop(0)
                self.stats['unmatched_requests'] += 1
                
        
        return matches

    def _create_vehicle_grid(self, vehicles: List[Vehicle]) -> Dict:
        """Create grid-based spatial index for vehicles"""
        points = [{'id': v.id, 'lat': v.lat, 'lon': v.lon, 'vehicle': v} 
                 for v in vehicles]
        return create_grid_index(points, grid_size=10)

    def _find_best_match(
        self, 
        request: Request,
        vehicles: List[Vehicle],
        vehicle_grid: Dict
    ) -> Optional[Dict]:
        """Find best vehicle and potential shared rides for a request"""
        best_score = float('inf')
        best_match = None
        
        # Get nearby vehicles
        nearby_vehicles = self._get_nearby_vehicles(request, vehicle_grid)
        
        for vehicle in nearby_vehicles:
            # Check basic feasibility
            if not self._is_feasible_match(vehicle, request):
                continue
            
            # Try single ride first
            single_match = self._try_single_match(vehicle, request)
            if single_match:
                score = self._calculate_match_score(single_match, pooled=False)
                if score < best_score:
                    best_score = score
                    best_match = single_match
            
            # Try pooling if trip is long enough
            if (request.trip_distance >= self.min_sharing_distance and 
                len(vehicle.current_requests) > 0):
                pooled_match = self._try_pooled_match(vehicle, request)
                if pooled_match:
                    score = self._calculate_match_score(pooled_match, pooled=True)
                    if score < best_score:
                        best_score = score
                        best_match = pooled_match
        
        return best_match

    def _is_feasible_match(self, vehicle: Vehicle, request: Request) -> bool:
        """Check if vehicle can feasibly serve request"""
        # Check capacity
        if vehicle.current_passengers >= vehicle.capacity:
            self.stats['capacity_constraints_violated'] += 1
            return False
        
        # Check pickup distance
        pickup_distance = calculate_distance(
            vehicle.lat, vehicle.lon,
            request.origin_lat, request.origin_lon
        )
        if pickup_distance * 1000 > self.max_pickup_distance:  # Convert to meters
            self.stats['distance_constraints_violated'] += 1
            return False
        
        # Check time compatibility with existing requests
        if vehicle.current_requests:
            time_compatible = all(
                abs((request.pickup_time - r.pickup_time).total_seconds()) <= 
                self.max_time_window * 60
                for r in vehicle.current_requests
            )
            if not time_compatible:
                self.stats['time_constraints_violated'] += 1
                return False
        
        return True

    def _try_single_match(self, vehicle: Vehicle, request: Request) -> Optional[Dict]:
        """Try matching single request to vehicle"""
        route = get_route([
            {'lat': vehicle.lat, 'lon': vehicle.lon},
            {'lat': request.origin_lat, 'lon': request.origin_lon},
            {'lat': request.dest_lat, 'lon': request.dest_lon}
        ])
        
        if not route:
            return None
            
        total_distance = route['trip']['summary']['length']
        total_duration = route['trip']['summary']['time']
        
        # Check if duration is acceptable
        if total_duration > self.max_wait_time:
            return None
            
        return {
            'vehicle_id': vehicle.id,
            'request_ids': [request.id],
            'route': route,
            'total_distance': total_distance,
            'total_duration': total_duration,
            'pickup_times': [request.pickup_time],
            'detour_ratio': total_distance / request.trip_distance
        }

    def _try_pooled_match(self, vehicle: Vehicle, request: Request) -> Optional[Dict]:
        """Try matching request with vehicle's existing requests"""
        # Create waypoints including all pickups and dropoffs
        waypoints = self._create_pooled_waypoints(vehicle, request)
        
        best_route = None
        min_total_distance = float('inf')
        
        # Try different orderings of waypoints
        for ordered_waypoints in self._get_feasible_orderings(waypoints, vehicle, request):
            route = get_route(ordered_waypoints)
            if route:
                total_distance = route['trip']['summary']['length']
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_route = route
        
        if best_route:
            total_duration = best_route['trip']['summary']['time']
            total_distance = best_route['trip']['summary']['length']
            
            # Calculate combined trip distance
            combined_distance = (request.trip_distance + 
                sum(r.trip_distance for r in vehicle.current_requests))
            
            return {
                'vehicle_id': vehicle.id,
                'request_ids': [r.id for r in vehicle.current_requests] + [request.id],
                'route': best_route,
                'total_distance': total_distance,
                'total_duration': total_duration,
                'pickup_times': [r.pickup_time for r in vehicle.current_requests] + 
                              [request.pickup_time],
                'detour_ratio': total_distance / combined_distance
            }
        
        return None

    def _calculate_match_score(self, match: Dict, pooled: bool = False) -> float:
        """Calculate score for a match"""
        # Base score from total distance and duration
        distance_score = match['total_distance'] / 10  # Normalize by 10km
        duration_score = match['total_duration'] / self.max_wait_time
        
        # Detour penalty
        detour_penalty = max(0, match['detour_ratio'] - 1) * 2
        
        # Pooling bonus (negative because lower score is better)
        pooling_bonus = -0.2 if pooled else 0
        
        return 0.4 * distance_score + 0.4 * duration_score + 0.2 * detour_penalty + pooling_bonus

    def _get_nearby_vehicles(self, request: Request, vehicle_grid: Dict) -> List[Vehicle]:
        """Get vehicles near the request location"""
        nearby_vehicles = []
        
        # Get points around request
        request_point = {'lat': request.origin_lat, 'lon': request.origin_lon}
        neighbors = self._get_grid_neighbors(request_point, vehicle_grid)
        
        # Collect vehicles from neighboring cells
        for cell in neighbors:
            if cell in vehicle_grid:
                nearby_vehicles.extend(
                    point['vehicle'] for point in vehicle_grid[cell]
                )
        
        return nearby_vehicles

    def _get_grid_neighbors(self, point: Dict, grid: Dict) -> List[Tuple[int, int]]:
        """Get neighboring grid cells for a point"""
        grid_size = 10  # Same as used in create_grid_index
        lat_idx = int((point['lat'] - ServiceArea.min_lat) / 
                     (ServiceArea.max_lat - ServiceArea.min_lat) * 
                     (grid_size - 1))
        lon_idx = int((point['lon'] - ServiceArea.min_lon) / 
                     (ServiceArea.max_lon - ServiceArea.min_lon) * 
                     (grid_size - 1))
        
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                cell = (lat_idx + i, lon_idx + j)
                if 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size:
                    neighbors.append(cell)
        
        return neighbors

    def _create_pooled_waypoints(self, vehicle: Vehicle, new_request: Request) -> List[Dict]:
        """Create waypoints for pooled route"""
        waypoints = [{'lat': vehicle.lat, 'lon': vehicle.lon}]  # Current location
        
        # Add new request
        waypoints.append({'lat': new_request.origin_lat, 'lon': new_request.origin_lon})
        waypoints.append({'lat': new_request.dest_lat, 'lon': new_request.dest_lon})
        
        # Add existing requests
        for request in vehicle.current_requests:
            waypoints.append({'lat': request.dest_lat, 'lon': request.dest_lon})
        
        return waypoints

    def _get_feasible_orderings(self, waypoints: List[Dict], 
                              vehicle: Vehicle, 
                              new_request: Request) -> List[List[Dict]]:
        """Get feasible orderings of waypoints respecting constraints"""
        # For simplicity, just try original and reverse order
        # Could be enhanced with more sophisticated algorithms
        return [waypoints, list(reversed(waypoints[1:])) + [waypoints[0]]]

    def _update_after_match(self, match: Dict, unmatched_requests: List[Request],
                          available_vehicles: List[Vehicle], 
                          vehicle_grid: Dict) -> None:
        """Update states after a match is made"""
        # Update vehicle
        vehicle = next(v for v in available_vehicles if v.id == match['vehicle_id'])
        vehicle.current_passengers += len(match['request_ids'])
        
        # Remove matched requests
        matched_ids = set(match['request_ids'])
        unmatched_requests[:] = [r for r in unmatched_requests if r.id not in matched_ids]
        
        # Update vehicle grid
        if vehicle.current_passengers >= vehicle.capacity:
            available_vehicles.remove(vehicle)
            for cell in vehicle_grid.values():
                cell[:] = [p for p in cell if p['id'] != vehicle.id]
        
        # Update stats
        self.stats['total_matches'] += 1
        if len(match['request_ids']) > 1:
            self.stats['pooled_matches'] += 1