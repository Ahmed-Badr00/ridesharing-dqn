from dataclasses import dataclass
import json
from typing import Dict, Optional
import os

@dataclass
class SimulationConfig:
    # Data settings
    data_path: str
    output_dir: str = "simulation_results"
    
    # Simulation parameters
    num_vehicles: int = 100
    max_capacity: int = 4
    timestep: int = 60  # seconds
    max_time_window: int = 15  # minutes
    max_pickup_distance: float = 2000  # meters
    max_detour_ratio: float = 1.5
    
    # Matching policy parameters
    max_wait_time: int = 300  # seconds
    min_sharing_distance: float = 1000  # meters
    
    # DQN parameters
    state_dim: int = 128
    action_dim: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 2000
    
    # Service area (Amman)
    min_lat: float = 31.668
    max_lat: float = 32.171
    min_lon: float = 35.715
    max_lon: float = 36.25

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SimulationConfig':
        """Create config from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in SimulationConfig.__dataclass_fields__
        })

    @classmethod
    def from_json(cls, json_path: str) -> 'SimulationConfig':
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def save(self, path: str) -> None:
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> None:
        """Validate configuration parameters"""
        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")
            
        if self.num_vehicles <= 0:
            raise ValueError("num_vehicles must be positive")
            
        if self.max_capacity <= 0:
            raise ValueError("max_capacity must be positive")
            
        if self.timestep <= 0:
            raise ValueError("timestep must be positive")
            
        if self.max_time_window <= 0:
            raise ValueError("max_time_window must be positive")
            
        if self.max_pickup_distance <= 0:
            raise ValueError("max_pickup_distance must be positive")
            
        if self.max_detour_ratio <= 1:
            raise ValueError("max_detour_ratio must be greater than 1")
            
        if not (0 <= self.epsilon <= 1):
            raise ValueError("epsilon must be between 0 and 1")
            
        if not (0 <= self.epsilon_min <= self.epsilon):
            raise ValueError("epsilon_min must be between 0 and epsilon")
            
        if not (0 < self.epsilon_decay < 1):
            raise ValueError("epsilon_decay must be between 0 and 1")

    def create_output_dir(self) -> None:
        """Create output directory structure"""
        directories = [
            self.output_dir,
            os.path.join(self.output_dir, 'models'),
            os.path.join(self.output_dir, 'metrics'),
            os.path.join(self.output_dir, 'logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)