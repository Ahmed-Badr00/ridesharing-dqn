import numpy as np
import tensorflow as tf
from collections import deque
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging
from valhalla import Actor

@dataclass
class State:
    vehicle_location: Tuple[float, float]
    nearby_vehicles: List[Tuple[float, float]]
    demand_matrix: np.ndarray
    time_of_day: int
    time_gaps: List[float] = None
    dist_gaps: List[float] = None
    historical_locations: List[Tuple[float, float]] = None

@dataclass
class Action:
    target_location: Tuple[float, float]
    is_reposition: bool
    expected_value: float = 0.0

class DQNDispatchPolicy:
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Routing
        self.valhalla = Actor('valhalla.json')
        
        # Historical patterns
        self.value_map = {}  # Cache for location values
        self.location_frequency = {}  # Track successful pickups
        
        # Action space discretization
        self.action_grid_size = int(np.sqrt(action_dim))
        self.action_locations = self._create_action_grid()

    def _build_network(self):
        """Build neural network with architecture suited for our state representation"""
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        
        # Process location features
        location_features = tf.keras.layers.Dense(32, activation='relu')(state_input)
        
        # Process demand matrix
        demand_features = tf.keras.layers.Dense(64, activation='relu')(state_input)
        demand_features = tf.keras.layers.Dense(32, activation='relu')(demand_features)
        
        # Process temporal features
        temporal_features = tf.keras.layers.Dense(16, activation='relu')(state_input)
        
        # Combine all features
        combined = tf.keras.layers.Concatenate()(
            [location_features, demand_features, temporal_features]
        )
        
        # Final layers
        x = tf.keras.layers.Dense(128, activation='relu')(combined)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        output = tf.keras.layers.Dense(self.action_dim, activation='linear')(x)
        
        model = tf.keras.Model(inputs=state_input, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def _create_action_grid(self) -> List[Tuple[float, float]]:
        """Create grid of possible target locations"""
        locations = []
        for i in range(self.action_grid_size):
            for j in range(self.action_grid_size):
                # Convert grid positions to lat/lon (example for specific area)
                lat = 31.95 + (i / self.action_grid_size) * 0.1  # Adjust range based on your data
                lon = 35.85 + (j / self.action_grid_size) * 0.1
                locations.append((lat, lon))
        return locations

    def get_action(self, state: State) -> Action:
        """Get action using epsilon-greedy policy with historical value consideration"""
        if random.random() < self.epsilon:
            # Random action but weighted by historical success
            weights = [self.location_frequency.get(loc, 1) for loc in self.action_locations]
            weights = np.array(weights) / sum(weights)
            target_location = random.choices(self.action_locations, weights=weights, k=1)[0]
            return Action(target_location=target_location, is_reposition=True)
            
        # Get Q values
        state_vector = self._process_state(state)
        q_values = self.q_network.predict(state_vector.reshape(1, -1))[0]
        
        # Combine Q-values with historical values
        combined_values = []
        for i, loc in enumerate(self.action_locations):
            q_value = q_values[i]
            hist_value = self.value_map.get(loc, 0)
            combined_value = 0.7 * q_value + 0.3 * hist_value
            combined_values.append(combined_value)
        
        # Get best action
        best_action_idx = np.argmax(combined_values)
        best_location = self.action_locations[best_action_idx]
        
        return Action(
            target_location=best_location,
            is_reposition=False,
            expected_value=combined_values[best_action_idx]
        )

    def _process_state(self, state: State) -> np.ndarray:
        """Convert state to vector representation including historical patterns"""
        # Location features
        loc_x, loc_y = state.vehicle_location
        
        # Nearby vehicle features
        nearby_x = [v[0] for v in state.nearby_vehicles]
        nearby_y = [v[1] for v in state.nearby_vehicles]
        nearby_density = len(nearby_x) / 10  # Normalized density
        
        # Demand features
        demand = state.demand_matrix.flatten()
        demand_mean = np.mean(demand)
        demand_std = np.std(demand)
        
        # Time features
        hour = state.time_of_day / 24.0
        time_sin = np.sin(2 * np.pi * hour)
        time_cos = np.cos(2 * np.pi * hour)
        
        # Historical pattern features
        if state.historical_locations:
            hist_x = [h[0] for h in state.historical_locations]
            hist_y = [h[1] for h in state.historical_locations]
            movement_pattern = [
                np.mean(hist_x), np.mean(hist_y),
                np.std(hist_x), np.std(hist_y)
            ]
        else:
            movement_pattern = [0, 0, 0, 0]
            
        # Time gap features if available
        if state.time_gaps:
            time_features = [
                np.mean(state.time_gaps),
                np.std(state.time_gaps),
                np.max(state.time_gaps)
            ]
        else:
            time_features = [0, 0, 0]
            
        # Distance features if available
        if state.dist_gaps:
            dist_features = [
                np.mean(state.dist_gaps),
                np.std(state.dist_gaps),
                np.max(state.dist_gaps)
            ]
        else:
            dist_features = [0, 0, 0]
        
        # Combine all features
        return np.concatenate([
            [loc_x, loc_y],
            [nearby_density],
            [demand_mean, demand_std],
            [time_sin, time_cos],
            movement_pattern,
            time_features,
            dist_features,
            demand
        ])

    def train(self, batch_size: int):
        """Train the network using experience replay with importance sampling"""
        if len(self.memory) < batch_size:
            return
            
        # Sample batch with priority
        batch = self._priority_sample(batch_size)
        
        states = []
        new_states = []
        for state, action, reward, next_state, done in batch:
            states.append(self._process_state(state))
            new_states.append(self._process_state(next_state))
            
        states = np.array(states)
        new_states = np.array(new_states)
        
        # Get current Q values
        current_q = self.q_network.predict(states)
        
        # Get future Q values
        future_q = self.target_network.predict(new_states)
        
        # Update Q values with double DQN
        next_actions = np.argmax(self.q_network.predict(new_states), axis=1)
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                current_q[i][self._location_to_idx(action.target_location)] = reward
            else:
                current_q[i][self._location_to_idx(action.target_location)] = reward + \
                    self.gamma * future_q[i][next_actions[i]]
                    
        # Train network
        history = self.q_network.fit(states, current_q, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target network periodically
        if len(self.memory) % 100 == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss

    def _priority_sample(self, batch_size: int) -> List:
        """Sample from memory with priority based on reward magnitude"""
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
            
        # Calculate priorities based on rewards
        priorities = []
        for _, _, reward, _, _ in self.memory:
            priorities.append(abs(reward) + 1.0)  # Add 1.0 to ensure non-zero priority
            
        # Convert to probabilities
        probs = np.array(priorities) / sum(priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.memory),
            batch_size,
            p=probs,
            replace=False
        )
        
        return [self.memory[idx] for idx in indices]

    def remember(self, state: State, action: Action, reward: float, 
                next_state: State, done: bool):
        """Store experience in replay memory and update value map"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Update location value map
        loc = action.target_location
        if loc not in self.value_map:
            self.value_map[loc] = reward
        else:
            self.value_map[loc] = 0.9 * self.value_map[loc] + 0.1 * reward
            
        # Update location frequency for successful pickups
        if reward > 0:
            self.location_frequency[loc] = self.location_frequency.get(loc, 0) + 1

    def update_target_network(self):
        """Copy weights from Q network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def _location_to_idx(self, location: Tuple[float, float]) -> int:
        """Convert location to action index"""
        lat, lon = location
        
        # Find closest grid point
        distances = [
            self._calculate_distance(location, loc)
            for loc in self.action_locations
        ]
        return np.argmin(distances)

    def _calculate_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        try:
            route = self.valhalla.route([
                {'lat': loc1[0], 'lon': loc1[1]},
                {'lat': loc2[0], 'lon': loc2[1]}
            ])
            return route['trip']['summary']['length'] * 1000  # Convert to meters
        except Exception as e:
            # Fallback to simple Euclidean distance
            return np.sqrt(
                (loc1[0] - loc2[0])**2 + 
                (loc1[1] - loc2[1])**2
            ) * 111000  # Rough conversion to meters

    def save_model(self, path: str):
        """Save the Q-network model"""
        self.q_network.save(path)
        
    def load_model(self, path: str):
        """Load a saved Q-network model"""
        self.q_network = tf.keras.models.load_model(path)
        self.update_target_network()