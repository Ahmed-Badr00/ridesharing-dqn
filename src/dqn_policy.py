import numpy as np
import logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppresses TensorFlow INFO and WARNING messages
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime

from .types import Vehicle, Request, ServiceArea
from .utils import calculate_distance, get_route, create_grid_index

class State:
    def __init__(self,
                 vehicle_location: Tuple[float, float],
                 demand_matrix: np.ndarray,
                 nearby_vehicles: List[Tuple[float, float]],
                 time_of_day: int,
                 current_time: datetime):
        self.vehicle_location = vehicle_location
        self.demand_matrix = demand_matrix
        self.nearby_vehicles = nearby_vehicles
        self.time_of_day = time_of_day
        self.current_time = current_time

    def to_vector(self, state_dim: int) -> np.ndarray:
        """Convert state to vector representation with padding to match state_dim"""
        # Location features
        loc_features = list(self.vehicle_location)
        
        # Demand features
        demand_flat = self.demand_matrix.flatten()
        demand_features = [
            np.mean(demand_flat),  # average demand
            np.max(demand_flat),   # peak demand
            np.sum(demand_flat),   # total demand
            *demand_flat          # full demand matrix
        ]
        
        # Time features
        hour = self.time_of_day
        time_features = [
            np.sin(2 * np.pi * hour / 24),  # circular time encoding
            np.cos(2 * np.pi * hour / 24),
            hour / 24                        # normalized hour
        ]
        
        # Vehicle density features
        if self.nearby_vehicles:
            density_features = [
                len(self.nearby_vehicles),  # number of nearby vehicles
                np.mean([v[0] for v in self.nearby_vehicles]),  # mean lat
                np.mean([v[1] for v in self.nearby_vehicles]),  # mean lon
                np.std([v[0] for v in self.nearby_vehicles]),   # lat spread
                np.std([v[1] for v in self.nearby_vehicles])    # lon spread
            ]
        else:
            density_features = [0, 0, 0, 0, 0]
        
        # Combine all features
        full_features = loc_features + demand_features + time_features + density_features
        
        # Pad or trim the feature vector to match the expected state_dim
        if len(full_features) < state_dim:
            # Pad with zeros if not enough features
            full_features = np.pad(full_features, (0, state_dim - len(full_features)), 'constant')
        else:
            # Trim if there are too many features
            full_features = full_features[:state_dim]
        
        return np.array(full_features)

class Action:
    def __init__(self, target_location: Tuple[float, float], expected_value: float = 0.0):
        self.target_location = target_location
        self.expected_value = expected_value

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
                 batch_size: int = 32,
                 service_area: Optional[ServiceArea] = None):  # Add service_area as an optional parameter

        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.service_area = service_area
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Action space discretization
        self.action_grid_size = int(np.sqrt(action_dim))
        self.action_locations = self._create_action_grid()
        
        # Value tracking
        self.value_map = {}
        self.visit_count = {}
        
        # Stats tracking
        self.stats = {
            'training_loss': [],
            'average_reward': [],
            'epsilon_history': [],
            'q_values': []
        }

    def _build_network(self) -> tf.keras.Model:
        """Build neural network for Q-function approximation"""
        initializer = tf.keras.initializers.HeUniform()
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,),
                                kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(self.action_dim, activation='linear',
                                kernel_initializer=initializer)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def _create_action_grid(self) -> List[Tuple[float, float]]:
        """Create grid of possible target locations"""
        locations = []
        
        # Create grid within service area
        lat_step = (ServiceArea.max_lat - ServiceArea.min_lat) / self.action_grid_size
        lon_step = (ServiceArea.max_lon - ServiceArea.min_lon) / self.action_grid_size
        
        for i in range(self.action_grid_size):
            for j in range(self.action_grid_size):
                lat = ServiceArea.min_lat + (i + 0.5) * lat_step
                lon = ServiceArea.min_lon + (j + 0.5) * lon_step
                locations.append((lat, lon))
        
        return locations

    def get_action(self, state: State) -> Action:
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action weighted by past values
            weights = [self.value_map.get(loc, 1.0) for loc in self.action_locations]
            weights = np.array(weights) / sum(weights)
            target_location = random.choices(self.action_locations, weights=weights, k=1)[0]
            return Action(target_location=target_location)
        
        # Get Q-values
        state_vector = state.to_vector(self.state_dim)  # Pass self.state_dim here
        q_values = self.q_network.predict(state_vector.reshape(1, -1))[0]
        
        # Track Q-values
        self.stats['q_values'].append({
            'mean': float(np.mean(q_values)),
            'max': float(np.max(q_values)),
            'min': float(np.min(q_values))
        })
        
        # Combine with historical values
        combined_values = []
        for i, loc in enumerate(self.action_locations):
            q_value = q_values[i]
            hist_value = self.value_map.get(loc, 0)
            visit_count = self.visit_count.get(loc, 0)
            
            # UCB-style exploration bonus
            exploration_bonus = np.sqrt(2 * np.log(sum(self.visit_count.values()) + 1) / 
                                    (visit_count + 1))
            
            combined_value = (
                0.6 * q_value +          # Q-value weight
                0.3 * hist_value +       # Historical value weight
                0.1 * exploration_bonus  # Exploration bonus weight
            )
            combined_values.append(combined_value)
        
        best_action_idx = np.argmax(combined_values)
        best_location = self.action_locations[best_action_idx]
        logging.info(f"Selected action with target location {best_location} and expected value {combined_values[best_action_idx]}.")

        
        return Action(
            target_location=best_location,
            expected_value=combined_values[best_action_idx]
        )

    def train(self, batch_size: Optional[int] = None) -> float:
        """Train the network using experience replay"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch with priority
        batch = self._priority_sample(batch_size)
        
        # Prepare batch data
        states = np.array([exp[0].to_vector(self.state_dim) for exp in batch])
        next_states = np.array([exp[3].to_vector(self.state_dim) for exp in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states)
        
        # Next Q-values (from target network)
        next_q_values = self.target_network.predict(next_states)
        
        # Update Q-values with Double DQN
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                target = reward
            else:
                # Double DQN: use main network to select action, target network to evaluate
                next_action = np.argmax(self.q_network.predict(next_state.to_vector(self.state_dim).reshape(1, -1))[0])
                target = reward + self.gamma * next_q_values[i][next_action]
            
            current_q_values[i][self._location_to_idx(action.target_location)] = target
        
        # Train network
        history = self.q_network.fit(states, current_q_values, 
                                   epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Track metrics
        self.stats['training_loss'].append(loss)
        
        # Update target network periodically
        if len(self.memory) % 100 == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.stats['epsilon_history'].append(self.epsilon)
        logging.info(f"Training loss: {loss}")
        logging.info(f"Current epsilon: {self.epsilon}")
 
        return loss

    def remember(self, state: State, action: Action, reward: float,
                next_state: State, done: bool) -> None:
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Update location values
        loc = action.target_location
        if loc not in self.value_map:
            self.value_map[loc] = reward
            self.visit_count[loc] = 1
        else:
            # Exponential moving average for value
            self.value_map[loc] = 0.9 * self.value_map[loc] + 0.1 * reward
            self.visit_count[loc] += 1
        
        # Track average reward
        self.stats['average_reward'].append(reward)

    def update_target_network(self) -> None:
        """Copy weights from Q network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def _priority_sample(self, batch_size: int) -> List:
        """Sample from memory with priority based on reward magnitude"""
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
        
        # Calculate priorities based on reward magnitude
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

    def _location_to_idx(self, location: Tuple[float, float]) -> int:
        """Convert location to action index"""
        distances = [calculate_distance(location[0], location[1], 
                                     loc[0], loc[1]) 
                    for loc in self.action_locations]
        return np.argmin(distances)

    def save_model(self, path: str) -> None:
        """Save the Q-network model"""
        self.q_network.save(path)
        
    def load_model(self, path: str) -> None:
        """Load a saved Q-network model"""
        self.q_network = tf.keras.models.load_model(path)
        self.update_target_network()