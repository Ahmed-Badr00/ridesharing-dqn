# Dynamic Ride-Sharing System with DQN and Pooling

A deep reinforcement learning-based ride-sharing system that uses both DQN for vehicle dispatch and intelligent pooling for ride matching. This implementation is specifically designed for time-series trip data with the following format:

```json
{
    "trip_id": "0012cf835ee80e59fefbe618282b2edc082940ddba6a4658e2626801026e2399",
    "time_gap": [0.0, 10.0, 15.0, ...],
    "dist_gap": [0, 0.0, 0.0, ...],
    "dist": 1.1892123285501033,
    "trip_time": 288.0,
    "driverID": "386ee1784220fe169abb70e3be7ae60de2e180b91e185271d21c064dec7f57aa",
    "weekID": 2,
    "timeID": 348.0,
    "dateID": 30,
    "lats": [31.955, 31.955, ...],
    "lngs": [35.855, 35.855, ...],
    "segmentID": 1
}
```

## Key Features

- DQN-based vehicle dispatch optimization
- Intelligent ride pooling with time and distance constraints
- Valhalla routing integration
- Real-time demand prediction
- Historical pattern learning
- Priority-based experience replay
- Double DQN implementation
- Dynamic action space based on historical success

## Project Structure

```
.
├── src/
│   ├── data_loader.py          # Handles trip data loading and processing
│   ├── matching_policy.py      # Implements pooling algorithm
│   ├── dqn_policy.py          # DQN implementation for dispatch
│   └── simulator_driver.py    # Main simulation coordinator
├── config/
│   ├── valhalla.json         # Valhalla routing configuration
│   └── config.yaml          # Simulation parameters
├── data/
│   └── trips/              # Trip data files
├── models/
│   └── saved/            # Saved model checkpoints
└── logs/                # Simulation logs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ridesharing-dqn.git
cd ridesharing-dqn
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Valhalla:
```bash
docker pull ghcr.io/gis-ops/valhalla:latest
docker run -d -p 8002:8002 --name valhalla ghcr.io/gis-ops/valhalla:latest
```

## Configuration

### Main Configuration (config/config.yaml)
```yaml
simulation:
  data_path: "data/trips/"
  start_date: "2024-01-01"
  end_date: "2024-01-31"
  num_vehicles: 100
  max_capacity: 4
  time_window: 300  # seconds

matching:
  max_wait_time: 300
  max_detour_ratio: 1.5
  max_pickup_distance: 2000

dispatch:
  learning_rate: 0.001
  gamma: 0.95
  epsilon: 1.0
  epsilon_decay: 0.995
  batch_size: 32
  memory_size: 10000
```

## Usage

### 1. Data Preparation
Place your trip data files in the `data/trips/` directory. The system expects JSON files in the format shown above.

### 2. Basic Training Run
```bash
python src/simulator_driver.py --mode train
```

### 3. Evaluation Run
```bash
python src/simulator_driver.py --mode eval --model models/saved/model_name
```

### 4. Custom Configuration Run
```bash
python src/simulator_driver.py --config path/to/config.yaml
```

## Implementation Details

### Data Loader
- Handles time-series trip data
- Processes temporal and spatial features
- Creates demand matrices
- Manages historical patterns

### DQN Policy
The DQN implementation includes:
- State features:
  - Vehicle location
  - Time gaps from actual trips
  - Distance gaps from actual trips
  - Historical demand patterns
  - Time of day features
  
- Action space:
  - Grid-based possible locations
  - Dynamic weighting based on success rates
  
- Training enhancements:
  - Priority experience replay
  - Double DQN architecture
  - Historical value mapping
  - Location frequency tracking

### Pooling Algorithm
The matching policy considers:
- Time constraints from actual trip data
- Distance constraints based on historical patterns
- Vehicle capacity
- Route optimization using Valhalla
- Real-time demand patterns

## Performance Metrics

The system tracks:
```python
metrics = {
    'matches': 0,              # Total successful matches
    'total_wait_time': 0,      # Total customer wait time
    'total_travel_time': 0,    # Total travel time
    'total_distance': 0,       # Total distance covered
    'requests_served': 0,      # Total requests served
    'requests_rejected': 0,    # Total rejected requests
    'vehicle_utilization': [], # Vehicle utilization rate
    'pooling_rate': []        # Rate of successful pooling
}
```

## API Reference

### DataLoader
```python
loader = DataLoader("path/to/data")
trips = loader.get_active_trips(current_time)
demand = loader.get_demand_matrix(current_time)
```

### DQNDispatchPolicy
```python
policy = DQNDispatchPolicy(state_dim=124, action_dim=100)
action = policy.get_action(state)
policy.train(batch_size=32)
```

### PoolingMatchingPolicy
```python
matcher = PoolingMatchingPolicy(max_wait_time=300)
matches = matcher.match(vehicles, requests)
```

## Debugging

### Common Issues

1. Valhalla Connection:
```bash
docker logs valhalla  # Check Valhalla logs
docker restart valhalla  # Restart if needed
```

2. Memory Issues:
```bash
# Reduce batch size in config:
dispatch:
  batch_size: 16
  memory_size: 5000
```

3. Data Loading Issues:
```python
# Enable verbose logging
logging.setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create Pull Request

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:
```bibtex
@software{ridesharing_dqn_2024,
  title={Dynamic Ride-Sharing System with DQN and Pooling},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ridesharing-dqn}
}
```