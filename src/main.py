import logging
from datetime import datetime
import tensorflow as tf
import os
from .config import SimulationConfig
from .simulator_driver import SimulatorDriver

def setup_logging(debug: bool = False) -> None:
    """Setup basic logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Create config with just the required data_path
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    config = SimulationConfig(
        data_path="data/extracted_trips_01_25.json"  # Modify this to your data path
    )
    
    # Setup basic logging
    setup_logging()
    
    # Create output directory
    config.create_output_dir()
    
    # Log configuration
    logging.info("Starting simulation with configuration:")
    for key, value in config.to_dict().items():
        logging.info(f"  {key}: {value}")
    
    # Run simulation
    try:
        simulator = SimulatorDriver(config)
        results = simulator.run()
        
        # Log results
        logging.info("\nSimulation Results:")
        for key, value in results.items():
            logging.info(f"{key}: {value}")
            
        logging.info("Simulation completed successfully!")
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()