#!/usr/bin/env python3
"""
Simple PrimAITE Environment Setup

This is a minimal example showing how to set up and load the PrimAITE environment
for cybersecurity simulation and RL training.

Usage:
    python simple_env_setup.py
"""

import yaml
from primaite.session.environment import PrimaiteGymEnv
from primaite import PRIMAITE_PATHS


def setup_and_load_environment():
    """Simple function to set up and load PrimAITE environment."""
    
    print("Setting up PrimAITE environment...")
    
    # Method 1: Load UC7 configuration (Red vs Blue team scenario)
    try:
        # Path to UC7 config file
        uc7_config_path = PRIMAITE_PATHS.user_config_path / "example_config/uc7_config.yaml"
        
        if not uc7_config_path.exists():
            print("UC7 config not found in user config, trying package data...")
            # Use the package data directly
            import primaite.config._package_data
            import pkg_resources
            uc7_config_path = pkg_resources.resource_filename("primaite", "config/_package_data/uc7_config.yaml")
        
        # Load and create environment
        with open(uc7_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded config from: {uc7_config_path}")
        
    except Exception as e:
        print(f"UC7 config failed: {e}")
        print("Trying data manipulation config instead...")
        
        # Method 2: Fallback to data manipulation scenario
        from primaite.config.load import data_manipulation_config_path
        config_path = data_manipulation_config_path()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded fallback config from: {config_path}")
    
    # Create the environment
    print("Creating PrimAITE Gym environment...")
    env = PrimaiteGymEnv(env_config=config)
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    # Take a random action
    print("Taking a random action...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Episode terminated: {terminated}")
    print(f"Episode truncated: {truncated}")
    
    # Close environment
    env.close()
    print("Environment closed.")
    
    return env


if __name__ == "__main__":
    setup_and_load_environment()
