#!/usr/bin/env python3
"""
PrimAITE Environment Setup Script

This script demonstrates how to set up and load the PrimAITE cybersecurity simulation environment.
Based on the codebase analysis, this shows the essential components needed to initialize 
a red team vs blue team cybersecurity scenario.

© Crown-owned copyright 2025, Defence Science and Technology Laboratory UK
"""

import yaml
from pathlib import Path
from typing import Dict, Union

# Import required PrimAITE components
from primaite.session.environment import PrimaiteGymEnv
from primaite import PRIMAITE_PATHS, getLogger
from primaite.config.load import load

# Setup logging
logger = getLogger(__name__)


class PrimAITEEnvironmentLoader:
    """Class to handle PrimAITE environment setup and loading."""
    
    def __init__(self):
        """Initialize the environment loader."""
        self.env = None
        self.config = None
        self.scenario_path = None
        
    def setup_primaite(self) -> bool:
        """
        Perform PrimAITE setup (equivalent to `primaite setup` command).
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("Performing PrimAITE setup...")
            
            # Create necessary directories
            PRIMAITE_PATHS.mkdirs()
            
            logger.info("PrimAITE setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"PrimAITE setup failed: {e}")
            return False
    
    def load_uc7_config(self) -> Dict:
        """
        Load the UC7 (Use Case 7) configuration for red vs blue team scenario.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            # Path to UC7 configuration (red team vs blue team scenario)
            self.scenario_path = PRIMAITE_PATHS.user_config_path / "example_config/uc7_config.yaml"
            
            if not self.scenario_path.exists():
                logger.warning("UC7 config not found, trying package data...")
                # Fallback to package data
                from primaite.config._package_data import uc7_config
                self.scenario_path = Path(__file__).parent.parent / "src/primaite/config/_package_data/uc7_config.yaml"
            
            # Load configuration
            self.config = load(self.scenario_path)
            logger.info(f"Loaded UC7 configuration from: {self.scenario_path}")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load UC7 config: {e}")
            raise
    
    def load_data_manipulation_config(self) -> Dict:
        """
        Load the data manipulation (UC2) configuration - simpler scenario.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            from primaite.config.load import data_manipulation_config_path
            
            self.scenario_path = data_manipulation_config_path()
            self.config = load(self.scenario_path)
            logger.info(f"Loaded data manipulation configuration from: {self.scenario_path}")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load data manipulation config: {e}")
            raise
    
    def create_environment(self, config_path: Union[str, Path, Dict] = None) -> PrimaiteGymEnv:
        """
        Create and initialize the PrimAITE Gymnasium environment.
        
        Args:
            config_path: Path to config file or config dictionary. If None, uses loaded config.
            
        Returns:
            PrimaiteGymEnv: Initialized environment
        """
        try:
            if config_path is None:
                if self.config is None:
                    logger.info("No config provided, loading UC7 by default...")
                    self.load_uc7_config()
                config_to_use = self.config
            else:
                config_to_use = config_path
            
            # Create the gymnasium environment
            self.env = PrimaiteGymEnv(env_config=config_to_use)
            
            logger.info("PrimAITE environment created successfully!")
            logger.info(f"Environment details:")
            logger.info(f"  - Action space: {self.env.action_space}")
            logger.info(f"  - Observation space: {self.env.observation_space}")
            logger.info(f"  - Episode counter: {self.env.episode_counter}")
            
            # Get agent information
            agent = self.env.agent
            logger.info(f"  - Agent name: {self.env._agent_name}")
            logger.info(f"  - Agent team: {agent.config.team}")
            logger.info(f"  - Agent type: {agent.config.type}")
            
            return self.env
            
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise
    
    def inspect_scenario(self):
        """Inspect the loaded scenario configuration."""
        if self.config is None:
            logger.warning("No configuration loaded!")
            return
        
        logger.info("=== Scenario Configuration ===")
        
        # Game settings
        if 'game' in self.config:
            game_config = self.config['game']
            logger.info(f"Max episode length: {game_config.get('max_episode_length', 'N/A')}")
            logger.info(f"Supported ports: {game_config.get('ports', [])}")
            logger.info(f"Supported protocols: {game_config.get('protocols', [])}")
        
        # Agents
        if 'agents' in self.config:
            logger.info(f"Number of agents: {len(self.config['agents'])}")
            for i, agent in enumerate(self.config['agents']):
                logger.info(f"  Agent {i+1}:")
                logger.info(f"    - Name: {agent.get('ref', 'N/A')}")
                logger.info(f"    - Team: {agent.get('team', 'N/A')}")
                logger.info(f"    - Type: {agent.get('type', 'N/A')}")
        
        # Simulation network
        if 'simulation' in self.config and 'network' in self.config['simulation']:
            network = self.config['simulation']['network']
            if 'nodes' in network:
                logger.info(f"Network nodes: {len(network['nodes'])}")
            if 'links' in network:
                logger.info(f"Network links: {len(network['links'])}")
    
    def reset_environment(self):
        """Reset the environment for a new episode."""
        if self.env is None:
            logger.error("Environment not created yet!")
            return None
        
        logger.info("Resetting environment...")
        obs, info = self.env.reset()
        logger.info(f"Environment reset. Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        return obs, info
    
    def get_action_masks(self):
        """Get available actions (action masking)."""
        if self.env is None:
            logger.error("Environment not created yet!")
            return None
        
        action_masks = self.env.action_masks()
        available_actions = action_masks.sum()
        logger.info(f"Available actions: {available_actions}/{len(action_masks)}")
        return action_masks


def main():
    """Main function demonstrating environment setup and loading."""
    logger.info("=== PrimAITE Environment Setup Demo ===")
    
    # Create environment loader
    env_loader = PrimAITEEnvironmentLoader()
    
    # Setup PrimAITE
    if not env_loader.setup_primaite():
        logger.error("Setup failed, exiting...")
        return
    
    try:
        # Try to load UC7 configuration (red vs blue team scenario)
        logger.info("\n=== Loading UC7 Configuration (Red vs Blue Team) ===")
        try:
            env_loader.load_uc7_config()
            logger.info("UC7 config loaded successfully!")
        except Exception as e:
            logger.warning(f"UC7 config failed to load: {e}")
            logger.info("Falling back to data manipulation scenario...")
            env_loader.load_data_manipulation_config()
        
        # Inspect the scenario
        env_loader.inspect_scenario()
        
        # Create the environment
        logger.info("\n=== Creating Environment ===")
        env = env_loader.create_environment()
        
        # Reset the environment
        logger.info("\n=== Resetting Environment ===")
        obs, info = env_loader.reset_environment()
        
        # Check action masking
        logger.info("\n=== Action Masking ===")
        action_masks = env_loader.get_action_masks()
        
        logger.info("\n=== Environment Setup Complete! ===")
        logger.info("You can now use the environment for:")
        logger.info("  - Training RL agents")
        logger.info("  - Running red vs blue team simulations")
        logger.info("  - Cybersecurity research")
        
        # Example of environment interaction
        logger.info("\n=== Example Environment Step ===")
        action = env.action_space.sample()  # Random action
        logger.info(f"Taking random action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Step result:")
        logger.info(f"  - Reward: {reward}")
        logger.info(f"  - Terminated: {terminated}")
        logger.info(f"  - Truncated: {truncated}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        # Clean up
        if env_loader.env:
            env_loader.env.close()
            logger.info("Environment closed.")


if __name__ == "__main__":
    main()
