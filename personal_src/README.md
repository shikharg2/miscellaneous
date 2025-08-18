# PrimAITE Environment Setup Scripts

This directory contains scripts to demonstrate how to set up and load the PrimAITE cybersecurity simulation environment.

## Scripts Overview

### 1. `simple_env_setup.py`
A minimal example that shows the basic steps to load and run a PrimAITE environment.

**What it does:**
- Loads either UC7 (red vs blue) or data manipulation configuration
- Creates a PrimaiteGymEnv environment
- Resets the environment and takes a sample action
- Shows basic environment interaction

**Usage:**
```bash
cd /Users/shikhar/Desktop/PrimAITE
source venv/bin/activate
python personal_src/simple_env_setup.py
```

### 2. `primaite_env_setup.py`
A comprehensive environment loader class with detailed logging and inspection capabilities.

**What it does:**
- Provides a `PrimAITEEnvironmentLoader` class for advanced environment management
- Supports both UC7 and data manipulation scenarios
- Includes environment inspection, action masking, and detailed logging
- Demonstrates proper setup, reset, and interaction patterns

**Usage:**
```bash
cd /Users/shikhar/Desktop/PrimAITE
source venv/bin/activate
python personal_src/primaite_env_setup.py
```

### 3. `minimal_red_blue_config.py`
Creates a minimal red team vs blue team scenario from scratch using a custom configuration.

**What it does:**
- Defines a minimal cybersecurity scenario with 3 nodes (web server, attacker PC, router)
- Sets up a blue team defender (RL agent) and red team attacker (scripted agent)
- Demonstrates how to create custom scenarios without relying on pre-built configs
- Shows the structure of PrimAITE configuration files

**Usage:**
```bash
cd /Users/shikhar/Desktop/PrimAITE
source venv/bin/activate
python personal_src/minimal_red_blue_config.py
```

## Key Components Demonstrated

### Environment Creation
```python
from primaite.session.environment import PrimaiteGymEnv

# Method 1: Load from YAML file
env = PrimaiteGymEnv(env_config="path/to/config.yaml")

# Method 2: Load from dictionary
env = PrimaiteGymEnv(env_config=config_dict)
```

### Basic Environment Interaction
```python
# Reset environment for new episode
obs, info = env.reset()

# Take an action
action = env.action_space.sample()  # or choose specific action
obs, reward, terminated, truncated, info = env.step(action)

# Check available actions (action masking)
action_masks = env.action_masks()

# Close environment
env.close()
```

### Configuration Structure
A PrimAITE configuration includes:
- **metadata**: Version and plugin information
- **io_settings**: Logging and output settings
- **game**: Episode length, ports, protocols
- **agents**: Blue team (defenders), red team (attackers), green team (users)
- **simulation**: Network topology, nodes, links, applications

## Red vs Blue Team Scenarios

### Blue Team (Defenders)
- **Type**: `proxy-agent` (RL trainable)
- **Team**: `BLUE`
- **Actions**: Node management, application scanning, security responses
- **Objective**: Maintain system security and availability

### Red Team (Attackers)
- **Type**: Various (e.g., `random-agent`, `tap-001`, `tap-003`)
- **Team**: `RED`
- **Actions**: Automated attack patterns, kill chains
- **Objective**: Compromise systems and steal/corrupt data

## Scenarios Available

1. **UC7**: Enterprise network with multiple LANs, realistic topology
2. **UC2**: Data manipulation scenario, simpler setup
3. **Custom**: Create your own scenarios using configuration dictionaries

## Prerequisites

1. PrimAITE must be installed and activated:
   ```bash
   cd /Users/shikhar/Desktop/PrimAITE
   source venv/bin/activate
   ```

2. Run setup if needed:
   ```bash
   primaite setup
   ```

## Next Steps

After running these scripts, you can:
- Train RL agents using Stable-Baselines3 or Ray RLLib
- Create custom cybersecurity scenarios
- Experiment with different red team attack patterns
- Develop and test cybersecurity policies

## For More Information

Check the main PrimAITE notebooks in `src/primaite/notebooks/`:
- `UC7-E2E-Demo.ipynb` - Complete UC7 demonstration
- `Training-an-SB3-Agent.ipynb` - RL training examples
- `UC7-Training.ipynb` - Red vs blue team training
