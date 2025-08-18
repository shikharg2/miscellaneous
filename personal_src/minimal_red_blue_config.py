#!/usr/bin/env python3
"""
Minimal Red vs Blue Team Configuration

This creates a simple red team vs blue team cybersecurity scenario
with minimal configuration to demonstrate the core concepts.
"""

# Minimal configuration for a red vs blue team scenario
MINIMAL_RED_BLUE_CONFIG = {
    "metadata": {
        "version": "4.0"
    },
    
    "io_settings": {
        "save_agent_actions": False,
        "save_step_metadata": False,
        "save_pcap_logs": False,
        "save_sys_logs": False,
        "save_agent_logs": False,
        "write_sys_log_to_terminal": False
    },
    
    "game": {
        "max_episode_length": 50,
        "ports": ["HTTP", "SSH", "FTP"],
        "protocols": ["TCP", "UDP", "ICMP"]
    },
    
    "agents": [
        # Blue team agent (defender) - RL trainable
        {
            "ref": "blue_defender",
            "team": "BLUE", 
            "type": "proxy-agent",
            "action_space": {
                "action_map": {
                    0: {"action": "do-nothing", "options": {}}
                }
            }
        },
        
        # Red team agent (attacker) - scripted
        {
            "ref": "red_attacker",
            "team": "RED",
            "type": "random-agent",
            "agent_settings": {
                "start_step": 5,
                "frequency": 10,
                "variance": 2
            }
        }
    ],
    
    "simulation": {
        "network": {
            "nodes": [
                {
                    "hostname": "web_server",
                    "type": "server",
                    "ip_address": "192.168.1.10",
                    "subnet_mask": "255.255.255.0",
                    "default_gateway": "192.168.1.1",
                    "services": [
                        {"type": "web-server"}
                    ]
                },
                {
                    "hostname": "attacker_pc",
                    "type": "computer", 
                    "ip_address": "192.168.1.20",
                    "subnet_mask": "255.255.255.0",
                    "default_gateway": "192.168.1.1"
                }
            ],
            "links": [
                "web_server:eth-1<->attacker_pc:eth-1"
            ]
        }
    }
}


def create_minimal_environment():
    """Create a minimal red vs blue environment from scratch."""
    from primaite.session.environment import PrimaiteGymEnv
    from primaite.config.load import data_manipulation_config_path
    import yaml
    
    print("Creating minimal red vs blue team environment...")
    
    try:
        # Load the working data manipulation config as a base
        with open(data_manipulation_config_path(), 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Loaded base configuration successfully")
        
        # Modify the config to use our custom red vs blue setup
        # Update the blue agent with our custom action space
        for agent in config['agents']:
            if agent.get('team') == 'BLUE':
                agent['action_space'] = {
                    "action_map": {
                        0: {"action": "do-nothing", "options": {}},
                        1: {"action": "node-shutdown", "options": {"node_name": "file_server"}},
                        2: {"action": "node-startup", "options": {"node_name": "file_server"}},
                        3: {"action": "node-reset", "options": {"node_name": "file_server"}},
                        4: {"action": "node-application-scan", "options": {"node_name": "file_server", "application_name": "database-client"}},
                        5: {"action": "node-application-fix", "options": {"node_name": "file_server", "application_name": "database-client"}},
                        6: {"action": "node-application-close", "options": {"node_name": "file_server", "application_name": "database-client"}},
                        7: {"action": "node-service-scan", "options": {"node_name": "file_server", "service_name": "ftp-server"}},
                        8: {"action": "node-file-scan", "options": {"node_name": "file_server", "folder_name": "downloads", "file_name": "malware.exe"}},
                        9: {"action": "node-file-delete", "options": {"node_name": "file_server", "folder_name": "downloads", "file_name": "malware.exe"}},
                        10: {"action": "host-nic-disable", "options": {"node_name": "file_server", "nic_num": 0}},
                        11: {"action": "host-nic-enable", "options": {"node_name": "file_server", "nic_num": 0}},
                        12: {"action": "node-shutdown", "options": {"node_name": "user_computer"}},
                        13: {"action": "node-startup", "options": {"node_name": "user_computer"}},
                        14: {"action": "node-application-scan", "options": {"node_name": "user_computer", "application_name": "web-browser"}},
                        15: {"action": "node-application-fix", "options": {"node_name": "user_computer", "application_name": "web-browser"}},
                        16: {"action": "node-os-scan", "options": {"node_name": "file_server"}},
                        17: {"action": "node-file-restore", "options": {"node_name": "file_server", "folder_name": "downloads", "file_name": "malware.exe"}},
                        18: {"action": "node-application-install", "options": {"node_name": "file_server", "application_name": "web-browser"}},
                        19: {"action": "node-application-remove", "options": {"node_name": "file_server", "application_name": "database-client"}},
                    }
                }
                print(f"✓ Updated blue agent '{agent['ref']}' with custom action space")
        
        # Create environment with modified config
        env = PrimaiteGymEnv(env_config=config)
        
        print("✓ Environment created successfully!")
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space: {env.observation_space}")
        
        # Show agent details
        agent = env.agent
        print(f"✓ Agent: {env._agent_name} (Team: {agent.config.team})")
        
        # Print available actions
        print("\n=== Available Actions ===")
        try:
            action_map = agent.action_manager.action_map
            print(f"Action map type: {type(action_map)}")
            if hasattr(agent.action_manager, 'action_space_config'):
                action_config = agent.action_manager.action_space_config
                print(f"Action space config: {action_config}")
            
            # Try to get actions from the config
            if hasattr(agent, 'config') and hasattr(agent.config, 'action_space'):
                action_space_config = agent.config.action_space
                if hasattr(action_space_config, 'action_map'):
                    action_map_config = action_space_config.action_map
                    for action_id, action_info in action_map_config.items():
                        print(f"Action {action_id}: {action_info}")
        except Exception as e:
            print(f"Could not access action map: {e}")
            print("Available action space size:", env.action_space.n)
        
        # Reset and run a few steps with specific actions
        print("\n=== Running Simulation ===")
        obs, info = env.reset()
        print(f"✓ Environment reset. Initial observation shape: {obs.shape}")
        
        # Demonstrate specific cybersecurity actions
        demo_actions = [
            (0, "Do nothing (baseline)"),
            (16, "OS Scan - Check file server operating system"),
            (4, "Application Scan - Scan database client for vulnerabilities"),
            (8, "File Scan - Check for malware.exe"),
            (9, "File Delete - Remove malware.exe"),
            (1, "Node Shutdown - Power down file server"),
            (2, "Node Startup - Power up file server"),
            (10, "Disable NIC - Cut network access"),
            (11, "Enable NIC - Restore network access"),
            (18, "Install Application - Add web browser"),
            (19, "Remove Application - Uninstall database client"),
            (17, "File Restore - Recover deleted malware.exe")
        ]
        
        for step, (action, description) in enumerate(demo_actions):
            if action < env.action_space.n:  # Check if action is valid
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step {step+1}: {description}")
                print(f"  Action: {action}, Reward: {reward:.3f}")
                
                # Show what actually happened by checking the agent history
                if hasattr(agent, 'history') and len(agent.history) > 0:
                    last_action = agent.history[-1]
                    if hasattr(last_action, 'action'):
                        print(f"  Executed: {last_action.action}")
                    if hasattr(last_action, 'response') and hasattr(last_action.response, 'success'):
                        print(f"  Success: {last_action.response.success}")
                
                if terminated or truncated:
                    print("  Episode ended early")
                    break
            else:
                print(f"Step {step+1}: Action {action} not available (max: {env.action_space.n-1})")
        
        # Check action masking
        print("\n=== Action Masking ===")
        action_masks = env.action_masks()
        available_actions = action_masks.sum()
        print(f"Available actions: {available_actions}/{len(action_masks)}")
        print(f"Valid actions: {[i for i, valid in enumerate(action_masks) if valid][:10]}...")
        
        env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    create_minimal_environment()
