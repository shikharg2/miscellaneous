# Caldera Attack Module

A Python module for interfacing with the [Caldera](https://github.com/mitre/caldera) cybersecurity framework to perform automated adversary emulation and attack scenarios.

## Overview

This module provides a comprehensive `CalderaAttack` class that allows you to:

- Connect to and authenticate with a Caldera server
- Retrieve available agents, adversaries, and attack abilities
- Create and manage attack operations
- Execute complete attack scenarios
- Monitor operation progress and retrieve detailed reports

## Features

- **Authentication**: Secure authentication with Caldera server
- **Agent Management**: List and interact with deployed Caldera agents
- **Adversary Profiles**: Access to various adversary emulation profiles
- **Attack Execution**: Create, start, stop, and monitor attack operations
- **Reporting**: Detailed operation reports and status tracking
- **Error Handling**: Comprehensive error handling and logging
- **Flexible Configuration**: Customizable server settings and timeouts

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Prerequisites

- A running Caldera server (default: `http://localhost:8888`)
- Valid Caldera credentials (default username: `red`, password: `admin`)
- At least one deployed Caldera agent for attack execution

## Quick Start

### Basic Usage

```python
from src.caldera_attack_module import CalderaAttack

# Initialize the attack module
caldera = CalderaAttack(
    server_url="http://localhost:8888",
    username="red",
    password="admin"
)

# Authenticate
if caldera.authenticate():
    print("Successfully connected to Caldera!")
    
    # Get available resources
    agents = caldera.get_agents()
    adversaries = caldera.get_adversaries()
    
    print(f"Found {len(agents)} agents and {len(adversaries)} adversaries")
    
    # Execute an attack scenario
    results = caldera.execute_attack_scenario(
        scenario_name="Test_Attack",
        adversary_name="Hunter",
        timeout=300
    )
    
    print(f"Attack completed: {results['success']}")
```

### Running the Example

You can run the provided example script:

```bash
# Basic example
python example_usage.py

# Interactive mode
python example_usage.py --interactive
```

## API Reference

### CalderaAttack Class

#### Constructor

```python
CalderaAttack(server_url="http://localhost:8888", username="red", password="admin", verify_ssl=False)
```

- `server_url`: Base URL of the Caldera server
- `username`: Username for authentication
- `password`: Password for authentication
- `verify_ssl`: Whether to verify SSL certificates

#### Key Methods

##### `authenticate() -> bool`
Authenticate with the Caldera server.

##### `get_agents() -> List[Dict]`
Retrieve all available agents.

##### `get_adversaries() -> List[Dict]`
Retrieve all adversary profiles.

##### `get_abilities() -> List[Dict]`
Retrieve all attack abilities/techniques.

##### `create_operation(name, adversary_id, group="red", planner="atomic") -> Optional[str]`
Create a new attack operation.

##### `start_operation(operation_id) -> bool`
Start an attack operation.

##### `stop_operation(operation_id) -> bool`
Stop a running operation.

##### `get_operation_status(operation_id) -> Dict`
Get the current status of an operation.

##### `execute_attack_scenario(scenario_name, adversary_name="Hunter", target_group="red", timeout=300) -> Dict`
Execute a complete attack scenario with monitoring and reporting.

## Configuration

### Environment Variables

You can set the following environment variables to configure default settings:

```bash
export CALDERA_URL="http://your-caldera-server:8888"
export CALDERA_USERNAME="your-username"
export CALDERA_PASSWORD="your-password"
```

### Custom Configuration

```python
# Custom configuration example
caldera = CalderaAttack(
    server_url="https://your-caldera-server.com",
    username="custom-user",
    password="secure-password",
    verify_ssl=True  # Enable SSL verification for production
)
```

## Attack Scenarios

The module supports various attack scenarios:

1. **Reconnaissance**: Information gathering and network discovery
2. **Initial Access**: Exploitation and foothold establishment
3. **Persistence**: Maintaining access to compromised systems
4. **Privilege Escalation**: Gaining higher-level permissions
5. **Defense Evasion**: Avoiding detection mechanisms
6. **Credential Access**: Harvesting credentials and secrets
7. **Discovery**: Internal network and system enumeration
8. **Lateral Movement**: Moving through the network
9. **Collection**: Data gathering and aggregation
10. **Exfiltration**: Data extraction and theft

## Security Considerations

⚠️ **Important Security Notes:**

1. **Authorized Use Only**: Only use this module in authorized environments
2. **Network Isolation**: Run attacks in isolated/sandboxed networks
3. **Proper Authentication**: Use strong, unique credentials
4. **SSL/TLS**: Enable SSL verification in production environments
5. **Logging**: Monitor and log all attack activities
6. **Cleanup**: Properly clean up operations and artifacts

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify Caldera server is running
   - Check network connectivity
   - Confirm correct server URL and port

2. **Authentication Failed**
   - Verify username and password
   - Check if user has appropriate permissions

3. **No Agents Found**
   - Deploy Caldera agents on target systems
   - Ensure agents can communicate with server

4. **Operation Timeout**
   - Increase timeout value
   - Check agent connectivity
   - Verify adversary profile compatibility

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

caldera = CalderaAttack()
# Debug information will be printed
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and authorized security testing purposes only. Users are responsible for ensuring they have proper authorization before conducting any security testing activities. The authors are not responsible for any misuse of this tool.

## Related Resources

- [Caldera Documentation](https://caldera.readthedocs.io/)
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [Adversary Emulation Library](https://github.com/center-for-threat-informed-defense/adversary_emulation_library)
