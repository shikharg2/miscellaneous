"""
Caldera Attack Module

This module provides a class for interfacing with the Caldera framework
to perform automated adversary emulation and attack scenarios.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime


class CalderaAttack:
    """
    A class to interact with Caldera REST API for performing automated attacks
    and adversary emulation scenarios.
    """
    
    def __init__(self, 
                 server_url: str = "http://localhost:8888",
                 username: str = "red", 
                 password: str = "admin",
                 verify_ssl: bool = False):
        """
        Initialize the Caldera Attack instance.
        
        Args:
            server_url (str): The base URL of the Caldera server
            username (str): Username for authentication
            password (str): Password for authentication
            verify_ssl (bool): Whether to verify SSL certificates
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.verify_ssl = verify_ssl
        self.session = requests.Session()
        self.session.verify = verify_ssl
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Authentication token
        self.auth_token = None
        
    def authenticate(self) -> bool:
        """
        Authenticate with the Caldera server.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            auth_url = f"{self.server_url}/login"
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            response = self.session.post(auth_url, json=auth_data)
            
            if response.status_code == 200:
                self.auth_token = response.cookies.get('session')
                self.logger.info("Successfully authenticated with Caldera server")
                return True
            else:
                self.logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False
    
    def get_agents(self) -> List[Dict]:
        """
        Retrieve all available agents from Caldera.
        
        Returns:
            List[Dict]: List of agent information
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            data = {"index": "agents"}
            
            response = self.session.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                agents = response.json()
                self.logger.info(f"Retrieved {len(agents)} agents")
                return agents
            else:
                self.logger.error(f"Failed to get agents: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving agents: {str(e)}")
            return []
    
    def get_adversaries(self) -> List[Dict]:
        """
        Retrieve all available adversaries (attack profiles).
        
        Returns:
            List[Dict]: List of adversary profiles
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            data = {"index": "adversaries"}
            
            response = self.session.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                adversaries = response.json()
                self.logger.info(f"Retrieved {len(adversaries)} adversaries")
                return adversaries
            else:
                self.logger.error(f"Failed to get adversaries: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving adversaries: {str(e)}")
            return []
    
    def get_abilities(self) -> List[Dict]:
        """
        Retrieve all available abilities (attack techniques).
        
        Returns:
            List[Dict]: List of attack abilities
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            data = {"index": "abilities"}
            
            response = self.session.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                abilities = response.json()
                self.logger.info(f"Retrieved {len(abilities)} abilities")
                return abilities
            else:
                self.logger.error(f"Failed to get abilities: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error retrieving abilities: {str(e)}")
            return []
    
    def create_operation(self, 
                        name: str,
                        adversary_id: str,
                        group: str = "red",
                        planner: str = "atomic",
                        source_id: str = "ed32b9c3-9593-4c33-b0db-e2007315096b") -> Optional[str]:
        """
        Create a new operation in Caldera.
        
        Args:
            name (str): Name of the operation
            adversary_id (str): ID of the adversary profile to use
            group (str): Agent group to target
            planner (str): Planner to use for the operation
            source_id (str): Source ID for facts
            
        Returns:
            Optional[str]: Operation ID if successful, None otherwise
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            
            operation_data = {
                "index": "operations",
                "name": name,
                "adversary": {"adversary_id": adversary_id},
                "group": group,
                "planner": {"id": planner},
                "source": {"id": source_id},
                "state": "paused",
                "autonomous": 1,
                "auto_close": False
            }
            
            response = self.session.put(url, headers=headers, json=operation_data)
            
            if response.status_code == 200:
                operation_id = response.json().get("id")
                self.logger.info(f"Created operation '{name}' with ID: {operation_id}")
                return operation_id
            else:
                self.logger.error(f"Failed to create operation: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating operation: {str(e)}")
            return None
    
    def start_operation(self, operation_id: str) -> bool:
        """
        Start a Caldera operation.
        
        Args:
            operation_id (str): ID of the operation to start
            
        Returns:
            bool: True if operation started successfully, False otherwise
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            
            data = {
                "index": "operations",
                "id": operation_id,
                "state": "running"
            }
            
            response = self.session.patch(url, headers=headers, json=data)
            
            if response.status_code == 200:
                self.logger.info(f"Started operation: {operation_id}")
                return True
            else:
                self.logger.error(f"Failed to start operation: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting operation: {str(e)}")
            return False
    
    def stop_operation(self, operation_id: str) -> bool:
        """
        Stop a running Caldera operation.
        
        Args:
            operation_id (str): ID of the operation to stop
            
        Returns:
            bool: True if operation stopped successfully, False otherwise
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            
            data = {
                "index": "operations",
                "id": operation_id,
                "state": "paused"
            }
            
            response = self.session.patch(url, headers=headers, json=data)
            
            if response.status_code == 200:
                self.logger.info(f"Stopped operation: {operation_id}")
                return True
            else:
                self.logger.error(f"Failed to stop operation: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping operation: {str(e)}")
            return False
    
    def get_operation_status(self, operation_id: str) -> Dict:
        """
        Get the status and details of an operation.
        
        Args:
            operation_id (str): ID of the operation
            
        Returns:
            Dict: Operation details and status
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            data = {
                "index": "operations",
                "id": operation_id
            }
            
            response = self.session.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                operation_data = response.json()
                return operation_data
            else:
                self.logger.error(f"Failed to get operation status: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting operation status: {str(e)}")
            return {}
    
    def get_operation_reports(self, operation_id: str) -> Dict:
        """
        Get detailed reports for an operation.
        
        Args:
            operation_id (str): ID of the operation
            
        Returns:
            Dict: Operation reports and results
        """
        try:
            url = f"{self.server_url}/api/v2/operations/{operation_id}/report"
            headers = {"KEY": "ADMIN123"}
            
            response = self.session.get(url, headers=headers)
            
            if response.status_code == 200:
                report_data = response.json()
                self.logger.info(f"Retrieved report for operation: {operation_id}")
                return report_data
            else:
                self.logger.error(f"Failed to get operation report: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting operation report: {str(e)}")
            return {}
    
    def execute_attack_scenario(self, 
                              scenario_name: str,
                              adversary_name: str = "Hunter",
                              target_group: str = "red",
                              timeout: int = 300) -> Dict:
        """
        Execute a complete attack scenario.
        
        Args:
            scenario_name (str): Name for the attack scenario
            adversary_name (str): Name of adversary profile to use
            target_group (str): Target agent group
            timeout (int): Timeout in seconds for the operation
            
        Returns:
            Dict: Results of the attack scenario execution
        """
        results = {
            "scenario_name": scenario_name,
            "start_time": datetime.now().isoformat(),
            "success": False,
            "operation_id": None,
            "report": {}
        }
        
        try:
            # Get available adversaries
            adversaries = self.get_adversaries()
            adversary_id = None
            
            for adv in adversaries:
                if adv.get("name") == adversary_name:
                    adversary_id = adv.get("adversary_id")
                    break
            
            if not adversary_id:
                self.logger.error(f"Adversary '{adversary_name}' not found")
                return results
            
            # Create operation
            operation_id = self.create_operation(
                name=scenario_name,
                adversary_id=adversary_id,
                group=target_group
            )
            
            if not operation_id:
                return results
            
            results["operation_id"] = operation_id
            
            # Start operation
            if not self.start_operation(operation_id):
                return results
            
            # Monitor operation
            start_time = time.time()
            while time.time() - start_time < timeout:
                status = self.get_operation_status(operation_id)
                
                if status.get("state") == "finished":
                    break
                elif status.get("state") == "out_of_time":
                    self.logger.info("Operation finished due to timeout")
                    break
                
                time.sleep(10)  # Check every 10 seconds
            
            # Get final report
            report = self.get_operation_reports(operation_id)
            results["report"] = report
            results["success"] = True
            results["end_time"] = datetime.now().isoformat()
            
            self.logger.info(f"Attack scenario '{scenario_name}' completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error executing attack scenario: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def cleanup_operations(self, operation_ids: List[str]) -> bool:
        """
        Clean up completed operations.
        
        Args:
            operation_ids (List[str]): List of operation IDs to clean up
            
        Returns:
            bool: True if cleanup successful
        """
        try:
            url = f"{self.server_url}/api/rest"
            headers = {"KEY": "ADMIN123"}
            
            for op_id in operation_ids:
                data = {
                    "index": "operations",
                    "id": op_id
                }
                response = self.session.delete(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    self.logger.info(f"Cleaned up operation: {op_id}")
                else:
                    self.logger.warning(f"Failed to cleanup operation {op_id}: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return False


# Example usage and utility functions
def main():
    """
    Example usage of the CalderaAttack class.
    """
    # Initialize Caldera attack instance
    caldera = CalderaAttack(
        server_url="http://localhost:8888",
        username="red",
        password="admin"
    )
    
    # Authenticate
    if not caldera.authenticate():
        print("Failed to authenticate with Caldera server")
        return
    
    # Get available resources
    agents = caldera.get_agents()
    adversaries = caldera.get_adversaries()
    abilities = caldera.get_abilities()
    
    print(f"Found {len(agents)} agents, {len(adversaries)} adversaries, {len(abilities)} abilities")
    
    # Execute an attack scenario
    results = caldera.execute_attack_scenario(
        scenario_name="RL_Training_Attack_Scenario",
        adversary_name="Hunter",
        target_group="red",
        timeout=600
    )
    
    print(f"Attack scenario results: {results}")


if __name__ == "__main__":
    main()
