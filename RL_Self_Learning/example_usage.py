#!/usr/bin/env python3
"""
Example usage of the Caldera Attack Module

This script demonstrates how to use the CalderaAttack class to perform
automated adversary emulation using the Caldera framework.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from caldera_attack_module import CalderaAttack


def run_basic_attack_scenario():
    """
    Run a basic attack scenario using Caldera.
    """
    print("Initializing Caldera Attack Module...")
    
    # Initialize the Caldera attack instance
    # Modify these parameters according to your Caldera server setup
    caldera = CalderaAttack(
        server_url="http://localhost:8888",  # Default Caldera server URL
        username="red",                       # Default username
        password="admin",                     # Default password
        verify_ssl=False                      # Disable SSL verification for local testing
    )
    
    print("Authenticating with Caldera server...")
    if not caldera.authenticate():
        print("❌ Failed to authenticate with Caldera server")
        print("Please ensure:")
        print("1. Caldera server is running on http://localhost:8888")
        print("2. Username and password are correct")
        print("3. Network connectivity is available")
        return False
    
    print("✅ Successfully authenticated with Caldera server")
    
    # Get available resources
    print("\nRetrieving available resources...")
    agents = caldera.get_agents()
    adversaries = caldera.get_adversaries()
    abilities = caldera.get_abilities()
    
    print(f"📊 Found {len(agents)} agents")
    print(f"🎭 Found {len(adversaries)} adversaries")
    print(f"⚡ Found {len(abilities)} abilities")
    
    if len(agents) == 0:
        print("⚠️  No agents found. Please deploy Caldera agents before running attacks.")
        return False
    
    if len(adversaries) == 0:
        print("⚠️  No adversaries found. Please ensure adversary profiles are loaded.")
        return False
    
    # List available adversaries
    print("\nAvailable adversaries:")
    for i, adv in enumerate(adversaries[:5]):  # Show first 5
        print(f"  {i+1}. {adv.get('name', 'Unknown')} - {adv.get('description', 'No description')}")
    
    # Execute attack scenario
    print("\n🚀 Starting attack scenario...")
    
    # Use the first available adversary or default to "Hunter"
    adversary_name = adversaries[0].get('name', 'Hunter') if adversaries else 'Hunter'
    
    results = caldera.execute_attack_scenario(
        scenario_name="RL_Training_Attack_Scenario",
        adversary_name=adversary_name,
        target_group="red",
        timeout=300  # 5 minutes timeout
    )
    
    print(f"\n📈 Attack scenario completed!")
    print(f"Operation ID: {results.get('operation_id', 'N/A')}")
    print(f"Success: {results.get('success', False)}")
    print(f"Start Time: {results.get('start_time', 'N/A')}")
    print(f"End Time: {results.get('end_time', 'N/A')}")
    
    if results.get('error'):
        print(f"❌ Error: {results['error']}")
    
    # Print summary of report if available
    report = results.get('report', {})
    if report:
        print(f"\n📊 Attack Report Summary:")
        print(f"Steps executed: {len(report.get('steps', []))}")
        # Add more report details as needed
    
    return results.get('success', False)


def interactive_mode():
    """
    Run in interactive mode to let users choose operations.
    """
    print("🎮 Interactive Caldera Attack Mode")
    print("=" * 40)
    
    caldera = CalderaAttack()
    
    if not caldera.authenticate():
        print("❌ Authentication failed. Exiting.")
        return
    
    while True:
        print("\nSelect an operation:")
        print("1. List agents")
        print("2. List adversaries")
        print("3. List abilities")
        print("4. Run attack scenario")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            agents = caldera.get_agents()
            print(f"\n📱 Agents ({len(agents)}):")
            for agent in agents:
                print(f"  - {agent.get('paw', 'Unknown')}: {agent.get('host', 'Unknown host')}")
        
        elif choice == "2":
            adversaries = caldera.get_adversaries()
            print(f"\n🎭 Adversaries ({len(adversaries)}):")
            for adv in adversaries:
                print(f"  - {adv.get('name', 'Unknown')}: {adv.get('description', 'No description')}")
        
        elif choice == "3":
            abilities = caldera.get_abilities()
            print(f"\n⚡ Abilities ({len(abilities)}):")
            for ability in abilities[:10]:  # Show first 10
                print(f"  - {ability.get('name', 'Unknown')}: {ability.get('description', 'No description')}")
            if len(abilities) > 10:
                print(f"  ... and {len(abilities) - 10} more")
        
        elif choice == "4":
            scenario_name = input("Enter scenario name: ").strip() or "Interactive_Scenario"
            adversary_name = input("Enter adversary name (or press Enter for default): ").strip() or "Hunter"
            
            print(f"\n🚀 Running scenario: {scenario_name}")
            results = caldera.execute_attack_scenario(
                scenario_name=scenario_name,
                adversary_name=adversary_name
            )
            
            if results.get('success'):
                print("✅ Scenario completed successfully!")
            else:
                print("❌ Scenario failed or encountered errors")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    print("🔐 Caldera Attack Module Example")
    print("=" * 40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print("Running basic attack scenario...")
        print("(Use --interactive flag for interactive mode)")
        success = run_basic_attack_scenario()
        
        if success:
            print("\n✅ Example completed successfully!")
        else:
            print("\n❌ Example encountered issues. Check the logs above.")
            sys.exit(1)
