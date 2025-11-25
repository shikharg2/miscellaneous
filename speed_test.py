#!/usr/bin/env python3
"""
Speed test utility using iperf3
Supports both public speedtest servers and private server connections
"""

import argparse
import sys
import iperf3

# List of publicly available iperf3 servers
PUBLIC_SERVERS = [
    'http://ping.online.net/',
    'http://ping6.online.net/',
   'iperf.scottlinux.com',
    'bouygues.iperf.fr',
    'speedtest.uztelecom.uz',
    'iperf.he.net',
    'ping.online.net'
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Speed test utility using iperf3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --public                    # Test with public server
  %(prog)s --private 192.168.1.100:5201  # Test with private server
  %(prog)s --public --duration 30      # Test for 30 seconds
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--public',
        action='store_true',
        help='Use publicly available speedtest servers'
    )
    group.add_argument(
        '--private',
        type=str,
        metavar='IP:PORT',
        help='Use private server (format: x.x.x.x:yyyy)'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        metavar='SECONDS',
        help='Duration of the test in seconds (default: 60)'
    )

    return parser.parse_args()


def parse_server_string(server_string):
    """Parse server string in format x.x.x.x:yyyy"""
    try:
        if ':' not in server_string:
            print(f"Error: Invalid format. Expected 'x.x.x.x:yyyy', got '{server_string}'")
            sys.exit(1)

        ip, port = server_string.rsplit(':', 1)
        port = int(port)

        if port < 1 or port > 65535:
            print(f"Error: Port must be between 1 and 65535, got {port}")
            sys.exit(1)

        return ip, port
    except ValueError as e:
        print(f"Error parsing server string: {e}")
        sys.exit(1)


def run_speed_test(server, port, duration):
    """Run iperf3 speed test"""
    print(f"\nConnecting to {server}:{port}")
    print(f"Test duration: {duration} seconds")
    print("-" * 60)

    try:
        client = iperf3.Client()
        client.server_hostname = server
        client.port = port
        client.duration = duration
        client.verbose = False

        print("\nRunning speed test...")
        result = client.run()

        if result.error:
            print(f"\nError: {result.error}")
            return False

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        # Download speed (receiving from server)
        print(f"\nDownload Speed:")
        print(f"  {result.received_Mbps:.2f} Mbps")
        print(f"  {result.received_MB_s:.2f} MB/s")

        # Upload speed (sending to server)
        print(f"\nUpload Speed:")
        print(f"  {result.sent_Mbps:.2f} Mbps")
        print(f"  {result.sent_MB_s:.2f} MB/s")

        # Additional statistics
        print(f"\nStatistics:")
        print(f"  Bytes sent: {result.sent_bytes:,} bytes")
        print(f"  Bytes received: {result.received_bytes:,} bytes")
        print(f"  Protocol: {result.protocol}")

        if hasattr(result, 'retransmits') and result.retransmits is not None:
            print(f"  Retransmits: {result.retransmits}")

        print("=" * 60)

        return True

    except Exception as e:
        print(f"\nError running speed test: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure iperf3 server is running and accessible")
        print("  - Check firewall settings")
        print("  - Verify the server address and port")
        return False


def test_public_server(duration):
    """Test with public servers"""
    print("\nTrying public iperf3 servers...")
    print(f"Available servers: {', '.join(PUBLIC_SERVERS)}")

    for server in PUBLIC_SERVERS:
        print(f"\n\nAttempting connection to: {server}")
        if run_speed_test(server, 5201, duration):
            return True
        print(f"Failed to connect to {server}, trying next...")

    print("\n\nError: Could not connect to any public servers")
    print("Note: Public servers may be down or busy. Try again later or use --private")
    return False


def main():
    """Main function"""
    args = parse_arguments()

    print("=" * 60)
    print("IPERF3 SPEED TEST")
    print("=" * 60)

    if args.public:
        success = test_public_server(args.duration)
    else:
        ip, port = parse_server_string(args.private)
        success = run_speed_test(ip, port, args.duration)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
