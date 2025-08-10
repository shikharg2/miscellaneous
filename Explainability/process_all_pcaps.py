#!/usr/bin/env python3
"""
process_all_pcaps.py

Process all PCAP files in Benign and Malware folders and convert them to images
using pcap2img.py, saving them to respective folders in the images/ directory.

Usage:
    python3 process_all_pcaps.py
    python3 process_all_pcaps.py --nbytes 784 --layers all --format png
"""

import os
import glob
import argparse
import subprocess
import sys
from pathlib import Path

def get_pcap_files(folder_path):
    """Get all .pcap files from a folder."""
    pcap_files = []
    if os.path.exists(folder_path):
        pcap_files = glob.glob(os.path.join(folder_path, "*.pcap"))
    return sorted(pcap_files)

def process_pcap_folder(folder_name, nbytes=784, layers='all', img_format='png', verbose=True):
    """Process all PCAP files in a given folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pcap2img_path = os.path.join(script_dir, "pcap2img.py")
    
    # Check if pcap2img.py exists
    if not os.path.exists(pcap2img_path):
        print(f"Error: pcap2img.py not found at {pcap2img_path}")
        return False
    
    # Define paths
    input_folder = os.path.join(script_dir, folder_name)
    output_folder = os.path.join(script_dir, "images", folder_name)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {folder_name} folder")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"{'='*60}")
    
    # Get all PCAP files
    pcap_files = get_pcap_files(input_folder)
    
    if not pcap_files:
        print(f"No PCAP files found in {input_folder}")
        return True
    
    print(f"Found {len(pcap_files)} PCAP files to process:")
    for pcap_file in pcap_files:
        print(f"  - {os.path.basename(pcap_file)}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each PCAP file
    success_count = 0
    error_count = 0
    
    for pcap_file in pcap_files:
        try:
            if verbose:
                print(f"\nProcessing: {os.path.basename(pcap_file)}")
            
            # Build command
            cmd = [
                sys.executable,  # Use the same Python interpreter
                pcap2img_path,
                pcap_file,
                "--nbytes", str(nbytes),
                "--layers", layers,
                "--format", img_format,
                "--outdir", output_folder
            ]
            
            if not verbose:
                cmd.append("--quiet")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                success_count += 1
                if verbose:
                    print(f"  ✓ Successfully processed {os.path.basename(pcap_file)}")
            else:
                error_count += 1
                print(f"  ✗ Error processing {os.path.basename(pcap_file)}:")
                print(f"    {result.stderr}")
                
        except Exception as e:
            error_count += 1
            print(f"  ✗ Exception processing {os.path.basename(pcap_file)}: {e}")
    
    print(f"\n{folder_name} folder summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files: {len(pcap_files)}")
    
    return error_count == 0

def main():
    parser = argparse.ArgumentParser(description="Process all PCAP files in Benign and Malware folders.")
    parser.add_argument("--nbytes", type=int, default=784, help="Number of bytes to collect (default 784)")
    parser.add_argument("--layers", choices=('all','l7'), default='all', help="'all' raw bytes or 'l7' payload only")
    parser.add_argument("--format", choices=('png','jpg'), default='png', help="Image format")
    parser.add_argument("--quiet", action='store_true', help="Minimal output")
    parser.add_argument("--folder", choices=('Benign', 'Malware', 'both'), default='both', 
                       help="Which folder(s) to process (default: both)")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("PCAP to Image Batch Processor")
        print("=" * 40)
        print(f"Settings:")
        print(f"  Bytes per image: {args.nbytes}")
        print(f"  Layers mode: {args.layers}")
        print(f"  Image format: {args.format}")
        print(f"  Processing: {args.folder}")
    
    success = True
    
    # Process folders based on user selection
    if args.folder in ('Benign', 'both'):
        success &= process_pcap_folder('Benign', args.nbytes, args.layers, args.format, verbose)
    
    if args.folder in ('Malware', 'both'):
        success &= process_pcap_folder('Malware', args.nbytes, args.layers, args.format, verbose)
    
    if verbose:
        print(f"\n{'='*60}")
        if success:
            print("✓ All files processed successfully!")
        else:
            print("✗ Some files encountered errors during processing.")
        print(f"{'='*60}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
