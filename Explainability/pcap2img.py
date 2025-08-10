#!/usr/bin/env python3
"""
pcap2img_single.py

Produce ONE image per pcap by taking the FIRST N bytes from the entire pcap (time-ordered).
Default N = 784 (28x28). Short captures are zero-padded.

Dependencies:
    pip install scapy pillow

Usage:
    python3 pcap2img_single.py example.pcap
    python3 pcap2img_single.py example.pcap --nbytes 784 --layers all --format png --outdir imgs
"""

import os
import argparse
import math
from scapy.all import PcapReader, raw, TCP, UDP, IP, IPv6
from PIL import Image

DEFAULT_NBYTES = 784  # 28*28

def packet_bytes(pkt, layers_mode='all'):
    """Return relevant bytes for a packet according to layers_mode."""
    if layers_mode.lower() == 'all':
        try:
            return raw(pkt)
        except Exception:
            try:
                return bytes(pkt)
            except Exception:
                return b''
    elif layers_mode.lower() == 'l7':
        if TCP in pkt:
            return bytes(pkt[TCP].payload)
        if UDP in pkt:
            return bytes(pkt[UDP].payload)
        return b''
    else:
        raise ValueError("layers_mode must be 'all' or 'l7'")

def make_square_and_image(buf: bytes):
    """Pad buffer to a perfect square length and return a PIL grayscale Image."""
    L = len(buf)
    side = int(math.ceil(math.sqrt(L)))
    target = side * side
    if L < target:
        buf = buf + (b'\x00' * (target - L))
    img = Image.new('L', (side, side))
    img.putdata(list(buf))
    return img, side

def pcap_first_n_to_image(infile, nbytes=DEFAULT_NBYTES, layers='all', img_format='png', outdir='images', verbose=True):
    """Read pcap streaming and collect the first nbytes of concatenated packet bytes (time order)."""
    os.makedirs(outdir, exist_ok=True)
    collected = bytearray()
    read_pkts = 0

    if verbose:
        print(f"[+] Reading {infile} to collect first {nbytes} bytes (layers={layers}) ...")

    with PcapReader(infile) as reader:
        for pkt in reader:
            read_pkts += 1
            part = packet_bytes(pkt, layers_mode=layers)
            if part:
                need = nbytes - len(collected)
                if need <= 0:
                    break
                # append only what's needed from this packet
                if len(part) <= need:
                    collected.extend(part)
                else:
                    collected.extend(part[:need])
                    break

    # pad if short
    if len(collected) < nbytes:
        collected.extend(b'\x00' * (nbytes - len(collected)))

    if verbose:
        print(f"[+] Packets read: {read_pkts}. Collected bytes: {len(collected)}")

    # create square image (should be 28x28 when nbytes=784)
    img, side = make_square_and_image(bytes(collected))

    # filename based on pcap basename
    base = os.path.basename(infile)
    name = os.path.splitext(base)[0]
    fname = f"{name}_first{nbytes}_s{side}.{img_format}"
    outpath = os.path.join(outdir, fname)
    img.save(outpath)
    if verbose:
        print(f"[+] Saved image: {outpath} ({side}x{side})")
    return outpath

def main():
    parser = argparse.ArgumentParser(description="Convert a pcap to a single image using first N bytes.")
    parser.add_argument("pcap", help="Input pcap file")
    parser.add_argument("--nbytes", type=int, default=DEFAULT_NBYTES, help="Number of bytes to collect (default 784)")
    parser.add_argument("--layers", choices=('all','l7'), default='all', help="'all' raw bytes or 'l7' payload only")
    parser.add_argument("--format", choices=('png','jpg'), default='png', help="Image format")
    parser.add_argument("--outdir", default='images', help="Output directory")
    parser.add_argument("--quiet", action='store_true', help="Minimal output")
    args = parser.parse_args()

    pcap_first_n_to_image(
        infile=args.pcap,
        nbytes=args.nbytes,
        layers=args.layers,
        img_format=args.format,
        outdir=args.outdir,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()
