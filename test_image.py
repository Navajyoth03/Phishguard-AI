#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - FORENSIC IMAGE INSPECTOR (CLI)
===========================================
Forensic image analysis engine that determines if an image is Authentic or
AI-Generated / Tampered (Background replacement, Inpainting, Splicing),
and outputs the exact forensic evidence explaining WHY.

Usage:
    python test_image.py "path/to/image.png"
    python test_image.py "path/to/image.jpg" --save-ela
    python test_image.py                      (interactive prompt)
"""

import os
import sys
import argparse
from PIL import Image
import image_forensics

# Force UTF-8 in Windows terminal
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Terminal colors
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"


def print_banner():
    print(f"{CYAN}{BOLD}")
    print("=" * 68)
    print("    PHISHGUARD - STANDALONE IMAGE FORENSICS INSPECTOR v3.0")
    print("    Multi-Signal Tampering, Inpainting & AI Diffusion Detection")
    print("=" * 68 + f"{RESET}\n")


def inspect_image(image_path: str, save_ela: bool = False):
    clean_path = image_path.strip().strip('"').strip("'")
    if not os.path.exists(clean_path):
        print(f"{RED}[ERROR] File not found: '{clean_path}'{RESET}")
        return

    print(f"{DIM}Loading image: {clean_path}...{RESET}")
    try:
        img = Image.open(clean_path)
    except Exception as e:
        print(f"{RED}[ERROR] Failed to open image: {e}{RESET}")
        return

    filename = os.path.basename(clean_path)
    report = image_forensics.analyze_image_forensics(img, filename=filename)

    risk = report['risk_score']
    is_tampered = report['is_tampered']
    verdict = report['verdict']
    detection_type = report['detection_type']

    # Colorize verdict
    if risk >= 0.70:
        verdict_color = RED
    elif risk >= 0.40:
        verdict_color = YELLOW
    else:
        verdict_color = GREEN

    print("\n" + "=" * 68)
    print(f" {BOLD}IMAGE FORENSIC REPORT: {filename}{RESET}")
    print("=" * 68)
    print(f" Dimensions:       {report['dimensions']}")
    print(f" Format:           {report['format']}")
    print(f" Tampering Risk:   {verdict_color}{BOLD}{risk * 100:.1f}%{RESET}")
    print(f" Authenticity:     {100.0 - (risk * 100.0):.1f}%")
    print(f" Verdict:          {verdict_color}{BOLD}{verdict}{RESET}")
    print(f" Classification:   {detection_type}")
    print("-" * 68)

    print(f"{BOLD}DETAILED FORENSIC METRICS:{RESET}")
    ela = report['metrics']['ela']
    fft = report['metrics']['fft']
    lap = report['metrics']['laplacian']
    noise = report['metrics']['noise']

    print(f"  • ELA Mean Error:              {ela['mean_error']} (Std: {ela['std_error']})")
    print(f"  • ELA Tile Disparity:          {ela['tile_disparity_ratio']}x  (Threshold: >2.1x indicates local editing)")
    print(f"  • Subject vs BG Disparity:     {ela['subject_bg_disparity']}x")
    print(f"  • 2D-FFT Azimuthal Variance:   {fft['azimuthal_variance']}   (Threshold: >0.35 indicates generative lattice)")
    print(f"  • 2D-FFT High Energy Ratio:    {fft['high_energy_ratio']}")
    print(f"  • Laplacian Edge Disparity:    {lap['edge_disparity']}x  (Threshold: >2.4x indicates feathering/splicing)")
    print(f"  • Sensor PRNU Noise Disparity: {noise['noise_disparity']}x  (Threshold: >2.5x indicates composite photo)")

    print("-" * 68)
    print(f"{BOLD}FORENSIC EVIDENCE & ANOMALIES:{RESET}")
    for idx, ind in enumerate(report['indicators'], 1):
        if is_tampered:
            print(f"  {RED}[!] Anomaly #{idx}:{RESET} {ind}")
        else:
            print(f"  {GREEN}[✓] Check #{idx}:{RESET} {ind}")

    if save_ela:
        out_name = f"ela_{os.path.splitext(filename)[0]}.png"
        _, _, _, heatmap_img = image_forensics.compute_ela(img)
        heatmap_img.save(out_name)
        print(f"\n{CYAN}[+] ELA Amplified Heatmap saved to: '{out_name}'{RESET}")

    print("=" * 68 + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="PhishGuard Forensic Image Manipulation Inspector")
    parser.add_argument("image", nargs="?", help="Path to image file (PNG, JPG, WEBP)")
    parser.add_argument("--save-ela", action="store_true", help="Save the visual ELA difference heatmap to disk")
    args = parser.parse_args()

    print_banner()

    image_path = args.image
    if not image_path:
        print(f"{BOLD}Enter image path to inspect (or drag & drop here):{RESET}")
        try:
            image_path = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not image_path:
        print("No image provided. Exiting.")
        sys.exit(0)

    inspect_image(image_path, save_ela=args.save_ela)


if __name__ == '__main__':
    main()
