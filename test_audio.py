#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - FORENSIC AUDIO DEEPFAKE INSPECTOR (CLI)
===================================================
Forensic acoustic analysis tool that determines if an audio recording
is Authentic Human Speech or an AI Voice Clone / Synthetic Deepfake,
and outputs the exact acoustic evidence explaining WHY.

Usage:
    python test_audio.py "path/to/audio.mp3"
    python test_audio.py "path/to/voice_call.wav"
    python test_audio.py                       (interactive prompt)
"""

import os
import sys
import argparse
import audio_forensics

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
    print("    PHISHGUARD - STANDALONE AUDIO DEEPFAKE INSPECTOR v3.0")
    print("    Acoustic Jitter, Vocoder Cutoff & Neural Voice Synthesis Detection")
    print("=" * 68 + f"{RESET}\n")


def inspect_audio(file_path: str):
    clean_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(clean_path):
        print(f"{RED}[ERROR] File not found: '{clean_path}'{RESET}")
        return

    print(f"{DIM}Loading & decoding audio: {clean_path}...{RESET}")
    try:
        samples, sr = audio_forensics.load_audio_from_path(clean_path)
    except Exception as e:
        print(f"{RED}[ERROR] Failed to load audio: {e}{RESET}")
        return

    filename = os.path.basename(clean_path)
    report = audio_forensics.analyze_audio_forensics(samples, sr=sr, source_label=filename)

    risk = report['risk_score']
    is_deepfake = report['is_deepfake']
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
    print(f" {BOLD}ACOUSTIC FORENSIC REPORT: {filename}{RESET}")
    print("=" * 68)
    print(f" Duration:          {report['duration_seconds']}s (Sample Rate: {report['sample_rate']} Hz)")
    print(f" Deepfake Risk:     {verdict_color}{BOLD}{risk * 100:.1f}%{RESET}")
    print(f" Human Authenticity:{100.0 - (risk * 100.0):.1f}%")
    print(f" Verdict:           {verdict_color}{BOLD}{verdict}{RESET}")
    print(f" Classification:    {detection_type}")
    print("-" * 68)

    print(f"{BOLD}DETAILED ACOUSTIC METRICS:{RESET}")
    pitch = report['metrics']['pitch']
    shimmer = report['metrics']['shimmer']
    spec = report['metrics']['spectral']

    print(f"  • Fundamental Pitch (F0):   {pitch['mean_f0']} Hz (Std: {pitch['std_f0']} Hz)")
    print(f"  • Vocal Cord Jitter:        {pitch['jitter_pct']}%  (Normal human: 0.4% - 1.4% | Clones: <0.25% or >2.8%)")
    print(f"  • Shimmer (Amplitude Pert): {shimmer['shimmer_pct']}%  (Normal human: 2.0% - 6.0%)")
    print(f"  • Dynamic Range:            {shimmer['dynamic_range_db']} dB")
    print(f"  • Spectral Flatness:        {spec['spectral_flatness']}  (Higher in metallic vocoders)")
    print(f"  • High-Frequency Ratio:     {spec['high_freq_ratio']}")
    print(f"  • Vocoder Cutoff Shelf:     {'DETECTED (Above 7.2kHz)' if spec['vocoder_cutoff_detected'] else 'None (Natural decay)'}")

    print("-" * 68)
    print(f"{BOLD}FORENSIC EVIDENCE & ANOMALIES:{RESET}")
    for idx, ind in enumerate(report['indicators'], 1):
        if is_deepfake:
            print(f"  {RED}[!] Anomaly #{idx}:{RESET} {ind}")
        else:
            print(f"  {GREEN}[✓] Check #{idx}:{RESET} {ind}")

    print("=" * 68 + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="PhishGuard Forensic Audio Deepfake Inspector")
    parser.add_argument("audio", nargs="?", help="Path to audio file (MP3, WAV, M4A, OGG)")
    args = parser.parse_args()

    print_banner()

    audio_path = args.audio
    if not audio_path:
        print(f"{BOLD}Enter audio file path to inspect (or drag & drop here):{RESET}")
        try:
            audio_path = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not audio_path:
        print("No audio file provided. Exiting.")
        sys.exit(0)

    inspect_audio(audio_path)


if __name__ == '__main__':
    main()
