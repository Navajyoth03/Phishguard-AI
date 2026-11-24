#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD AI — TERMINAL MULTI-MODAL TESTER
===========================================
Test all 5 modalities (URL, Text, Visual, Audio, Video) and Cross-Attention Fusion
directly in your terminal without requiring the browser extension.

Usage:
    python test_cli.py                      # Interactive menu
    python test_cli.py --preset paypal      # Test credential harvesting phishing
    python test_cli.py --preset deepfake    # Test deepfake video phishing
    python test_cli.py --preset vishing     # Test audio / vishing scam
    python test_cli.py --preset safe        # Test benign legitimate website
    python test_cli.py --url "http://..." --text "..."
"""

import sys
import os
import argparse
import base64
import io
import time

# Force UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
from PIL import Image

# Import PhishGuard core
import server
import test_video

def print_banner():
    print("=" * 72)
    print("  🛡️  PHISHGUARD AI — MULTI-MODAL TERMINAL TEST HARNESS")
    print("  URL (RF) • Text (TF-IDF) • Visual (DOM) • Audio • Video (CV) • Fusion")
    print("=" * 72)

def format_meter(score, length=32):
    filled = int(round(score * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)

def run_test(name, url, text="", dom_features=None, video_path=None):
    dom_features = dom_features or {}
    print(f"\n[RUNNING TEST] {name}")
    print(f"  Target URL : {url}")
    if text:
        preview = text[:80] + ("..." if len(text) > 80 else "")
        print(f"  Text Body  : \"{preview}\"")

    video_frames = []
    if video_path and os.path.exists(video_path):
        print(f"  Video File : {video_path} (extracting frames...)")
        raw_frames, fps, dur = test_video.extract_frames(video_path, num_samples=8)
        for f in raw_frames:
            buf = io.BytesIO()
            Image.fromarray(np.uint8(f)).save(buf, format='JPEG')
            video_frames.append(base64.b64encode(buf.getvalue()).decode('ascii'))
        dom_features['has_video'] = True
        print(f"  Captured   : {len(video_frames)} frames across {dur:.1f}s")

    t0 = time.time()

    # Run individual modalities
    u_res = server.analyze_url(url)
    t_res = server.analyze_text(text, url)
    img_res = server.analyze_image(url, dom_features)
    aud_res = server.analyze_audio(url, text, dom_features)
    vid_res = server.analyze_video(url, text, dom_features, video_frames)

    # Fusion
    active_modalities = {'url': u_res}
    if text and len(text.strip()) >= 5:
        active_modalities['text'] = t_res
    if dom_features.get('has_video') or vid_res.get('risk_score', 0) > 0.35:
        active_modalities['video'] = vid_res
    if dom_features.get('has_audio') or aud_res.get('risk_score', 0) > 0.35:
        active_modalities['audio'] = aud_res
    if dom_features.get('num_images', 0) > 0 or img_res.get('risk_score', 0) > 0.20:
        active_modalities['image'] = img_res

    fusion = server.fusion_engine.fuse(active_modalities)
    latency = int((time.time() - t0) * 1000)

    score = fusion['final_risk_score']
    level = fusion['risk_level']
    meter = format_meter(score)

    print("\n" + "─" * 72)
    print(f"  FINAL VERDICT : {level}")
    print(f"  FUSION RISK   : {score:.1%} [{meter}]")
    print(f"  ACTION        : {fusion.get('action_required', 'Proceed with caution')}")
    print(f"  LATENCY       : {latency} ms")
    print("─" * 72)

    print("\n📊 MODALITY BREAKDOWN:")
    risks = fusion.get('individual_risks', {})
    for mod in ['url', 'text', 'image', 'audio', 'video']:
        if mod in risks:
            r = risks[mod]
            m_meter = format_meter(r, length=20)
            tag = "CRITICAL" if r >= 0.70 else ("ELEVATED" if r >= 0.50 else "BENIGN")
            print(f"  • {mod.upper():<6} [{r:.1%}] [{m_meter}]  ({tag})")

    # Forensic Red Flags
    indicators = []
    if u_res.get('indicators'): indicators.extend([f"URL: {i}" for i in u_res['indicators'][:2]])
    if t_res.get('indicators'): indicators.extend([f"Text: {i}" for i in t_res['indicators'][:2]])
    if img_res.get('indicators'): indicators.extend([f"Visual: {i}" for i in img_res['indicators'][:2]])
    if aud_res.get('indicators'): indicators.extend([f"Audio: {i}" for i in aud_res['indicators'][:2]])
    if vid_res.get('indicators'): indicators.extend([f"Video: {i}" for i in vid_res['indicators'][:3]])

    if indicators:
        print("\n🚩 FORENSIC EVIDENCE & ANOMALIES:")
        for ind in indicators:
            print(f"  └─► {ind}")

    if fusion.get('agreement_multiplier'):
        mult = fusion['agreement_multiplier']
        sign = "↑ Agreement Boost" if mult >= 1.0 else "↓ Conflict Dampening"
        print(f"\n🤝 CROSS-ATTENTION COHERENCE: {mult:.4f}x ({sign})")

    print("=" * 72)

PRESETS = {
    'paypal': {
        'name': '1. Traditional Banking Phishing (Credential Harvesting)',
        'url': 'http://paypal-security-update-99.tk/login?verify=account',
        'text': 'URGENT: Your PayPal account has been restricted due to suspicious activity. Enter your password and OTP immediately to avoid permanent account termination.',
        'dom_features': {
            'has_login_form': True,
            'has_password': True,
            'num_images': 6,
            'ext_img_ratio': 0.75,
            'has_logo_match': True
        }
    },
    'deepfake': {
        'name': '2. Deepfake Video Phishing (CEO Restructuring Scam)',
        'url': 'http://urgent-corporate-update-login-75.xyz/stream',
        'text': 'URGENT CEO Announcement: Please watch this mandatory video update regarding company restructuring and verify your employee credentials with your OTP.',
        'video_path': 'demo_pages/id0_id9_0008_h264.mp4',
        'dom_features': {
            'has_video': True,
            'has_login_form': True,
            'has_password': True
        }
    },
    'vishing': {
        'name': '3. Audio Vishing Scam (Tech Support Scam)',
        'url': 'http://microsoft-windows-security-alert-call.tk/warning',
        'text': 'CRITICAL ALERT: Your computer is infected with spyware. Call 1-800-HELP-DESK immediately. Do not restart your computer or your data will be permanently lost.',
        'dom_features': {
            'has_audio': True,
            'has_alarm_audio': True,
            'has_autoplay_audio': True,
            'has_tts_audio': True
        }
    },
    'safe': {
        'name': '4. Legitimate Verified Website (Google Search / ICICI Bank)',
        'url': 'https://www.google.com/search?q=weather+today',
        'text': 'Google Search. About 1,420,000,000 results. Today weather forecast: Partly cloudy.',
        'dom_features': {
            'has_login_form': False,
            'has_password': False,
            'num_images': 2,
            'ext_img_ratio': 0.0
        }
    }
}

def interactive_menu():
    print_banner()
    print("\nSelect a test to run:")
    print("  [1] Traditional Phishing (PayPal Credential Harvesting)")
    print("  [2] Deepfake Video Phishing (id0_id9_0008_h264.mp4)")
    print("  [3] Audio Vishing Scam (Siren + Emergency Tech-Support Call)")
    print("  [4] Benign Legitimate Site (Google.com)")
    print("  [5] Custom URL & Text Input")
    print("  [0] Exit")

    choice = input("\nEnter choice [1-5]: ").strip()

    if choice == '1':
        p = PRESETS['paypal']
        run_test(p['name'], p['url'], p['text'], p['dom_features'])
    elif choice == '2':
        p = PRESETS['deepfake']
        run_test(p['name'], p['url'], p['text'], p['dom_features'], p['video_path'])
    elif choice == '3':
        p = PRESETS['vishing']
        run_test(p['name'], p['url'], p['text'], p['dom_features'])
    elif choice == '4':
        p = PRESETS['safe']
        run_test(p['name'], p['url'], p['text'], p['dom_features'])
    elif choice == '5':
        url = input("Enter target URL: ").strip()
        text = input("Enter page text (optional, press Enter to skip): ").strip()
        run_test("Custom Analysis", url, text)
    else:
        print("Exiting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PhishGuard Terminal Test Harness")
    parser.add_argument('--preset', choices=['paypal', 'deepfake', 'vishing', 'safe'], help="Run a specific preset test")
    parser.add_argument('--url', type=str, help="Custom URL to analyze")
    parser.add_argument('--text', type=str, default="", help="Custom text content")
    parser.add_argument('--video', type=str, default=None, help="Path to video file")

    args = parser.parse_args()

    if args.preset:
        print_banner()
        p = PRESETS[args.preset]
        run_test(p['name'], p['url'], p.get('text', ''), p.get('dom_features', {}), p.get('video_path'))
    elif args.url:
        print_banner()
        run_test("Custom Terminal Test", args.url, args.text, video_path=args.video)
    else:
        interactive_menu()
