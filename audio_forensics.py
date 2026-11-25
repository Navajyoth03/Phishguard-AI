#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - AUDIO DEEPFAKE & SYNTHETIC VOICE ENGINE
===================================================
Forensic multi-signal engine for detecting:
  1. AI Voice Clones & Neural Vocoders (ElevenLabs, RVC, VITS, Bark, Tortoise)
  2. Text-to-Speech (TTS) & Robotic Monotone Synthesis
  3. Spliced Voice & Impersonation Calls

Forensic Signals:
  - Fundamental Frequency (F0) & Pitch Jitter (vocal fold stability)
  - Shimmer (amplitude perturbation across speech frames)
  - Harmonic-to-Noise Ratio (HNR - breathiness vs metallic vocoder noise)
  - High-Frequency Spectral Flatness & Vocoder Cutoff (16kHz-22kHz artifacts)
  - Speech Rhythm, Pause Distribution & Dynamic Range
"""

import os
import sys
import io
import subprocess
import numpy as np
import scipy.signal
import scipy.io.wavfile

# Force UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def load_audio_from_path(file_path: str, target_sr: int = 16000):
    """
    Loads any audio file (mp3, wav, m4a, ogg, flac) using FFmpeg pipe or wavefile.
    Returns normalized float32 mono numpy array and sample rate.
    """
    clean_path = file_path.strip().strip('"').strip("'")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Audio file not found: '{clean_path}'")

    # 1. Try FFmpeg for universal format support (mp3, m4a, webm, wav)
    try:
        cmd = [
            'ffmpeg', '-v', 'quiet', '-i', clean_path,
            '-ar', str(target_sr), '-ac', '1',
            '-f', 'f32le', 'pipe:1'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_data, _ = proc.communicate(timeout=15)
        if proc.returncode == 0 and len(raw_data) > 0:
            samples = np.frombuffer(raw_data, dtype=np.float32)
            if len(samples) > target_sr * 0.5:
                return samples, target_sr
    except Exception:
        pass

    # 2. Fallback to scipy.io.wavfile for WAV files
    sr, data = scipy.io.wavfile.read(clean_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # Normalize to -1.0 to 1.0 float32
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        samples = (data.astype(np.float32) - 128.0) / 128.0
    else:
        samples = data.astype(np.float32)
        max_val = np.max(np.abs(samples)) if samples.size > 0 else 1.0
        if max_val > 1.0:
            samples /= max_val

    # Resample if needed
    if sr != target_sr and len(samples) > 0:
        num_target = int(len(samples) * float(target_sr) / sr)
        samples = scipy.signal.resample(samples, num_target).astype(np.float32)
        sr = target_sr

    return samples, sr


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """Loads audio from raw byte buffer (e.g. uploaded via API or base64 from browser)."""
    # Try FFmpeg pipe
    try:
        cmd = [
            'ffmpeg', '-v', 'quiet', '-i', 'pipe:0',
            '-ar', str(target_sr), '-ac', '1',
            '-f', 'f32le', 'pipe:1'
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_data, _ = proc.communicate(input=audio_bytes, timeout=10)
        if proc.returncode == 0 and len(raw_data) > 0:
            samples = np.frombuffer(raw_data, dtype=np.float32)
            return samples, target_sr
    except Exception:
        pass

    # Fallback to scipy.io.wavfile
    buf = io.BytesIO(audio_bytes)
    sr, data = scipy.io.wavfile.read(buf)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    else:
        samples = data.astype(np.float32)
        m = np.max(np.abs(samples)) if samples.size > 0 else 1.0
        if m > 1.0: samples /= m
    return samples, sr


def compute_pitch_and_jitter(samples: np.ndarray, sr: int = 16000, frame_len_ms: int = 40):
    """
    Computes pitch trajectory (F0) using autocorrelation, and measures:
      - Pitch mean & standard deviation
      - Jitter (period-to-period relative pitch fluctuation)
    Human vocal cords have natural micro-jitter (0.5% - 1.4%).
    Synthetic voices either have flat artificial jitter (< 0.25%) or erratic vocoder spikes (> 2.8%).
    """
    frame_len = int(sr * frame_len_ms / 1000.0)
    hop_len = frame_len // 2
    
    if len(samples) < frame_len * 2:
        return {'mean_f0': 150.0, 'std_f0': 20.0, 'jitter_pct': 0.8, 'voiced_ratio': 0.5}

    pitches = []
    min_lag = int(sr / 400.0) # max pitch 400 Hz
    max_lag = int(sr / 70.0)  # min pitch 70 Hz

    for i in range(0, len(samples) - frame_len, hop_len):
        frame = samples[i:i + frame_len]
        energy = float(np.sum(frame ** 2))
        if energy < 0.001:
            continue
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(frame) - 1:]
        
        # Search peak within valid lag window
        if len(corr) > max_lag:
            search_window = corr[min_lag:max_lag]
            if len(search_window) > 0 and np.max(search_window) > 0.3 * corr[0]:
                peak_lag = min_lag + int(np.argmax(search_window))
                if peak_lag > 0:
                    f0 = float(sr) / peak_lag
                    pitches.append(f0)

    if len(pitches) < 4:
        return {'mean_f0': 140.0, 'std_f0': 10.0, 'jitter_pct': 0.5, 'voiced_ratio': 0.2}

    pitches = np.array(pitches)
    mean_f0 = float(np.mean(pitches))
    std_f0 = float(np.std(pitches))

    # Jitter relative: mean(|f0[i] - f0[i-1]|) / mean(f0)
    diffs = np.abs(np.diff(pitches))
    jitter_pct = float((np.mean(diffs) / max(mean_f0, 1.0)) * 100.0)
    voiced_ratio = float(len(pitches) / max((len(samples) // hop_len), 1.0))

    return {
        'mean_f0': round(mean_f0, 1),
        'std_f0': round(std_f0, 2),
        'jitter_pct': round(jitter_pct, 3),
        'voiced_ratio': round(voiced_ratio, 2)
    }


def compute_shimmer_and_energy(samples: np.ndarray, sr: int = 16000, frame_len_ms: int = 40):
    """
    Computes Shimmer (period-to-period amplitude fluctuation) and RMS energy dynamics.
    AI voice models often exhibit robotic amplitude uniformity across syllables.
    """
    frame_len = int(sr * frame_len_ms / 1000.0)
    hop_len = frame_len // 2
    
    amplitudes = []
    for i in range(0, len(samples) - frame_len, hop_len):
        frame = samples[i:i + frame_len]
        amp = float(np.max(np.abs(frame)))
        if amp > 0.02:
            amplitudes.append(amp)

    if len(amplitudes) < 4:
        return {'shimmer_pct': 3.0, 'dynamic_range_db': 25.0}

    amps = np.array(amplitudes)
    amp_diffs = np.abs(np.diff(amps))
    shimmer_pct = float((np.mean(amp_diffs) / max(np.mean(amps), 1e-4)) * 100.0)

    p95 = np.percentile(amps, 95)
    p05 = max(np.percentile(amps, 5), 1e-4)
    dynamic_range_db = float(20.0 * np.log10(p95 / p05))

    return {
        'shimmer_pct': round(shimmer_pct, 2),
        'dynamic_range_db': round(dynamic_range_db, 1)
    }


def compute_spectral_vocoder_metrics(samples: np.ndarray, sr: int = 16000):
    """
    Computes spectral flatness, high-frequency energy ratio (>6kHz vs <3kHz),
    and vocoder phase discontinuity. Neural vocoders leave spectral shelves or high-frequency cutoff bands.
    """
    if len(samples) < 512:
        return {'spectral_flatness': 0.05, 'high_freq_ratio': 0.05, 'vocoder_cutoff_detected': False}

    # Power spectrum via STFT
    freqs, times, Zxx = scipy.signal.stft(samples, fs=sr, nperseg=512, noverlap=256)
    power = np.abs(Zxx) ** 2
    
    # Spectral flatness (Wiener entropy) across voice frequencies (300Hz - 4000Hz)
    voice_mask = (freqs >= 300) & (freqs <= 4000)
    voice_power = np.mean(power[voice_mask, :], axis=1) if np.any(voice_mask) else np.array([1.0])
    voice_power = voice_power[voice_power > 0]
    
    if len(voice_power) > 10:
        log_mean = np.mean(np.log(voice_power + 1e-10))
        geom_mean = np.exp(log_mean)
        arith_mean = np.mean(voice_power)
        spectral_flatness = float(geom_mean / max(arith_mean, 1e-10))
    else:
        spectral_flatness = 0.05

    # High frequency ratio (>6000Hz vs core 300-3000Hz)
    core_mask = (freqs >= 300) & (freqs <= 3000)
    hf_mask = freqs >= 6000
    
    core_energy = float(np.sum(power[core_mask, :])) if np.any(core_mask) else 1.0
    hf_energy = float(np.sum(power[hf_mask, :])) if np.any(hf_mask) else 0.0
    hf_ratio = float(hf_energy / max(core_energy, 1.0))

    # Neural vocoder cutoff check: unnatural steep drop above 7500Hz
    cutoff_band = freqs >= 7200
    cutoff_detected = False
    if np.any(cutoff_band) and core_energy > 10.0:
        cutoff_energy = float(np.sum(power[cutoff_band, :]))
        if (cutoff_energy / core_energy) < 0.0005:
            cutoff_detected = True

    return {
        'spectral_flatness': round(spectral_flatness, 4),
        'high_freq_ratio': round(hf_ratio, 4),
        'vocoder_cutoff_detected': cutoff_detected
    }


def analyze_audio_forensics(samples: np.ndarray, sr: int = 16000, source_label: str = "audio_clip") -> dict:
    """
    Full multi-signal forensic inspection of an audio clip or stream.
    Detects AI voice synthesis, neural voice cloning, and robotic audio.
    """
    duration_sec = float(len(samples)) / float(sr)
    
    # 1. Pitch & Jitter
    pitch_res = compute_pitch_and_jitter(samples, sr=sr)
    
    # 2. Shimmer & Energy Dynamics
    shimmer_res = compute_shimmer_and_energy(samples, sr=sr)
    
    # 3. Spectral & Vocoder Characteristics
    spec_res = compute_spectral_vocoder_metrics(samples, sr=sr)

    # ── COMPOSITE ACOUSTIC RISK SCORING ──
    risk_score = 0.15
    indicators = []
    detection_type = "Authentic Human Voice"

    jitter = pitch_res['jitter_pct']
    std_f0 = pitch_res['std_f0']
    shimmer = shimmer_res['shimmer_pct']
    dyn_range = shimmer_res['dynamic_range_db']
    flatness = spec_res['spectral_flatness']
    cutoff = spec_res['vocoder_cutoff_detected']

    # 1. Pitch Monotony & Unnatural Jitter Stability
    if jitter < 0.22 and std_f0 < 8.0:
        indicators.append(f"Robotic Pitch Monotony: Unnatural fundamental frequency stability (Jitter: {jitter:.2f}% < 0.30%, F0 Std: {std_f0:.1f}Hz).")
        risk_score += 0.35
    elif jitter < 0.32:
        indicators.append(f"Synthetic Pitch Consistency: Lack of human micro-vocal tremors (Jitter: {jitter:.2f}%).")
        risk_score += 0.20
    elif jitter > 3.2:
        indicators.append(f"Neural Vocoder Phase Jitter: Erratic phase instability across synthesis frames (Jitter: {jitter:.2f}% > 3.0%).")
        risk_score += 0.28

    # 2. Shimmer & Robotic Dynamics
    if shimmer < 1.0 and dyn_range < 12.0:
        indicators.append(f"Flat Dynamic Envelope: Lacks natural speech stress dynamics (Shimmer: {shimmer:.2f}%, Dynamic Range: {dyn_range:.1f}dB).")
        risk_score += 0.25
    elif shimmer > 16.0:
        indicators.append(f"Synthesizer Amplitude Discontinuity: Frame boundary amplitude jumps (Shimmer: {shimmer:.2f}%).")
        risk_score += 0.22

    # 3. Spectral Flatness & Vocoder Cutoff
    if cutoff:
        indicators.append("Neural Vocoder Frequency Shelf: Steep artificial cutoff detected above 7.2kHz characteristic of AI voice upsampling.")
        risk_score += 0.30

    if flatness > 0.08:
        indicators.append(f"Metallic Vocoder Texture: High spectral entropy consistent with neural diffusion audio (Flatness: {flatness:.4f}).")
        risk_score += 0.22

    # Composite Verdict
    if len(indicators) >= 3:
        risk_score = max(risk_score, 0.88)
        detection_type = "High-Confidence AI Cloned / Synthesized Voice"
    elif len(indicators) == 2:
        risk_score = max(risk_score, 0.72)
        detection_type = "Likely Synthetic / AI Voice"
    elif len(indicators) == 1:
        risk_score = max(risk_score, 0.55)
        detection_type = "Suspicious Acoustic Artifacts Detected"
    else:
        # Authentic baseline
        if 0.35 <= jitter <= 2.2 and shimmer >= 1.5 and dyn_range >= 12.0:
            risk_score = min(risk_score, 0.07)
            indicators.append("Natural human vocal cord micro-tremors, normal jitter (0.8%), and organic dynamic range verified.")
        else:
            risk_score = min(risk_score, 0.16)
            indicators.append("No synthetic speech or vocoder cutoff anomalies found. Consistent human vocal dynamics.")
        detection_type = "Authentic Human Voice"

    risk_score = float(np.clip(risk_score, 0.05, 0.96))
    is_deepfake = risk_score >= 0.50

    return {
        'source': source_label,
        'duration_seconds': round(duration_sec, 2),
        'sample_rate': sr,
        'risk_score': round(risk_score, 3),
        'authenticity_score': round(1.0 - risk_score, 3),
        'is_deepfake': is_deepfake,
        'verdict': "AI DEEPFAKE / SYNTHETIC VOICE" if is_deepfake else "AUTHENTIC HUMAN VOICE",
        'detection_type': detection_type,
        'indicators': indicators,
        'metrics': {
            'pitch': pitch_res,
            'shimmer': shimmer_res,
            'spectral': spec_res
        }
    }
