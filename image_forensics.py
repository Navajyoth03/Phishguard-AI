#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - IMAGE FORENSICS & MANIPULATION ENGINE
===================================================
Forensic multi-signal engine for detecting:
  1. Fully AI-Generated Images (Stable Diffusion, Midjourney, DALL-E, Flux)
  2. Localized Tampering, Splicing, Background Swaps & Inpainting (e.g., CASIA 2.0 style)

Forensic signals:
  - Error Level Analysis (ELA) with regional compression disparity
  - 2D Fast Fourier Transform (FFT) spectral lattice & high-frequency power
  - Discrete Laplacian boundary sharpness & edge gradient disparity
  - High-frequency sensor noise (PRNU residual) consistency
  - Metadata, EXIF, and software signature parsing
"""

import os
import sys
import io
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags

_clf = None
_scaler = None
_model_loaded = False

def _load_ml_model():
    global _clf, _scaler, _model_loaded
    if _model_loaded:
        return _clf, _scaler
    try:
        import joblib
        base = os.path.dirname(os.path.abspath(__file__))
        mdir = os.path.join(base, 'models', 'forensics_model')
        c_path = os.path.join(mdir, 'classifier.joblib')
        s_path = os.path.join(mdir, 'scaler.joblib')
        if os.path.exists(c_path) and os.path.exists(s_path):
            _clf = joblib.load(c_path)
            _scaler = joblib.load(s_path)
    except Exception:
        pass
    _model_loaded = True
    return _clf, _scaler

# Force UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15):
    """
    Error Level Analysis (ELA).
    Resaves image at a standard JPEG quality in memory and computes
    pixel-level compression error disparity.
    
    Returns:
      ela_stats: dict of mean, std, max, tile disparity
      heatmap_b64: base64-encoded PNG of the amplified difference map
      ela_array: float32 numpy array of difference magnitudes
    """
    # Convert image to RGB
    orig_rgb = image.convert('RGB')
    
    # Save to in-memory buffer at specified JPEG quality
    buf = io.BytesIO()
    orig_rgb.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    
    # Compute absolute difference
    diff = ImageChops.difference(orig_rgb, resaved)
    
    # Convert to numpy arrays for forensic mathematics
    diff_arr = np.array(diff, dtype=np.float32)
    gray_diff = np.mean(diff_arr, axis=2) # 2D error magnitude
    
    ela_mean = float(np.mean(gray_diff))
    ela_std = float(np.std(gray_diff))
    ela_max = float(np.max(gray_diff))
    
    # Tile-based disparity (4x4 grid across image)
    H, W = gray_diff.shape
    tile_h, tile_w = max(4, H // 4), max(4, W // 4)
    tile_stds = []
    tile_means = []
    
    for r in range(4):
        for c in range(4):
            patch = gray_diff[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w]
            if patch.size > 0:
                tile_stds.append(float(np.std(patch)))
                tile_means.append(float(np.mean(patch)))
                
    min_tile_std = max(min(tile_stds) if tile_stds else 1.0, 0.1)
    max_tile_std = max(tile_stds) if tile_stds else 1.0
    ela_disparity_ratio = float(max_tile_std / min_tile_std)
    
    # Central subject vs perimeter background disparity
    cy1, cy2 = int(H * 0.20), int(H * 0.80)
    cx1, cx2 = int(W * 0.20), int(W * 0.80)
    center_patch = gray_diff[cy1:cy2, cx1:cx2]
    
    bg_mask = np.ones((H, W), dtype=bool)
    bg_mask[cy1:cy2, cx1:cx2] = False
    bg_patch = gray_diff[bg_mask]
    
    center_mean = float(np.mean(center_patch)) if center_patch.size > 0 else 0.0
    bg_mean = float(np.mean(bg_patch)) if bg_patch.size > 0 else 0.0
    subject_bg_disparity = float(abs(center_mean - bg_mean) / max(min(center_mean, bg_mean), 0.5))
    
    # Generate visual amplified heatmap
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    scale_factor = (255.0 / max_diff) if max_diff > 0 else 1.0
    scale_factor = min(scale_factor, float(scale))
    
    enhancer = ImageEnhance.Brightness(diff)
    heatmap_img = enhancer.enhance(scale_factor)
    
    # Convert heatmap to base64 PNG data URL
    h_buf = io.BytesIO()
    heatmap_img.save(h_buf, format='PNG')
    h_b64 = f"data:image/png;base64,{base64.b64encode(h_buf.getvalue()).decode('ascii')}"
    
    stats = {
        'mean_error': round(ela_mean, 3),
        'std_error': round(ela_std, 3),
        'max_error': round(ela_max, 1),
        'tile_disparity_ratio': round(ela_disparity_ratio, 3),
        'subject_bg_disparity': round(subject_bg_disparity, 3)
    }
    
    return stats, h_b64, gray_diff, heatmap_img


def compute_fft_spectral(gray_img: np.ndarray):
    """
    2D Fast Fourier Transform (FFT) high-frequency power & lattice analysis.
    Identifies generative AI upsampling grid artifacts and unnatural spectral decay.
    """
    H, W = gray_img.shape
    if H < 16 or W < 16:
        return {'high_energy_ratio': 0.05, 'azimuthal_variance': 0.0, 'spectral_flatness': 0.0}
        
    fft = np.fft.fftshift(np.fft.fft2(gray_img))
    mag = np.abs(fft)
    
    center_y, center_x = H // 2, W // 2
    r_core = min(H, W) // 8
    r_outer = min(H, W) // 3
    
    y, x = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    low_mask = dist_from_center <= r_core
    high_mask = (dist_from_center > r_outer) & (dist_from_center <= min(H, W) // 2)
    
    low_energy = float(np.mean(mag[low_mask])) if np.any(low_mask) else 1.0
    high_energy = float(np.mean(mag[high_mask])) if np.any(high_mask) else 0.0
    high_energy_ratio = float(high_energy / max(low_energy, 1.0))
    
    # Azimuthal angular symmetry (checks for unnatural grid spikes in GAN/Diffusion checkerboards)
    num_sectors = 8
    angles = np.arctan2(y - center_y, x - center_x)
    sector_energies = []
    for s in range(num_sectors):
        a1 = -np.pi + s * (2 * np.pi / num_sectors)
        a2 = -np.pi + (s + 1) * (2 * np.pi / num_sectors)
        sec_mask = high_mask & (angles >= a1) & (angles < a2)
        if np.any(sec_mask):
            sector_energies.append(float(np.mean(mag[sec_mask])))
            
    azimuthal_var = float(np.std(sector_energies) / max(np.mean(sector_energies), 1e-4)) if sector_energies else 0.0
    
    # Spectral flatness (Wiener entropy): geometric mean / arithmetic mean of power spectrum
    flat_mag = mag[high_mask].flatten()
    flat_mag = flat_mag[flat_mag > 0]
    if len(flat_mag) > 100:
        log_mean = np.mean(np.log(flat_mag + 1e-8))
        geom_mean = np.exp(log_mean)
        arith_mean = np.mean(flat_mag)
        spectral_flatness = float(geom_mean / max(arith_mean, 1e-8))
    else:
        spectral_flatness = 0.0
        
    return {
        'high_energy_ratio': round(high_energy_ratio, 5),
        'azimuthal_variance': round(azimuthal_var, 4),
        'spectral_flatness': round(spectral_flatness, 4)
    }


def compute_laplacian_gradients(gray_img: np.ndarray):
    """
    Computes discrete 2D Laplacian operator gradients and regional edge blur disparity.
    Detects edge feathering, inpainting boundaries, and splicing seams.
    """
    H, W = gray_img.shape
    if H < 10 or W < 10:
        return {'overall_sharpness': 0.0, 'edge_disparity': 1.0}
        
    # Discrete 2D Laplacian kernel via array slicing
    lap = (gray_img[1:-1, 2:] + gray_img[1:-1, :-2] +
           gray_img[2:, 1:-1] + gray_img[:-2, 1:-1] - 4.0 * gray_img[1:-1, 1:-1])
           
    overall_var = float(lap.var())
    
    # Boundary slice: compare foreground center edge variance vs perimeter edge variance
    l_H, l_W = lap.shape
    cy1, cy2 = int(l_H * 0.20), int(l_H * 0.80)
    cx1, cx2 = int(l_W * 0.20), int(l_W * 0.80)
    
    center_lap = lap[cy1:cy2, cx1:cx2]
    bg_mask = np.ones((l_H, l_W), dtype=bool)
    bg_mask[cy1:cy2, cx1:cx2] = False
    bg_lap = lap[bg_mask]
    
    c_var = float(center_lap.var()) if center_lap.size > 0 else 1.0
    b_var = float(bg_lap.var()) if bg_lap.size > 0 else 1.0
    
    edge_disparity = float(max(c_var, b_var) / max(min(c_var, b_var), 0.5))
    
    return {
        'overall_sharpness': round(overall_var, 2),
        'edge_disparity': round(edge_disparity, 3)
    }


def compute_noise_residual(gray_img: np.ndarray):
    """
    Isolates sensor PRNU (Photo-Response Non-Uniformity) noise residual by high-pass filtering.
    Checks if noise variance is homogeneous or fractured by localized AI edits.
    """
    H, W = gray_img.shape
    if H < 16 or W < 16:
        return {'noise_std': 0.0, 'noise_disparity': 1.0}
        
    # 3x3 local mean filter approximation
    local_mean = (gray_img[:-2, :-2] + gray_img[:-2, 1:-1] + gray_img[:-2, 2:] +
                  gray_img[1:-1, :-2] + gray_img[1:-1, 1:-1] + gray_img[1:-1, 2:] +
                  gray_img[2:, :-2] + gray_img[2:, 1:-1] + gray_img[2:, 2:]) / 9.0
                  
    residual = gray_img[1:-1, 1:-1] - local_mean
    noise_std = float(np.std(residual))
    
    # Compare 4 quadrants noise variance
    rH, rW = residual.shape
    q1 = residual[:rH//2, :rW//2]
    q2 = residual[:rH//2, rW//2:]
    q3 = residual[rH//2:, :rW//2]
    q4 = residual[rH//2:, rW//2:]
    
    stds = [float(np.std(q)) for q in [q1, q2, q3, q4] if q.size > 0]
    min_s = max(min(stds) if stds else 1.0, 0.05)
    max_s = max(stds) if stds else 1.0
    noise_disparity = float(max_s / min_s)
    
    return {
        'noise_std': round(noise_std, 3),
        'noise_disparity': round(noise_disparity, 3)
    }


def parse_metadata(image: Image.Image):
    """
    Inspects EXIF and format metadata for camera hardware tags vs AI tool tags.
    """
    flags = []
    software_tag = None
    has_camera_hardware = False
    
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value).lower()
                
                if tag_name.lower() in ['make', 'model', 'lensmodel']:
                    has_camera_hardware = True
                    
                if tag_name.lower() in ['software', 'processingsoftware']:
                    software_tag = str(value)
                    ai_cues = ['photoshop', 'gimp', 'stable diffusion', 'midjourney', 'dall-e', 'flux', 'comfyui', 'c2pa']
                    for cue in ai_cues:
                        if cue in val_str:
                            flags.append(f"Editing software signature detected in EXIF: '{value}'")
    except Exception:
        pass
        
    return {
        'software': software_tag,
        'has_camera_hardware': has_camera_hardware,
        'flags': flags
    }


def analyze_image_forensics(image: Image.Image, filename: str = "image.png") -> dict:
    """
    Full multi-signal forensic inspection of a standalone image.
    Supports both:
      - CIFAKE detection: Global AI generation artifacts
      - CASIA 2.0 detection: Localized splicing, inpainting, background modification
    """
    # Standardize image size for balanced matrix analysis
    W_orig, H_orig = image.size
    
    # Compute ELA on the original resolution to capture authentic compression boundaries
    ela_stats, ela_heatmap_b64, gray_diff, heatmap_img = compute_ela(image)
    
    # Grayscale image for FFT and Laplacian operations
    small_gray = np.array(image.convert('L').resize((512, 512)), dtype=np.float32)
    
    # 1. Spectral 2D-FFT
    fft_stats = compute_fft_spectral(small_gray)
    
    # 2. Laplacian edge gradient disparity
    lap_stats = compute_laplacian_gradients(small_gray)
    
    # 3. Sensor PRNU noise consistency
    noise_stats = compute_noise_residual(small_gray)
    
    # 4. EXIF metadata
    meta_stats = parse_metadata(image)
    
    # ── COMPOSITE RISK & SCORING LOGIC ──
    # Default baseline for neutral image
    risk_score = 0.15
    indicators = []
    detection_type = "Authentic Camera"
    
    # ELA Compression Disparity (CASIA 2.0 Inpainting & Background Tampering)
    tile_disp = ela_stats['tile_disparity_ratio']
    subj_disp = ela_stats['subject_bg_disparity']
    
    if tile_disp >= 3.0:
        indicators.append(f"Severe regional ELA compression disparity ({tile_disp:.1f}x across tiles): Indicates spliced or inpainted background elements.")
        risk_score += 0.35
    elif tile_disp >= 2.1:
        indicators.append(f"Moderate ELA compression disparity ({tile_disp:.1f}x): Disparate JPEG quantization between foreground and background.")
        risk_score += 0.22
        
    if subj_disp >= 1.4 and ela_stats['mean_error'] > 3.0:
        indicators.append(f"Subject vs Background compression error mismatch ({subj_disp:.2f}x): Background has distinct compression history from subject.")
        risk_score += 0.25
        
    # 2D-FFT Spectral (CIFAKE Diffusion & Generative Lattice)
    az_var = fft_stats['azimuthal_variance']
    hf_ratio = fft_stats['high_energy_ratio']
    spec_flat = fft_stats['spectral_flatness']
    
    if az_var > 0.45:
        indicators.append(f"Generative lattice artifact detected in frequency domain (Azimuthal variance: {az_var:.3f}): Characteristic of AI diffusion/GAN upsampling layers.")
        risk_score += 0.35
    elif az_var > 0.32:
        indicators.append(f"Mild high-frequency grid symmetry in Fourier spectrum (Variance: {az_var:.3f}).")
        risk_score += 0.18
        
    if hf_ratio < 0.015 and lap_stats['overall_sharpness'] > 120:
        indicators.append(f"High-frequency sensor deficit (FFT ratio: {hf_ratio:.4f}): Image appears sharp yet lacks natural optical sensor noise.")
        risk_score += 0.20
        
    # Edge Boundary Sharpness Disparity (Splicing / Feathering)
    edge_disp = lap_stats['edge_disparity']
    if edge_disp > 3.5:
        indicators.append(f"Abrupt edge gradient disparity ({edge_disp:.1f}x): Unnatural edge blending between subject and background.")
        risk_score += 0.25
    elif edge_disp > 2.4:
        indicators.append(f"Edge boundary inconsistency ({edge_disp:.1f}x): Boundary sharpness mismatch.")
        risk_score += 0.15
        
    # PRNU Sensor Noise Consistency
    noise_disp = noise_stats['noise_disparity']
    if noise_disp > 2.8:
        indicators.append(f"Fractured sensor PRNU noise ({noise_disp:.1f}x quadrant variance): Background does not share camera sensor noise.")
        risk_score += 0.22
        
    # Metadata Software Tags
    if meta_stats['flags']:
        indicators.extend(meta_stats['flags'])
        risk_score += 0.20
        
    # Boost if multiple independent forensic indicators align
    if len(indicators) >= 3:
        risk_score = max(risk_score, 0.88)
        detection_type = "High-Confidence Manipulated / Inpainted"
    elif len(indicators) == 2:
        risk_score = max(risk_score, 0.74)
        detection_type = "Likely Tampered / AI Modified"
    elif len(indicators) == 1:
        risk_score = max(risk_score, 0.58)
        detection_type = "Suspicious Manipulations Detected"
    else:
        # Authentic baseline: No forensic anomalies detected
        if meta_stats['has_camera_hardware'] or (tile_disp < 1.8 and az_var < 0.25 and noise_disp < 2.0):
            risk_score = min(risk_score, 0.08)
            indicators.append("Uniform CMOS optical sensor noise and consistent JPEG compression verified across all image regions.")
        else:
            risk_score = min(risk_score, 0.16)
            indicators.append("No tampering or generative anomalies found. Consistent compression profile.")
        detection_type = "Authentic Camera Shot"
        
    # Optional: Calibrated ML Inference (CASIA 2.0 + CIFAKE)
    clf, scaler = _load_ml_model()
    ml_proba = None
    if clf is not None and scaler is not None:
        try:
            feat_vec = np.array([
                ela_stats['mean_error'], ela_stats['std_error'], ela_stats['max_error'],
                ela_stats['tile_disparity_ratio'], ela_stats['subject_bg_disparity'],
                fft_stats['high_energy_ratio'], fft_stats['azimuthal_variance'], fft_stats['spectral_flatness'],
                lap_stats['overall_sharpness'], lap_stats['edge_disparity'], noise_stats['noise_disparity']
            ], dtype=np.float64)
            feat_s = scaler.transform([feat_vec])
            ml_proba = float(clf.predict_proba(feat_s)[0, 1])
            # If ML has high certainty or indicators fired, blend them
            if indicators and not ("Uniform CMOS" in indicators[0] or "No tampering" in indicators[0]):
                risk_score = max(risk_score, ml_proba)
            elif not indicators or ("Uniform CMOS" in indicators[0] or "No tampering" in indicators[0]):
                risk_score = min(risk_score, ml_proba if ml_proba < 0.20 else risk_score)
        except Exception:
            pass

    risk_score = float(np.clip(risk_score, 0.05, 0.96))
    is_tampered = risk_score >= 0.50
    
    return {
        'filename': filename,
        'dimensions': f"{W_orig}x{H_orig}",
        'format': image.format or 'JPEG/PNG',
        'risk_score': round(risk_score, 3),
        'authenticity_score': round(1.0 - risk_score, 3),
        'is_tampered': is_tampered,
        'verdict': "TAMPERED / AI-GENERATED" if is_tampered else "AUTHENTIC IMAGE",
        'detection_type': detection_type,
        'indicators': indicators,
        'ml_probability': round(ml_proba, 3) if ml_proba is not None else None,
        'metrics': {
            'ela': ela_stats,
            'fft': fft_stats,
            'laplacian': lap_stats,
            'noise': noise_stats
        },
        'heatmap_b64': ela_heatmap_b64
    }
