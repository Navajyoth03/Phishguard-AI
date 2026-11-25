#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - ADVANCED MULTIMODAL FUSION ENGINE  v3.0
=====================================================
Optimized Multi-Modal Late Fusion Architecture for Phishing Detection:

Features:
  1. Dynamic Self-Attention Weighting across active modalities
  2. Asymmetric Threat Aggregation (Security Sensor Principle: dominant-risk preservation)
  3. Cross-Modal Anomaly & Trojan Horse Detection (Cloaked clean host + hostile payload override)
  4. 64-D Latent Space Embedding Correlation (Pairwise Cosine Similarity)
  5. Calibrated Multi-Level Risk Categorization & Explainability Flags
"""

import os, sys, math, time, json
from datetime import datetime
import numpy as np
import scipy.spatial.distance as dist

print("=" * 60)
print("[PHISHGUARD] ADVANCED MULTIMODAL FUSION ENGINE v3.0 - LOADING")
print("=" * 60)


class FusionEngine:
    """
    Advanced Multimodal Late Fusion Engine with Asymmetric Risk Aggregation,
    Cross-Attention Dynamic Weighting, and Trojan-Horse Override.
    """
    def __init__(self):
        # Base architectural priors for each modality
        self.base_weights = {
            'url':   1.4,
            'text':  1.3,
            'image': 1.2,
            'audio': 1.1,
            'video': 1.1
        }

    def _align_embeddings(self, emb1, emb2):
        """
        Ensures embeddings are aligned to 64-dimensional latent space.
        """
        e1 = np.asarray(emb1, dtype=np.float64).flatten()
        e2 = np.asarray(emb2, dtype=np.float64).flatten()
        min_len = min(len(e1), len(e2), 64)
        return e1[:min_len], e2[:min_len]

    def _compute_embedding_correlation(self, results_dict):
        """
        Computes pairwise cosine similarity between all available 64-D modality embeddings.
        Returns:
          - avg_similarity: float [0, 1]
          - agreement_multiplier: float [0.85, 1.15]
          - cross_modal_coherence: dict of pairwise similarity scores
        """
        keys = [k for k in results_dict.keys() if 'embedding' in results_dict[k] and results_dict[k]['embedding'] is not None]
        if len(keys) < 2:
            return 1.0, 1.0, {}

        pairwise_sims = {}
        all_sims = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                m1, m2 = keys[i], keys[j]
                emb1 = results_dict[m1]['embedding']
                emb2 = results_dict[m2]['embedding']

                e1, e2 = self._align_embeddings(emb1, emb2)
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)

                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                try:
                    # Cosine similarity = 1 - cosine distance
                    cosine_dist = dist.cosine(e1, e2)
                    sim = float(np.clip(1.0 - cosine_dist, -1.0, 1.0))
                    # Absolute alignment in directional feature space
                    pairwise_sims[f"{m1}_x_{m2}"] = round(sim, 4)
                    all_sims.append(abs(sim))
                except Exception:
                    continue

        if not all_sims:
            return 1.0, 1.0, {}

        avg_similarity = float(np.mean(all_sims))

        # Modality agreement multiplier:
        # High embedding coherence -> models agree on latent structural patterns -> boost confidence [0.90, 1.15]
        # Low embedding coherence -> mixed structural signals -> slight conservative dampening [0.90]
        agreement_multiplier = round(0.90 + (avg_similarity * 0.25), 4)

        return avg_similarity, agreement_multiplier, pairwise_sims

    def fuse(self, multi_modal_inputs):
        """
        Fuses predictions from active modalities using Asymmetric Dynamic Attention.

        Input: dictionary of modality results:
          {
            'url':   {'risk_score': 0.85, 'confidence': 0.90, 'embedding': np.array, 'indicators': [...]},
            'text':  {'risk_score': 0.92, 'confidence': 0.95, 'embedding': np.array, 'indicators': [...]},
            'image': {'risk_score': 0.78, ...},
            ...
          }
        """
        if not multi_modal_inputs:
            return {"error": "No input modalities provided"}

        individual_risks = {}
        individual_confidences = {}
        flags = []
        raw_weights = {}

        # 1. EXTRACT RISKS & COMPUTE DYNAMIC ATTENTION WEIGHTS
        for modality, result in multi_modal_inputs.items():
            if result is None or 'risk_score' not in result:
                continue

            risk = float(result['risk_score'])
            # Confidence is calibrated certainty: |risk - 0.5| * 2.0 or provided confidence
            conf = float(result.get('confidence', abs(risk - 0.5) * 2.0))
            conf = float(np.clip(conf, 0.1, 1.0))

            base_w = self.base_weights.get(modality, 1.0)

            # Dynamic attention formula:
            # - Scales with base architectural importance
            # - Scales with model confidence (polarized predictions carry more authority)
            # - Scales with threat intensity (high risk signals receive attention amplification)
            attention = base_w * (1.0 + 1.2 * conf) * (1.0 + 0.8 * max(0.0, risk - 0.35))
            raw_weights[modality] = attention

            individual_risks[modality] = round(risk, 4)
            individual_confidences[modality] = round(conf, 4)

            # Track modality-specific red flags
            if risk >= 0.70:
                flags.append(f"High threat signal in {modality.upper()} component ({risk:.1%}).")

        if not raw_weights:
            return {"error": "Failed to calculate weights - no valid modalities"}

        # Normalize attention weights
        total_raw_weight = sum(raw_weights.values())
        norm_weights = {m: round(raw_weights[m] / total_raw_weight, 4) for m in raw_weights}

        # 2. BASE WEIGHTED AVERAGE RISK
        weighted_avg_risk = sum(norm_weights[m] * individual_risks[m] for m in raw_weights)

        # 3. ASYMMETRIC AGGREGATION (Security Sensor Principle)
        # If any single modality detects an unequivocal threat (risk >= 0.70),
        # benign modalities must NOT passively dilute the critical alarm!
        max_modality = max(individual_risks, key=individual_risks.get)
        max_risk = individual_risks[max_modality]

        if max_risk >= 0.70:
            # Dynamic threat dominance factor alpha [0.50 -> 0.85]
            alpha = min(0.85, 0.50 + 0.35 * ((max_risk - 0.70) / 0.30))
            asymmetric_risk = (alpha * max_risk) + ((1.0 - alpha) * weighted_avg_risk)
        else:
            asymmetric_risk = weighted_avg_risk

        # 4. CROSS-MODAL ANOMALY / TROJAN HORSE OVERRIDE
        # Scenario: A clean or whitelisted URL (e.g. youtube.com, sites.google.com, notion.site)
        # harboring an active credential harvesting payload in Visual DOM, Audio scam, or Deepfake video
        url_risk = individual_risks.get('url', 0.5)
        payload_modalities = [m for m in ['image', 'audio', 'video'] if m in individual_risks]
        max_payload_risk = max([individual_risks[m] for m in payload_modalities] or [0.0])
        max_payload_mod = max(payload_modalities, key=lambda m: individual_risks[m]) if payload_modalities else None

        trojan_horse_detected = False
        is_deepfake_threat = (multi_modal_inputs.get('video', {}).get('is_deepfake', False) or individual_risks.get('video', 0) >= 0.50)

        if url_risk <= 0.20 and (max_payload_risk >= 0.65 or is_deepfake_threat):
            trojan_horse_detected = True
            # Active deepfake or malicious payload hosted on a clean/trusted domain (e.g. YouTube deepfake)
            target_risk = max(max_payload_risk, 0.78)
            asymmetric_risk = max(asymmetric_risk, target_risk)
            flags.append(
                f"🚨 Synthetic / Deepfake Payload Detected on Platform: Clean host URL ({url_risk:.1%}) "
                f"harboring detected deepfake video in {max_payload_mod.upper()} ({max_payload_risk:.1%})."
            )

        # 5. BENIGN CONSENSUS DAMPENING
        # If all frontline modalities agree the page is safe, dampen residual noise
        all_safe = all(r <= 0.15 for r in individual_risks.values())
        if all_safe:
            asymmetric_risk = min(asymmetric_risk, 0.05)

        # 6. EMBEDDING AGREEMENT CALCULATION
        avg_sim, agreement_multiplier, pairwise_coherence = self._compute_embedding_correlation(multi_modal_inputs)

        # Apply agreement multiplier moderately (within [0.92, 1.10]) only when no contradiction exists
        if not trojan_horse_detected:
            final_risk = asymmetric_risk * agreement_multiplier
        else:
            final_risk = asymmetric_risk

        # Constrain to [0.01, 0.99]
        final_risk = float(np.clip(final_risk, 0.01, 0.99))

        # 7. RISK LEVEL & ACTION CLASSIFICATION
        if final_risk >= 0.80:
            risk_level = "🔴 CRITICAL"
            action = "DO NOT OPEN OR INTERACT. IMMINENT PHISHING THREAT."
        elif final_risk >= 0.60:
            risk_level = "🟠 HIGH"
            action = "VERY SUSPICIOUS. High probability of deceptive attack."
        elif final_risk >= 0.40:
            risk_level = "🟡 MEDIUM"
            action = "PROCEED WITH CAUTION. Verify domain identity and source."
        elif final_risk >= 0.20:
            risk_level = "🔵 LOW"
            action = "GENERALLY SAFE. Keep standard browsing awareness."
        else:
            risk_level = "🟢 SAFE"
            action = "SAFE. Verified legitimate web content."

        return {
            "final_risk_score": round(final_risk, 4),
            "risk_level": risk_level,
            "action_required": action,
            "dominant_modality": max_modality,
            "dominant_risk": max_risk,
            "attention_weights": norm_weights,
            "agreement_multiplier": agreement_multiplier,
            "embedding_similarity": round(avg_sim, 4),
            "pairwise_coherence": pairwise_coherence,
            "trojan_horse_override": trojan_horse_detected,
            "modalities_analyzed": list(individual_risks.keys()),
            "individual_risks": individual_risks,
            "individual_confidences": individual_confidences,
            "red_flags": flags,
            "timestamp": datetime.now().isoformat()
        }


# ==============================================================================
# VERIFICATION SUITE
# ==============================================================================
def test_fusion_engine():
    print("\n" + "=" * 80)
    print("🧪 RUNNING COMPREHENSIVE FUSION ENGINE VERIFICATION SUITE")
    print("=" * 80)

    engine = FusionEngine()

    # TEST CASE 1: Coordinated Phishing Attack (URL + Text + Image all agree)
    print("\n--- TEST CASE 1: Coordinated Phishing Attack (URL + Text + Image) ---")
    mock_case1 = {
        'url':   {'risk_score': 0.85, 'confidence': 0.90, 'embedding': np.ones(64) * 0.8},
        'text':  {'risk_score': 0.92, 'confidence': 0.95, 'embedding': np.ones(64) * 0.82},
        'image': {'risk_score': 0.78, 'confidence': 0.85, 'embedding': np.ones(64) * 0.79}
    }
    res1 = engine.fuse(mock_case1)
    print(f"  Final Risk Score : {res1['final_risk_score']:.2%}")
    print(f"  Risk Level       : {res1['risk_level']}")
    print(f"  Dominant Modality: {res1['dominant_modality']}")
    print(f"  Attention Weights: {res1['attention_weights']}")
    assert res1['final_risk_score'] >= 0.80, "Test 1 failed!"
    print("  ✓ PASS: Correctly identified as CRITICAL threat.")

    # TEST CASE 2: Vishing Attack on Clean URL (URL looks safe, but Audio is high-risk vishing)
    print("\n--- TEST CASE 2: Vishing Attack on Clean URL (Clean URL + Vishing Audio) ---")
    mock_case2 = {
        'url':   {'risk_score': 0.10, 'confidence': 0.85, 'embedding': np.random.rand(64)},
        'audio': {'risk_score': 0.88, 'confidence': 0.92, 'embedding': np.random.rand(64)}
    }
    res2 = engine.fuse(mock_case2)
    print(f"  Final Risk Score : {res2['final_risk_score']:.2%}")
    print(f"  Risk Level       : {res2['risk_level']}")
    print(f"  Dominant Modality: {res2['dominant_modality']}")
    print(f"  Trojan Override  : {res2['trojan_horse_override']}")
    print(f"  Red Flags        : {res2['red_flags']}")
    assert res2['final_risk_score'] >= 0.80, "Test 2 failed to protect user from vishing!"
    print("  ✓ PASS: Security sensor principle protected user from vishing bypass.")

    # TEST CASE 3: Cloud Cloaked Trojan Horse (sites.google.com + Fake PayPal Login Form)
    print("\n--- TEST CASE 3: Cloud Cloaked Trojan Horse (Google Sites + Fake PayPal Login) ---")
    mock_case3 = {
        'url':   {'risk_score': 0.05, 'confidence': 0.95, 'embedding': np.random.rand(64)},
        'image': {'risk_score': 0.94, 'confidence': 0.92, 'embedding': np.random.rand(64)},
        'text':  {'risk_score': 0.86, 'confidence': 0.88, 'embedding': np.random.rand(64)}
    }
    res3 = engine.fuse(mock_case3)
    print(f"  Final Risk Score : {res3['final_risk_score']:.2%}")
    print(f"  Risk Level       : {res3['risk_level']}")
    print(f"  Trojan Override  : {res3['trojan_horse_override']}")
    print(f"  Red Flags        : {res3['red_flags']}")
    assert res3['trojan_horse_override'] is True, "Test 3 failed to trigger Trojan Horse override!"
    assert res3['final_risk_score'] >= 0.82, "Test 3 failed to reach CRITICAL!"
    print("  ✓ PASS: Successfully detected Trojan Horse and overrode clean URL.")

    # TEST CASE 4: Completely Safe Legitimate Website (Google / Wikipedia)
    print("\n--- TEST CASE 4: Completely Safe Website (Google / Wikipedia) ---")
    mock_case4 = {
        'url':   {'risk_score': 0.03, 'confidence': 0.95, 'embedding': np.ones(64) * 0.1},
        'text':  {'risk_score': 0.04, 'confidence': 0.90, 'embedding': np.ones(64) * 0.1},
        'image': {'risk_score': 0.05, 'confidence': 0.85, 'embedding': np.ones(64) * 0.12}
    }
    res4 = engine.fuse(mock_case4)
    print(f"  Final Risk Score : {res4['final_risk_score']:.2%}")
    print(f"  Risk Level       : {res4['risk_level']}")
    assert res4['final_risk_score'] <= 0.05, "Test 4 failed to identify clean website!"
    print("  ✓ PASS: Clean consensus dampener verified safe website.")

    print("\n" + "=" * 80)
    print("✅ ALL 4 MULTIMODAL FUSION TEST CASES PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_fusion_engine()
