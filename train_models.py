#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHISHGUARD - MULTI-MODAL MODEL TRAINING PIPELINE  v2.0
========================================================
Trains ALL ML models for the Multi-Modal Phishing Detection System
using REAL-WORLD datasets (PhishTank, Kaggle, Enron/Nazario).

Models trained:
  1. URL  Phishing Detector    — Ensemble (RF + GB + LR) on 63 structural features
  2. Text Phishing Classifier  — TF-IDF (15k features) + Logistic Regression
  3. Image/Visual DOM Detector — Logistic Regression on 16 DOM/layout features
  4. Audio Vishing Detector    — Logistic Regression on 12 acoustic features
  5. Video Deepfake Detector   — Logistic Regression on 8 CV forensic features

Data sources:
  - datasets/malicious_phish.csv       (651k real URLs — Kaggle)
  - datasets/phishing_site_urls.csv    (549k real URLs — Kaggle)
  - datasets/verified_online.csv       (PhishTank verified URLs)
  - datasets/phishing_emails_kaggle.csv(18.6k real emails — Enron/Nazario)
  - datasets/visual_dataset.csv        (DOM/layout feature benchmark)
  - datasets/audio_dataset.csv         (acoustic/vishing feature benchmark)
  - datasets/video_dataset.csv         (deepfake forensic feature benchmark)

Usage:
  python train_models.py                # Train all models
  python train_models.py --url          # URL model only
  python train_models.py --text         # Text model only
  python train_models.py --image        # Image/Visual model only
  python train_models.py --audio        # Audio model only
  python train_models.py --video        # Video model only
"""

import os, sys, re, csv, json, random, string, time, math, argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Increase CSV field size limit for large email bodies (2GB max 32-bit C long limit)
csv.field_size_limit(2147483647)

from urllib.parse import urlparse, parse_qs
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, matthews_corrcoef)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
DATASETS_DIR    = os.path.join(BASE_DIR, 'datasets')
URL_MODEL_DIR   = os.path.join(MODELS_DIR, 'url_model')
TEXT_MODEL_DIR  = os.path.join(MODELS_DIR, 'text_model')
IMAGE_MODEL_DIR = os.path.join(MODELS_DIR, 'image_model')
AUDIO_MODEL_DIR = os.path.join(MODELS_DIR, 'audio_model')
VIDEO_MODEL_DIR = os.path.join(MODELS_DIR, 'video_model')

for d in [MODELS_DIR, URL_MODEL_DIR, TEXT_MODEL_DIR, IMAGE_MODEL_DIR, AUDIO_MODEL_DIR, VIDEO_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# How many real URLs to sample for training (balances speed vs coverage)
URL_SAMPLE_SIZE   = 50_000   # 50k phishing + 50k benign = 100k total
EMBEDDING_DIM     = 64

# ─── BANNER ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("  PHISHGUARD - MULTI-MODAL TRAINING PIPELINE v2.0")
print("  Training on REAL-WORLD datasets")
print("=" * 70)
print(f"  Output directory : {MODELS_DIR}")
print(f"  Dataset directory: {DATASETS_DIR}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1:  URL MODEL  (Real Data)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary & Constants ──────────────────────────────────────────────────
LEGIT_DOMAINS = [
    "google.com","gmail.com","youtube.com","facebook.com","twitter.com",
    "linkedin.com","github.com","stackoverflow.com","wikipedia.org","amazon.com",
    "microsoft.com","apple.com","netflix.com","instagram.com","reddit.com",
    "discord.com","spotify.com","zoom.us","adobe.com","dropbox.com",
    "stripe.com","shopify.com","twitch.tv","medium.com","notion.so",
    "cloudflare.com","vercel.com","heroku.com","digitalocean.com","aws.amazon.com",
    "flipkart.com","amazon.in","myntra.com","bigbasket.com","swiggy.com",
    "zomato.com","paytm.com","phonepe.com","razorpay.com","freshworks.com",
    "zoho.com","irctc.co.in","sbi.co.in","hdfcbank.com","icicibank.com",
    "axisbank.com","india.gov.in","ndtv.com","thehindu.com","makemytrip.com",
    "bookmyshow.com","naukri.com","ola.com","oyo.com","meesho.com",
    "iitb.ac.in","iitd.ac.in","iitm.ac.in","iisc.ac.in","nptel.ac.in",
    "byjus.com","unacademy.com","vedantu.com","zerodha.com","groww.in",
    "policybazaar.com","practo.com","1mg.com","netmeds.com","pharmeasy.in",
]

BRAND_NAMES = [
    "paypal","google","amazon","facebook","apple","microsoft","netflix",
    "sbi","hdfc","icici","irctc","paytm","flipkart","whatsapp","instagram",
    "twitter","linkedin","ebay","yahoo","outlook","onedrive","dropbox",
    "zoom","spotify","ola","zomato","swiggy","razorpay","phonepe",
    "airtel","vodafone","jio","bsnl","lic","bajaj","tatamotors","reliance",
]

SUSPICIOUS_WORDS = [
    "verify","secure","login","signin","account","update","confirm",
    "banking","payment","credential","recovery","auth","validate",
    "required","urgent","alert","suspended","restricted","unlock",
    "activate","billing","invoice","refund","claim","prize","reward",
    "password","otp","kyc","aadhar","pancard","ifsc","swift",
]

SUSP_TLD_SET = {'.tk','.ml','.ga','.cf','.xyz','.club','.online','.site',
               '.website','.space','.top','.click','.download','.review',
               '.work','.loan','.win','.bid','.gq','.pw','.cc','.buzz',
               '.rest','.link','.life','.live','.shop','.store'}

SUSP_KEYWORD_SET = set(SUSPICIOUS_WORDS)
LEGIT_DOMAIN_SET = set(LEGIT_DOMAINS)
BRAND_SET = set(BRAND_NAMES)

SHORTENERS = {'bit.ly','tinyurl','goo.gl','ow.ly','cutt.ly','rb.gy','t.co',
              'tiny.cc','is.gd','shorturl','buff.ly','tr.im'}

SUSP_PARAMS = {'redirect','url','return','next','dest','goto','callback','ref'}

HOMOGLYPH_TABLE = str.maketrans({
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '8': 'b',
    '@': 'a', '$': 's', '!': 'i',
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i', 'ј': 'j'
})

DECEPTIVE_TRIGRAMS = {
    'tok', 'aut', 'uth', 'ver', 'eri', 'rif', 'upd', 'dat', 'sec',
    'log', 'sso', 'pwd', 'pas', 'pay', 'bil', 'bnk', 'act', 'val',
    'kyc', 'mfa', 'otp', 'con', 'rec', 'rdr', 'acc'
}

EPHEMERAL_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.click', '.buzz',
    '.cfd', '.rest', '.icu', '.site', '.space', '.website', '.club',
    '.online', '.live', '.loan', '.win', '.bid'
}


def normalize_homoglyphs(text: str) -> str:
    s = text.lower().replace('rn', 'm').replace('vv', 'w').replace('cl', 'd')
    return s.translate(HOMOGLYPH_TABLE)


def levenshtein_dist(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def extract_url_features(url: str) -> dict:
    """Extract 63 structural features from a URL."""
    f = {}
    try:
        # Ensure URL has a scheme for urlparse
        if not url.startswith(('http://', 'https://', 'ftp://')):
            url = 'http://' + url

        p    = urlparse(url)
        host = (p.hostname or "").lower()
        path = p.path or ""
        qstr = p.query or ""
        url_l = url.lower()

        # 1-7: Lengths & ratios
        f['url_length']        = len(url)
        f['host_length']       = len(host)
        f['path_length']       = len(path)
        f['query_length']      = len(qstr)
        f['host_ratio']        = len(host) / max(len(url), 1)
        f['path_ratio']        = len(path) / max(len(url), 1)
        f['is_very_long']      = 1 if len(url) > 100 else 0

        # 8-22: Character counts
        f['num_dots']          = url.count('.')
        f['num_hyphens']       = url.count('-')
        f['num_underscores']   = url.count('_')
        f['num_slashes']       = url.count('/')
        f['num_qs']            = url.count('?')
        f['num_equals']        = url.count('=')
        f['num_amps']          = url.count('&')
        f['num_at']            = url.count('@')
        f['num_percent']       = url.count('%')
        f['num_digits']        = sum(c.isdigit() for c in url)
        f['num_letters']       = sum(c.isalpha() for c in url)
        f['digit_ratio']       = f['num_digits'] / max(len(url), 1)
        f['letter_ratio']      = f['num_letters'] / max(len(url), 1)
        special                = sum(c in '-_./?=&%#@:;+~$' for c in url)
        f['special_ratio']     = special / max(len(url), 1)
        f['alphanumeric_ratio']= (f['num_digits']+f['num_letters']) / max(len(url), 1)

        # 23-30: Structural flags
        f['has_ip']            = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', host) else 0
        f['has_https']         = 1 if p.scheme == 'https' else 0
        f['has_at']            = 1 if '@' in url else 0
        f['has_double_slash']  = 1 if '//' in url[8:] else 0
        f['has_dash_host']     = 1 if '-' in host else 0
        f['num_dashes_host']   = host.count('-')
        f['has_www']           = 1 if url_l.startswith('http://www.') or url_l.startswith('https://www.') else 0
        f['has_port']          = 1 if p.port and p.port not in (80, 443) else 0

        # 31-37: Domain structure
        parts                  = host.replace('www.', '').split('.')
        f['domain_length']     = len(parts[-2]) if len(parts) >= 2 else len(host)
        f['tld_length']        = len(parts[-1]) if parts else 0
        f['subdomain_count']   = max(len(parts) - 2, 0)
        f['has_subdomain']     = 1 if f['subdomain_count'] > 0 else 0
        f['many_subdomains']   = 1 if f['subdomain_count'] > 2 else 0
        f['domain_has_digits'] = 1 if any(c.isdigit() for c in (parts[-2] if len(parts) >= 2 else '')) else 0
        root_domain            = '.'.join(parts[-2:]) if len(parts) >= 2 else host
        f['is_known_legit']    = 1 if root_domain in LEGIT_DOMAIN_SET else 0

        # 38-40: TLD analysis
        suffix = '.' + parts[-1] if parts else ''
        f['has_susp_tld']      = 1 if suffix in SUSP_TLD_SET else 0
        f['tld_in_url_count']  = sum(url_l.count(t) for t in SUSP_TLD_SET)
        f['is_shortened']      = 1 if any(s in url_l for s in SHORTENERS) else 0

        # 41-44: Keyword analysis
        kw_hits                = sum(1 for kw in SUSP_KEYWORD_SET if kw in url_l)
        f['num_susp_kw']       = kw_hits
        f['has_susp_kw']       = 1 if kw_hits > 0 else 0
        path_kw                = sum(1 for kw in SUSP_KEYWORD_SET if kw in path.lower())
        f['num_susp_kw_path']  = path_kw
        f['has_susp_kw_path']  = 1 if path_kw > 0 else 0

        # 45-48: Brand impersonation
        brand_in_url           = sum(1 for b in BRAND_SET if b in url_l)
        f['brand_in_url']      = brand_in_url
        brand_in_host          = sum(1 for b in BRAND_SET if b in host)
        brand_legit            = any(b in root_domain for b in BRAND_SET)
        f['brand_host_imperson'] = 1 if brand_in_host > 0 and not brand_legit else 0
        brand_in_path          = sum(1 for b in BRAND_SET if b in path.lower())
        f['brand_in_path']     = 1 if brand_in_path > 0 else 0
        f['brand_not_host']    = 1 if (brand_in_url > 0 and not brand_legit) else 0

        # 49-52: Path analysis
        path_segs              = [s for s in path.split('/') if s]
        f['path_depth']        = len(path_segs)
        f['avg_seg_len']       = sum(len(s) for s in path_segs) / max(len(path_segs), 1)
        f['has_long_path']     = 1 if len(path) > 50 else 0
        f['has_php_ext']       = 1 if path.lower().endswith('.php') else 0

        # 53-54: Query parameters
        qparams                = parse_qs(qstr)
        f['num_qparams']       = len(qparams)
        f['has_susp_param']    = 1 if any(k in SUSP_PARAMS for k in qparams) else 0

        # 55-57: Entropy (randomness)
        if url:
            freq = Counter(url)
            entr = -sum((c/len(url)) * math.log2(c/len(url)) for c in freq.values())
            f['entropy']       = min(entr, 8.0)
            f['norm_entropy']  = entr / 8.0
        else:
            f['entropy'] = f['norm_entropy'] = 0.0

        # Max consecutive repeated character
        max_rep = run = 1
        for i in range(1, len(url)):
            run = run + 1 if url[i] == url[i-1] else 1
            max_rep = max(max_rep, run)
        f['max_repeat']        = min(max_rep / 10.0, 1.0)

        # 58-60: Typosquatting & Homoglyphs
        domain_candidate = parts[-2] if len(parts) >= 2 else host
        norm_domain = normalize_homoglyphs(domain_candidate)
        subdomain_str = '.'.join(parts[:-2]) if len(parts) > 2 else ''
        norm_subdomain = normalize_homoglyphs(subdomain_str)

        is_typosquat = 0
        min_lev = 99
        has_homo = 1 if (norm_domain != domain_candidate or norm_subdomain != subdomain_str) else 0

        domain_tokens = [t for t in re.split(r'[-_.]', domain_candidate) if len(t) >= 3]
        if '-' in domain_candidate and len(domain_candidate) >= 3:
            domain_tokens.append(domain_candidate)

        for tok in domain_tokens:
            norm_tok = normalize_homoglyphs(tok)
            for b in BRAND_SET:
                if tok == b:
                    continue
                if norm_tok == b:
                    is_typosquat = 1
                    has_homo = 1
                    min_lev = 0
                    break
                if abs(len(tok) - len(b)) <= 2:
                    d = levenshtein_dist(tok, b)
                    if d < min_lev:
                        min_lev = d
                    if d == 1 or (d == 2 and len(b) >= 6):
                        is_typosquat = 1
                        break
            if is_typosquat:
                break

        f['is_typosquat']       = is_typosquat
        f['min_brand_lev_dist'] = min(min_lev / 10.0, 1.0) if min_lev < 90 else 1.0
        f['has_homoglyph']      = has_homo

        # 61-62: Character N-Gram sub-word features
        subword_hits = sum(1 for tri in DECEPTIVE_TRIGRAMS if tri in url_l)
        f['subword_phish_density'] = min(subword_hits / max(len(url_l) / 10.0, 1.0), 10.0)

        if len(url_l) >= 3:
            trigrams = [url_l[i:i+3] for i in range(len(url_l) - 2)]
            tri_counts = Counter(trigrams)
            tri_entr = -sum((c / len(trigrams)) * math.log2(c / len(trigrams)) for c in tri_counts.values())
            f['char_trigram_entropy'] = min(tri_entr / 6.0, 1.0)
        else:
            f['char_trigram_entropy'] = 0.0

        # 63: Domain Freshness / Ephemeral Infrastructure Heuristic
        has_dated_pattern = 1 if re.search(r'-(202[4-9]|\d{2,4})\b', host) else 0
        has_ephemeral_tld = 1 if suffix in EPHEMERAL_TLDS else 0
        f['domain_freshness_risk'] = 1 if (has_ephemeral_tld and (has_dated_pattern or f['entropy'] > 4.1 or is_typosquat)) else 0

    except Exception:
        f = {k: 0 for k in [
            'url_length','host_length','path_length','query_length','host_ratio',
            'path_ratio','is_very_long','num_dots','num_hyphens','num_underscores',
            'num_slashes','num_qs','num_equals','num_amps','num_at','num_percent',
            'num_digits','num_letters','digit_ratio','letter_ratio','special_ratio',
            'alphanumeric_ratio','has_ip','has_https','has_at','has_double_slash',
            'has_dash_host','num_dashes_host','has_www','has_port','domain_length',
            'tld_length','subdomain_count','has_subdomain','many_subdomains',
            'domain_has_digits','is_known_legit','has_susp_tld','tld_in_url_count',
            'is_shortened','num_susp_kw','has_susp_kw','num_susp_kw_path',
            'has_susp_kw_path','brand_in_url','brand_host_imperson','brand_in_path',
            'brand_not_host','path_depth','avg_seg_len','has_long_path','has_php_ext',
            'num_qparams','has_susp_param','entropy','norm_entropy','max_repeat',
            'is_typosquat','min_brand_lev_dist','has_homoglyph','subword_phish_density',
            'char_trigram_entropy','domain_freshness_risk'
        ]}
    return f

FEATURE_NAMES = list(extract_url_features("https://google.com").keys())


def extract_feature_matrix(urls, verbose=True):
    n = len(urls)
    num_feat = len(FEATURE_NAMES)
    X = np.zeros((n, num_feat), dtype=np.float32)
    for i, u in enumerate(urls):
        feat = extract_url_features(u)
        for j, k in enumerate(FEATURE_NAMES):
            X[i, j] = feat.get(k, 0.0)
        if verbose and (i+1) % 10000 == 0:
            print(f"     Feature extraction: {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%)")
    return X


# ─── Load Real URL Datasets ──────────────────────────────────────────────────
def load_real_url_data(sample_per_class=URL_SAMPLE_SIZE):
    """
    Loads real-world URL datasets from CSV files and creates a balanced dataset.
    Sources:
      - malicious_phish.csv:    columns [url, type] where type ∈ {phishing, benign, defacement, malware}
      - phishing_site_urls.csv: columns [URL, Label] where Label ∈ {bad, good}
      - verified_online.csv:    PhishTank feed — all phishing (column: url)
    """
    print(f"\n[1/6] Loading real-world URL datasets...")
    t0 = time.time()

    phishing_urls = []
    benign_urls = []

    # --- Source 1: malicious_phish.csv ---
    path1 = os.path.join(DATASETS_DIR, 'malicious_phish.csv')
    if os.path.exists(path1):
        count_p, count_b = 0, 0
        with open(path1, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url', '').strip()
                label = row.get('type', '').strip().lower()
                if not url:
                    continue
                if label in ('phishing', 'defacement', 'malware'):
                    phishing_urls.append(url)
                    count_p += 1
                elif label == 'benign':
                    benign_urls.append(url)
                    count_b += 1
        print(f"     malicious_phish.csv    : {count_p:>8,} phishing, {count_b:>8,} benign")
    else:
        print(f"     ⚠ malicious_phish.csv not found")

    # --- Source 2: phishing_site_urls.csv ---
    path2 = os.path.join(DATASETS_DIR, 'phishing_site_urls.csv')
    if os.path.exists(path2):
        count_p, count_b = 0, 0
        with open(path2, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('URL', '').strip()
                label = row.get('Label', '').strip().lower()
                if not url:
                    continue
                if label == 'bad':
                    phishing_urls.append(url)
                    count_p += 1
                elif label == 'good':
                    benign_urls.append(url)
                    count_b += 1
        print(f"     phishing_site_urls.csv : {count_p:>8,} phishing, {count_b:>8,} benign")
    else:
        print(f"     ⚠ phishing_site_urls.csv not found")

    # --- Source 3: verified_online.csv (PhishTank — all phishing) ---
    path3 = os.path.join(DATASETS_DIR, 'verified_online.csv')
    if os.path.exists(path3):
        count = 0
        with open(path3, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url', '').strip()
                if url:
                    phishing_urls.append(url)
                    count += 1
        print(f"     verified_online.csv    : {count:>8,} phishing")
    else:
        print(f"     ⚠ verified_online.csv not found")

    # Deduplicate
    phishing_urls = list(set(phishing_urls))
    benign_urls = list(set(benign_urls))

    print(f"\n     Total unique: {len(phishing_urls):,} phishing, {len(benign_urls):,} benign")

    # Balance & sample
    n_phish = min(sample_per_class, len(phishing_urls))
    n_benign = min(sample_per_class, len(benign_urls))
    n_use = min(n_phish, n_benign)  # balance

    random.shuffle(phishing_urls)
    random.shuffle(benign_urls)

    sampled_phish = phishing_urls[:n_use]
    sampled_benign = benign_urls[:n_use]

    urls = sampled_phish + sampled_benign
    labels = [1] * n_use + [0] * n_use

    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)

    print(f"     Balanced sample: {n_use:,} phishing + {n_use:,} benign = {n_use*2:,} total")
    print(f"     Data loaded in {time.time()-t0:.1f}s")
    return list(urls), list(labels)


# ─── URL Model Trainer ────────────────────────────────────────────────────────
def train_url_model(urls=None, labels=None):
    print("\n" + "="*70)
    print("  URL MODEL TRAINING  (Real-World Data)")
    print("="*70)

    if urls is None or labels is None:
        urls, labels = load_real_url_data()

    print(f"\n[2/6] Extracting URL features ({len(urls):,} URLs, {len(FEATURE_NAMES)} features)...")
    t0 = time.time()
    X = extract_feature_matrix(urls, verbose=True)
    print(f"     Done in {time.time()-t0:.1f}s")

    y = np.array(labels)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n[3/6] Training URL ensemble model...")
    print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Feature selection (top 50 features)
    n_select = min(50, X_train_scaled.shape[1])
    selector = SelectKBest(f_classif, k=n_select)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)
    print(f"     Selected {n_select} best features via ANOVA F-test")

    # PCA embedding for fusion (64-dim)
    n_pca = min(EMBEDDING_DIM, X_train_sel.shape[1])
    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
    pca.fit(X_train_sel)
    explained = pca.explained_variance_ratio_.sum()
    print(f"     PCA: {n_select} -> {n_pca} dims  (explained variance: {explained:.1%})")

    # Ensemble
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_split=5,
        class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE
    )
    gb  = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        subsample=0.85, random_state=RANDOM_STATE
    )
    lr  = LogisticRegression(
        C=2.0, max_iter=1000, class_weight='balanced',
        solver='lbfgs', random_state=RANDOM_STATE
    )

    t1 = time.time()
    print("     Training RandomForest (200 trees)...")
    rf.fit(X_train_sel, y_train)
    print(f"     Training GradientBoosting (150 trees)... ({time.time()-t1:.0f}s so far)")
    gb.fit(X_train_sel, y_train)
    print(f"     Training LogisticRegression... ({time.time()-t1:.0f}s so far)")
    lr.fit(X_train_sel, y_train)

    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft'
    )
    model.fit(X_train_sel, y_train)
    print(f"     Total training time: {time.time()-t1:.1f}s")

    # Evaluate
    y_pred  = model.predict(X_test_sel)
    y_proba = model.predict_proba(X_test_sel)[:, 1]

    metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auc':       round(float(roc_auc_score(y_test, y_proba)), 4),
        'mcc':       round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'train_size': int(len(X_train)),
        'test_size':  int(len(X_test)),
        'features':   len(FEATURE_NAMES),
        'selected_features': n_select,
        'pca_dims':   n_pca,
        'pca_explained_variance': round(float(explained), 4),
        'data_source': 'malicious_phish.csv + phishing_site_urls.csv + verified_online.csv (real-world)',
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  URL Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    MCC      : {metrics['mcc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    # Cross-validation
    print("     Running 5-fold cross-validation...")
    cv_scores = cross_val_score(lr, X_train_sel, y_train, cv=5, scoring='f1', n_jobs=-1)
    metrics['cv_f1_mean'] = round(float(cv_scores.mean()), 4)
    metrics['cv_f1_std'] = round(float(cv_scores.std()), 4)
    print(f"     CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save artifacts
    print(f"\n[4/6] Saving URL model to {URL_MODEL_DIR}...")
    joblib.dump(model,    os.path.join(URL_MODEL_DIR, 'model.joblib'),    compress=3)
    joblib.dump(scaler,   os.path.join(URL_MODEL_DIR, 'scaler.joblib'),   compress=3)
    joblib.dump(selector, os.path.join(URL_MODEL_DIR, 'selector.joblib'), compress=3)
    joblib.dump(pca,      os.path.join(URL_MODEL_DIR, 'pca.joblib'),      compress=3)

    with open(os.path.join(URL_MODEL_DIR, 'feature_names.json'), 'w') as fp:
        json.dump(FEATURE_NAMES, fp)
    with open(os.path.join(URL_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  ✓ URL model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2:  TEXT MODEL  (Real Data)
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_text_data():
    """
    Loads real-world email/text dataset.
    Source: phishing_emails_kaggle.csv — columns [Email Text, Email Type]
            where Email Type ∈ {'Phishing Email', 'Safe Email'}
    """
    print(f"\n[1/4] Loading real-world email dataset...")
    t0 = time.time()

    path = os.path.join(DATASETS_DIR, 'phishing_emails_kaggle.csv')
    texts = []
    labels = []

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('Email Text', '').strip()
                email_type = row.get('Email Type', '').strip()
                if not text or len(text) < 20:
                    continue
                if email_type == 'Phishing Email':
                    texts.append(text)
                    labels.append(1)
                elif email_type == 'Safe Email':
                    texts.append(text)
                    labels.append(0)

        n_phish = sum(labels)
        n_safe = len(labels) - n_phish
        print(f"     phishing_emails_kaggle.csv: {n_phish:,} phishing, {n_safe:,} safe emails")
    else:
        print(f"     ⚠ phishing_emails_kaggle.csv not found — falling back to synthetic data")
        return None, None

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    print(f"     Total: {len(texts):,} emails loaded in {time.time()-t0:.1f}s")
    return list(texts), list(labels)


def train_text_model(texts=None, labels=None):
    print("\n" + "="*70)
    print("  TEXT MODEL TRAINING  (Real-World Emails)")
    print("="*70)

    if texts is None or labels is None:
        result = load_real_text_data()
        if result[0] is None:
            print("  ⚠ No text data available, skipping.")
            return None
        texts, labels = result

    y = np.array(labels)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n     Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")
    print(f"     Phishing in train: {sum(y_train):,} | Safe in train: {len(y_train)-sum(y_train):,}")

    print(f"\n[2/4] Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        stop_words='english',
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)
    print(f"     Vocabulary size: {len(vectorizer.vocabulary_):,}")

    print(f"\n[3/4] Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        C=5.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=RANDOM_STATE,
    )
    t1 = time.time()
    classifier.fit(X_train, y_train)
    print(f"     Training time: {time.time()-t1:.2f}s")

    # Evaluate
    y_pred  = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auc':       round(float(roc_auc_score(y_test, y_proba)), 4),
        'mcc':       round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'train_size': int(len(X_train_raw)),
        'test_size':  int(len(X_test_raw)),
        'vocab_size': len(vectorizer.vocabulary_),
        'data_source': 'phishing_emails_kaggle.csv (18.6k real emails — Enron/Nazario)',
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Text Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    MCC      : {metrics['mcc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    # Cross-validation
    print("     Running 5-fold cross-validation...")
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
    metrics['cv_f1_mean'] = round(float(cv_scores.mean()), 4)
    metrics['cv_f1_std'] = round(float(cv_scores.std()), 4)
    print(f"     CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save
    print(f"\n[4/4] Saving text model to {TEXT_MODEL_DIR}...")
    joblib.dump(vectorizer,  os.path.join(TEXT_MODEL_DIR, 'vectorizer.joblib'),  compress=3)
    joblib.dump(classifier,  os.path.join(TEXT_MODEL_DIR, 'classifier.joblib'), compress=3)
    with open(os.path.join(TEXT_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  ✓ Text model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3:  IMAGE / VISUAL DOM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_visual_dataset():
    """
    Load visual/DOM layout feature dataset.
    Features: ext_img_ratio, has_favicon, foreign_favicon, hidden_elems,
              fixed_overlay, form_action_external, has_pwd, brand_logo_present,
              input_count, iframe_count, opacity_trick, z_index_abuse,
              fake_url_bar, right_click_disabled, page_has_login_form, brand_visual
    """
    path = os.path.join(DATASETS_DIR, 'visual_dataset.csv')
    if not os.path.exists(path):
        print(f"     ⚠ visual_dataset.csv not found — generating synthetic visual data")
        return _generate_visual_data()

    print(f"     Loading visual_dataset.csv...")
    features = []
    labels = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = None
        for row in reader:
            if cols is None:
                cols = [c for c in reader.fieldnames if c != 'label']
            feat = []
            for c in cols:
                try:
                    feat.append(float(row.get(c, 0)))
                except (ValueError, TypeError):
                    feat.append(0.0)
            features.append(feat)
            try:
                labels.append(int(row.get('label', 0)))
            except (ValueError, TypeError):
                labels.append(0)

    print(f"     Visual dataset: {len(features):,} samples, {len(features[0]) if features else 0} features")
    return np.array(features, dtype=np.float32), np.array(labels), cols


def _generate_visual_data():
    """Fallback: generate synthetic visual/DOM feature data."""
    n = 5000
    cols = ['ext_img_ratio','has_favicon','foreign_favicon','hidden_elems',
            'fixed_overlay','form_action_external','has_pwd','brand_logo_present',
            'input_count','iframe_count','opacity_trick','z_index_abuse',
            'fake_url_bar','right_click_disabled','page_has_login_form','brand_visual']

    features = np.zeros((n, len(cols)), dtype=np.float32)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        is_phish = random.random() < 0.5
        labels[i] = 1 if is_phish else 0
        if is_phish:
            features[i] = [
                random.uniform(0.3, 0.9),   # ext_img_ratio
                random.choice([0, 1]),       # has_favicon
                random.choice([0, 1]) if random.random() > 0.4 else 0,  # foreign_favicon
                random.randint(0, 8),        # hidden_elems
                random.choice([0, 1]) if random.random() > 0.3 else 0,  # fixed_overlay
                random.choice([0, 1]) if random.random() > 0.3 else 0,  # form_action_external
                1 if random.random() > 0.3 else 0,  # has_pwd
                random.choice([0, 1]),       # brand_logo_present
                random.randint(2, 10),       # input_count
                random.randint(0, 3),        # iframe_count
                random.choice([0, 1]) if random.random() > 0.5 else 0,  # opacity_trick
                random.choice([0, 1]) if random.random() > 0.5 else 0,  # z_index_abuse
                random.choice([0, 1]) if random.random() > 0.6 else 0,  # fake_url_bar
                random.choice([0, 1]) if random.random() > 0.5 else 0,  # right_click_disabled
                1 if random.random() > 0.3 else 0,  # page_has_login_form
                random.choice([0, 1]),       # brand_visual
            ]
        else:
            features[i] = [
                random.uniform(0.0, 0.3),    # ext_img_ratio
                1,                            # has_favicon
                0,                            # foreign_favicon
                random.randint(0, 2),         # hidden_elems
                0,                            # fixed_overlay
                0,                            # form_action_external
                1 if random.random() > 0.7 else 0,  # has_pwd
                0,                            # brand_logo_present
                random.randint(0, 4),         # input_count
                0,                            # iframe_count
                0,                            # opacity_trick
                0,                            # z_index_abuse
                0,                            # fake_url_bar
                0,                            # right_click_disabled
                1 if random.random() > 0.6 else 0,  # page_has_login_form
                0,                            # brand_visual
            ]

    print(f"     Generated {n:,} synthetic visual samples, {len(cols)} features")
    return features, labels, cols


def train_image_model():
    print("\n" + "="*70)
    print("  IMAGE/VISUAL DOM MODEL TRAINING")
    print("="*70)

    result = load_visual_dataset()
    X, y, feature_cols = result

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifier = LogisticRegression(
        C=1.0, max_iter=500, class_weight='balanced',
        solver='lbfgs', random_state=RANDOM_STATE
    )
    t1 = time.time()
    classifier.fit(X_train_s, y_train)
    print(f"     Training time: {time.time()-t1:.2f}s")

    y_pred = classifier.predict(X_test_s)
    y_proba = classifier.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auc':       round(float(roc_auc_score(y_test, y_proba)), 4),
        'mcc':       round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'train_size': int(len(X_train)),
        'test_size':  int(len(X_test)),
        'features':   len(feature_cols),
        'feature_names': feature_cols,
        'data_source': 'visual_dataset.csv (DOM/layout feature benchmark)',
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Image/Visual Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    MCC      : {metrics['mcc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Phishing'])}")

    # Cross-validation
    cv_scores = cross_val_score(classifier, X_train_s, y_train, cv=5, scoring='f1', n_jobs=-1)
    metrics['cv_f1_mean'] = round(float(cv_scores.mean()), 4)
    metrics['cv_f1_std'] = round(float(cv_scores.std()), 4)
    print(f"     CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save
    print(f"\n     Saving image model to {IMAGE_MODEL_DIR}...")
    joblib.dump(classifier, os.path.join(IMAGE_MODEL_DIR, 'classifier.joblib'), compress=3)
    joblib.dump(scaler, os.path.join(IMAGE_MODEL_DIR, 'scaler.joblib'), compress=3)
    with open(os.path.join(IMAGE_MODEL_DIR, 'feature_names.json'), 'w') as fp:
        json.dump(feature_cols, fp)
    with open(os.path.join(IMAGE_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  ✓ Image/Visual model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4:  AUDIO / VISHING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_dataset():
    """
    Load acoustic/vishing feature dataset.
    Features: pitch_variance, speech_rate, silence_ratio, jitter, shimmer,
              background_noise_level, call_center_bg, urgency_keywords,
              robotic_tone, voice_consistency, caller_id_spoof, recording_quality
    """
    path = os.path.join(DATASETS_DIR, 'audio_dataset.csv')
    if not os.path.exists(path):
        print(f"     ⚠ audio_dataset.csv not found — generating synthetic audio data")
        return _generate_audio_data()

    print(f"     Loading audio_dataset.csv...")
    features = []
    labels = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = None
        for row in reader:
            if cols is None:
                cols = [c for c in reader.fieldnames if c != 'label']
            feat = []
            for c in cols:
                try:
                    feat.append(float(row.get(c, 0)))
                except (ValueError, TypeError):
                    feat.append(0.0)
            features.append(feat)
            try:
                labels.append(int(row.get('label', 0)))
            except (ValueError, TypeError):
                labels.append(0)

    print(f"     Audio dataset: {len(features):,} samples, {len(features[0]) if features else 0} features")
    return np.array(features, dtype=np.float32), np.array(labels), cols


def _generate_audio_data():
    """Fallback: generate synthetic acoustic feature data."""
    n = 5000
    cols = ['pitch_variance','speech_rate','silence_ratio','jitter','shimmer',
            'background_noise_level','call_center_bg','urgency_keywords',
            'robotic_tone','voice_consistency','caller_id_spoof','recording_quality']

    features = np.zeros((n, len(cols)), dtype=np.float32)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        is_vish = random.random() < 0.5
        labels[i] = 1 if is_vish else 0
        if is_vish:
            features[i] = [
                random.uniform(0.01, 0.15),   # pitch_variance (low, robotic)
                random.uniform(3.5, 6.0),      # speech_rate (fast, urgent)
                random.uniform(0.05, 0.25),    # silence_ratio
                random.uniform(0.02, 0.08),    # jitter (high)
                random.uniform(0.03, 0.12),    # shimmer (high)
                random.uniform(0.3, 0.8),      # background_noise
                random.choice([0, 1]) if random.random() > 0.4 else 0,  # call_center_bg
                random.randint(1, 6),          # urgency_keywords
                random.uniform(0.3, 0.9),      # robotic_tone
                random.uniform(0.2, 0.6),      # voice_consistency
                random.choice([0, 1]) if random.random() > 0.4 else 0,  # caller_id_spoof
                random.uniform(0.2, 0.5),      # recording_quality (low)
            ]
        else:
            features[i] = [
                random.uniform(0.08, 0.35),    # pitch_variance (natural)
                random.uniform(2.0, 4.0),      # speech_rate (normal)
                random.uniform(0.1, 0.4),      # silence_ratio
                random.uniform(0.005, 0.025),  # jitter (low)
                random.uniform(0.01, 0.04),    # shimmer (low)
                random.uniform(0.05, 0.3),     # background_noise (low)
                0,                              # call_center_bg
                0,                              # urgency_keywords
                random.uniform(0.0, 0.2),      # robotic_tone (low)
                random.uniform(0.6, 0.95),     # voice_consistency (high)
                0,                              # caller_id_spoof
                random.uniform(0.6, 0.95),     # recording_quality (high)
            ]

    print(f"     Generated {n:,} synthetic audio samples, {len(cols)} features")
    return features, labels, cols


def train_audio_model():
    print("\n" + "="*70)
    print("  AUDIO / VISHING MODEL TRAINING")
    print("="*70)

    result = load_audio_dataset()
    X, y, feature_cols = result

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifier = LogisticRegression(
        C=1.0, max_iter=500, class_weight='balanced',
        solver='lbfgs', random_state=RANDOM_STATE
    )
    t1 = time.time()
    classifier.fit(X_train_s, y_train)
    print(f"     Training time: {time.time()-t1:.2f}s")

    y_pred = classifier.predict(X_test_s)
    y_proba = classifier.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auc':       round(float(roc_auc_score(y_test, y_proba)), 4),
        'mcc':       round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'train_size': int(len(X_train)),
        'test_size':  int(len(X_test)),
        'features':   len(feature_cols),
        'feature_names': feature_cols,
        'data_source': 'audio_dataset.csv (acoustic/vishing feature benchmark)',
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Audio/Vishing Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    MCC      : {metrics['mcc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Vishing'])}")

    # Cross-validation
    cv_scores = cross_val_score(classifier, X_train_s, y_train, cv=5, scoring='f1', n_jobs=-1)
    metrics['cv_f1_mean'] = round(float(cv_scores.mean()), 4)
    metrics['cv_f1_std'] = round(float(cv_scores.std()), 4)
    print(f"     CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save
    print(f"\n     Saving audio model to {AUDIO_MODEL_DIR}...")
    joblib.dump(classifier, os.path.join(AUDIO_MODEL_DIR, 'classifier.joblib'), compress=3)
    joblib.dump(scaler, os.path.join(AUDIO_MODEL_DIR, 'scaler.joblib'), compress=3)
    with open(os.path.join(AUDIO_MODEL_DIR, 'feature_names.json'), 'w') as fp:
        json.dump(feature_cols, fp)
    with open(os.path.join(AUDIO_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  ✓ Audio/Vishing model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5:  VIDEO / DEEPFAKE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_video_dataset():
    """
    Load video/deepfake forensic feature dataset.
    Features: laplacian_var, fft_noise, face_warp_score, blink_rate,
              lip_sync_error, compression_artifacts, temporal_consistency, skin_texture_anomaly
    """
    path = os.path.join(DATASETS_DIR, 'video_dataset.csv')
    if not os.path.exists(path):
        print(f"     ⚠ video_dataset.csv not found — generating synthetic video data")
        return _generate_video_data()

    print(f"     Loading video_dataset.csv...")
    features = []
    labels = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        cols = None
        for row in reader:
            if cols is None:
                cols = [c for c in reader.fieldnames if c != 'label']
            feat = []
            for c in cols:
                try:
                    feat.append(float(row.get(c, 0)))
                except (ValueError, TypeError):
                    feat.append(0.0)
            features.append(feat)
            try:
                labels.append(int(row.get('label', 0)))
            except (ValueError, TypeError):
                labels.append(0)

    print(f"     Video dataset: {len(features):,} samples, {len(features[0]) if features else 0} features")
    return np.array(features, dtype=np.float32), np.array(labels), cols


def _generate_video_data():
    """Fallback: generate synthetic deepfake forensic feature data."""
    n = 5000
    cols = ['laplacian_var','fft_noise','face_warp_score','blink_rate',
            'lip_sync_error','compression_artifacts','temporal_consistency','skin_texture_anomaly']

    features = np.zeros((n, len(cols)), dtype=np.float32)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        is_fake = random.random() < 0.5
        labels[i] = 1 if is_fake else 0
        if is_fake:
            features[i] = [
                random.uniform(5, 50),          # laplacian_var (low = blurry deepfake)
                random.uniform(0.3, 0.8),       # fft_noise (high)
                random.uniform(0.3, 0.9),       # face_warp_score (high)
                random.uniform(0.0, 0.1),       # blink_rate (abnormally low)
                random.uniform(0.2, 0.7),       # lip_sync_error (high)
                random.uniform(0.3, 0.8),       # compression_artifacts
                random.uniform(0.2, 0.6),       # temporal_consistency (low)
                random.uniform(0.3, 0.8),       # skin_texture_anomaly (high)
            ]
        else:
            features[i] = [
                random.uniform(40, 200),        # laplacian_var (high = sharp)
                random.uniform(0.05, 0.3),      # fft_noise (low)
                random.uniform(0.0, 0.15),      # face_warp_score (low)
                random.uniform(0.15, 0.4),      # blink_rate (normal)
                random.uniform(0.0, 0.1),       # lip_sync_error (low)
                random.uniform(0.05, 0.2),      # compression_artifacts (low)
                random.uniform(0.7, 0.98),      # temporal_consistency (high)
                random.uniform(0.0, 0.15),      # skin_texture_anomaly (low)
            ]

    print(f"     Generated {n:,} synthetic video samples, {len(cols)} features")
    return features, labels, cols


def train_video_model():
    print("\n" + "="*70)
    print("  VIDEO / DEEPFAKE MODEL TRAINING")
    print("="*70)

    result = load_video_dataset()
    X, y, feature_cols = result

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"     Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifier = LogisticRegression(
        C=1.0, max_iter=500, class_weight='balanced',
        solver='lbfgs', random_state=RANDOM_STATE
    )
    t1 = time.time()
    classifier.fit(X_train_s, y_train)
    print(f"     Training time: {time.time()-t1:.2f}s")

    y_pred = classifier.predict(X_test_s)
    y_proba = classifier.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auc':       round(float(roc_auc_score(y_test, y_proba)), 4),
        'mcc':       round(float(matthews_corrcoef(y_test, y_pred)), 4),
        'train_size': int(len(X_train)),
        'test_size':  int(len(X_test)),
        'features':   len(feature_cols),
        'feature_names': feature_cols,
        'data_source': 'video_dataset.csv (deepfake forensic feature benchmark)',
    }

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Video/Deepfake Model Results:")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    F1-Score : {metrics['f1']:.4f}")
    print(f"    AUC-ROC  : {metrics['auc']:.4f}")
    print(f"    MCC      : {metrics['mcc']:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Real','Deepfake'])}")

    # Cross-validation
    cv_scores = cross_val_score(classifier, X_train_s, y_train, cv=5, scoring='f1', n_jobs=-1)
    metrics['cv_f1_mean'] = round(float(cv_scores.mean()), 4)
    metrics['cv_f1_std'] = round(float(cv_scores.std()), 4)
    print(f"     CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save
    print(f"\n     Saving video model to {VIDEO_MODEL_DIR}...")
    joblib.dump(classifier, os.path.join(VIDEO_MODEL_DIR, 'classifier.joblib'), compress=3)
    joblib.dump(scaler, os.path.join(VIDEO_MODEL_DIR, 'scaler.joblib'), compress=3)
    with open(os.path.join(VIDEO_MODEL_DIR, 'feature_names.json'), 'w') as fp:
        json.dump(feature_cols, fp)
    with open(os.path.join(VIDEO_MODEL_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)

    print("  ✓ Video/Deepfake model saved successfully!")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def save_combined_metrics(results):
    """Save a combined metrics.json with all model results."""
    combined = {
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_version': '2.0',
        'models': {}
    }
    for name, metrics in results.items():
        if metrics is not None:
            combined['models'][name] = metrics

    path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(path, 'w') as fp:
        json.dump(combined, fp, indent=2)
    print(f"\n  Combined metrics saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='PhishGuard Multi-Modal Model Trainer v2.0')
    parser.add_argument('--url',   action='store_true', help='Train URL model only')
    parser.add_argument('--text',  action='store_true', help='Train text model only')
    parser.add_argument('--image', action='store_true', help='Train image/visual model only')
    parser.add_argument('--audio', action='store_true', help='Train audio/vishing model only')
    parser.add_argument('--video', action='store_true', help='Train video/deepfake model only')
    args = parser.parse_args()

    any_specific = args.url or args.text or args.image or args.audio or args.video
    train_all = not any_specific

    total_start = time.time()
    results = {}

    if args.url or train_all:
        results['url'] = train_url_model()

    if args.text or train_all:
        results['text'] = train_text_model()

    if args.image or train_all:
        results['image'] = train_image_model()

    if args.audio or train_all:
        results['audio'] = train_audio_model()

    if args.video or train_all:
        results['video'] = train_video_model()

    # Save combined metrics
    save_combined_metrics(results)

    elapsed = time.time() - total_start
    print("\n" + "="*70)
    print("  TRAINING COMPLETE — ALL MODELS")
    print("="*70)

    headers = f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'MCC':>10}"
    print(headers)
    print("  " + "-"*68)
    for name, metrics in results.items():
        if metrics:
            print(f"  {name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc']:>10.4f} "
                  f"{metrics['mcc']:>10.4f}")
    print("  " + "-"*68)
    print(f"\n  Total training time: {elapsed:.1f}s")
    print(f"  Models saved to: {MODELS_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
