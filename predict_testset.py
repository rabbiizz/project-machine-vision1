# ============================================================
#  PREDICT v4 — FINAL (MATCH TRAIN v4 + NO BIAS + ROBUST)
# ============================================================

import os
import numpy as np
import pandas as pd
import cv2
import torch
import clip
from PIL import Image

from tensorflow.keras.models import load_model
import joblib

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH   = "food_compare.keras"
SCALER_PATH  = "scaler.pkl"

TEST_CSV     = r"D:\project machine\project machine vision\Dataset_for_development\data_from_intragram.csv"
IMAGE_ROOT   = r"D:\project machine\project machine vision\Dataset_for_development\Instagram Photos"

OUTPUT_CSV   = "predicted.csv"
BATCH_SIZE   = 64
FEAT_DIM     = 777

# ============================================================
# DEVICE + CLIP
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

print("[INFO] Loading CLIP...")
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()
print("[INFO] CLIP Ready\n")

# ============================================================
# BUILD INDEX (lowercase key)
# ============================================================

def build_index(root):
    idx = {}
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[f.lower()] = os.path.join(root_dir, f)
    print(f"[INFO] Indexed {len(idx)} images")
    return idx

IMG_INDEX = build_index(IMAGE_ROOT)

# ============================================================
# FIND IMAGE (robust)
# ============================================================

def find_img(name):
    name = str(name).strip().lower()

    if not name.endswith((".jpg", ".jpeg", ".png")):
        name += ".jpg"

    if name in IMG_INDEX:
        return IMG_INDEX[name]

    name_no_ext = os.path.splitext(name)[0]

    for k in IMG_INDEX:
        k_no_ext = os.path.splitext(k)[0]

        if name_no_ext == k_no_ext:
            return IMG_INDEX[k]

        if name_no_ext in k:
            return IMG_INDEX[k]

    return None

# ============================================================
# FEATURE EXTRACT (🔥 ตรงกับ TRAIN)
# ============================================================

def clip_feature(path):
    try:
        with torch.no_grad():
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            vec = clip_model.encode_image(img).cpu().numpy()[0]

        return vec.astype(np.float32) / (np.linalg.norm(vec) + 1e-8)

    except:
        print(f"[WARN] CLIP fail: {path}")
        return None


def aesthetic_feature(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = gray.shape

        # 1 Brightness
        brightness = gray.mean() / 255.0

        # 2 Contrast
        contrast = gray.std() / 255.0

        # 3 Colorfulness (เหมือน extract)
        R = rgb[:, :, 0].astype(float)
        G = rgb[:, :, 1].astype(float)
        B = rgb[:, :, 2].astype(float)
        rg = R - G
        yb = 0.5 * (R + G) - B
        colorfulness = (
            np.sqrt(np.std(rg)**2 + np.std(yb)**2) +
            0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
        ) / 255.0

        # 4 Sharpness
        sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0, 1.0)

        # 5 Saturation
        saturation = hsv[:, :, 1].mean() / 255.0

        # 6 Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)

        # 7 Symmetry
        left  = gray[:, :w//2].astype(float)
        right = np.fliplr(gray[:, w//2:w//2*2]).astype(float)

        min_w = min(left.shape[1], right.shape[1])
        symmetry = 1 - np.mean(
            np.abs(left[:, :min_w] - right[:, :min_w])
        ) / 255.0

        # 8 Food area
        mask1 = cv2.inRange(hsv, np.array([0,50,50]),   np.array([30,255,255]))
        mask2 = cv2.inRange(hsv, np.array([160,50,50]), np.array([180,255,255]))
        food_ratio = np.sum((mask1 + mask2) > 0) / (h * w)

        # 9 Background blur
        center_mask = np.zeros_like(gray)
        cv2.ellipse(center_mask, (w//2, h//2), (w//4, h//4), 0, 0, 360, 255, -1)

        bg_mask = cv2.bitwise_not(center_mask)
        bg_region = cv2.bitwise_and(gray, gray, mask=bg_mask)

        blur_bg = 1 - min(
            cv2.Laplacian(bg_region, cv2.CV_64F).var() / 1000.0,
            1.0
        )

        feat = np.array([
            brightness, contrast, colorfulness,
            sharpness, saturation, edge_density,
            symmetry, food_ratio, blur_bg
        ], dtype=np.float32)

        return np.clip(feat, 0.0, 1.0)

    except:
        print(f"[WARN] Aesthetic fail: {path}")
        return None


def full_feature(path):
    clip_f = clip_feature(path)
    aest_f = aesthetic_feature(path)

    if clip_f is None or aest_f is None:
        return None

    feat = np.concatenate([clip_f, aest_f]).astype(np.float32)

    if feat.shape[0] != FEAT_DIM:
        return None

    return feat

# ============================================================
# CACHE
# ============================================================

feature_cache = {}

def get_feat(name):
    name = str(name).strip()

    if name in feature_cache:
        return feature_cache[name]

    path = find_img(name)

    if path is None:
        print(f"[WARN] Missing image: {name}")
        feat = None
    else:
        feat = full_feature(path)

    feature_cache[name] = feat
    return feat

# ============================================================
# LOAD MODEL + SCALER
# ============================================================

print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

print("[INFO] Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("[INFO] Ready\n")

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(TEST_CSV)
df.columns = df.columns.str.strip()

print(f"[INFO] Total pairs: {len(df)}")

# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

print("[INFO] Extracting features...")

F1_list, F2_list, valid_idx = [], [], []

for i, row in df.iterrows():

    f1 = get_feat(row["Image 1"])
    f2 = get_feat(row["Image 2"])

    if f1 is None or f2 is None:
        continue

    F1_list.append(f1)
    F2_list.append(f2)
    valid_idx.append(i)

F1 = np.array(F1_list, dtype=np.float32)
F2 = np.array(F2_list, dtype=np.float32)

print(f"[INFO] Valid pairs: {len(F1)} / {len(df)}")

# ============================================================
# NORMALIZE (เหมือน TRAIN)
# ============================================================

F1 = scaler.transform(F1)
F2 = scaler.transform(F2)

# ============================================================
# PREDICT
# ============================================================

print("[INFO] Predicting...")

probs = model.predict([F1, F2], batch_size=BATCH_SIZE).flatten()

# 🔥 debug distribution
print(f"[DEBUG] prob min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")

labels = (probs >= 0.5).astype(int)
winners = np.where(labels == 1, 1, 2)

# ============================================================
# PUT BACK TO DF
# ============================================================

df["Winner"] = np.nan
df["prob_img1"] = np.nan
df["prob_img2"] = np.nan
df["confidence"] = np.nan

for idx, w, p in zip(valid_idx, winners, probs):
    df.loc[idx, "Winner"] = w
    df.loc[idx, "prob_img1"] = p
    df.loc[idx, "prob_img2"] = 1 - p
    df.loc[idx, "confidence"] = max(p, 1 - p)

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n[SAVED] {OUTPUT_CSV}")

# ============================================================
# SUMMARY
# ============================================================

valid = df["Winner"].dropna().astype(int)

print("\n" + "="*50)
print("SUMMARY")
print("="*50)

print(f"Valid pairs : {len(valid)}")
print(f"Image1 wins : {(valid == 1).sum()}")
print(f"Image2 wins : {(valid == 2).sum()}")
print(f"Avg conf    : {df['confidence'].mean():.4f}")