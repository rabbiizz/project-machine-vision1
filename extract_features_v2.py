import os
import numpy as np
import pandas as pd
import cv2
import torch
import clip
from PIL import Image

# =============================
# CONFIG
# =============================

DATASETS = [
    {
        "csv": r"D:\project machine\project machine vision\Dataset_for_development\data_from_questionaire.csv",
        "img_dir": r"D:\project machine\project machine vision\Dataset_for_development\Questionair Images"
    },
    {
        "csv": r"D:\project machine\project machine vision\Dataset_for_development\data_from_intragram.csv",
        "img_dir": r"D:\project machine\project machine vision\Dataset_for_development\Instagram Photos"
    }
]

OUT_DIR = "features_out"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================
# FEATURE EXTRACTOR
# =============================

class FeatureExtractor:

    def __init__(self):
        print(f"[INFO] Device: {device}")
        print("[INFO] Loading CLIP ViT-L/14...")
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.model.eval()
        print("[INFO] CLIP Ready\n")

        self.cache = {}

    # ---------- IMAGE INDEX ----------
    def build_index(self, root):
        index = {}
        for root_dir, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    index[f.lower()] = os.path.join(root_dir, f)
        print(f"[INFO] images: {len(index)}")
        return index

    # ---------- FIND IMAGE ----------
    def find(self, name, index):
        name = str(name).strip().lower()

        if not name.endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"

        if name in index:
            return index[name]

        name_no_ext = os.path.splitext(name)[0]

        for k in index:
            k_no_ext = os.path.splitext(k)[0]

            if name_no_ext == k_no_ext:
                return index[k]

            if name_no_ext in k:
                return index[k]

        return None

    # ---------- CLIP FEATURE ----------
    def clip_feat(self, path):
        try:
            with torch.no_grad():
                img = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                vec = self.model.encode_image(img).cpu().numpy()[0]

            return vec / (np.linalg.norm(vec) + 1e-8)

        except:
            print(f"[CLIP ERROR] {path}")
            return np.zeros(768, dtype=np.float32)

    # ---------- AESTHETIC FEATURE ----------
    def aesthetic_feat(self, path):
        try:
            img = cv2.imread(path)
            if img is None:
                return np.zeros(9, dtype=np.float32)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = gray.shape

            brightness = gray.mean() / 255.0
            contrast = gray.std() / 255.0

            R = rgb[:, :, 0].astype(float)
            G = rgb[:, :, 1].astype(float)
            B = rgb[:, :, 2].astype(float)

            rg = R - G
            yb = 0.5 * (R + G) - B

            colorfulness = (
                np.sqrt(np.std(rg)**2 + np.std(yb)**2) +
                0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
            ) / 255.0

            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0, 1.0)

            saturation = hsv[:, :, 1].mean() / 255.0

            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)

            left  = gray[:, :w//2].astype(float)
            right = np.fliplr(gray[:, w//2:w//2*2]).astype(float)

            min_w = min(left.shape[1], right.shape[1])
            symmetry = 1 - np.mean(
                np.abs(left[:, :min_w] - right[:, :min_w])
            ) / 255.0

            mask1 = cv2.inRange(hsv, np.array([0,50,50]),   np.array([30,255,255]))
            mask2 = cv2.inRange(hsv, np.array([160,50,50]), np.array([180,255,255]))
            food_ratio = np.sum((mask1 + mask2) > 0) / (h * w)

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

            feat = np.clip(feat, 0.0, 1.0)
            feat = np.nan_to_num(feat)

            return feat

        except:
            print(f"[AESTHETIC ERROR] {path}")
            return np.zeros(9, dtype=np.float32)

    # ---------- FULL FEATURE ----------
    def extract(self, path):
        clip_f = self.clip_feat(path)
        aest_f = self.aesthetic_feat(path)

        # 🔥 FIX scale imbalance
        aest_f = aest_f * 3.0

        feat = np.concatenate([clip_f, aest_f])

        if feat.shape[0] != 777:
            print(f"[ERROR] Feature dim != 777: {path}")
            return np.zeros(777, dtype=np.float32)

        return feat

    # ---------- CACHE ----------
    def get(self, filename, index):
        key = str(filename).strip().lower()

        if key in self.cache:
            return self.cache[key]

        path = self.find(filename, index)

        if path is None:
            print(f"[WARNING] Missing: {filename}")
            feat = np.zeros(777, dtype=np.float32)
        else:
            feat = self.extract(path)

        self.cache[key] = feat
        return feat


# =============================
# MAIN
# =============================

extractor = FeatureExtractor()

F1_list, F2_list, Y_list = [], [], []

for data in DATASETS:

    print(f"\n[DATASET] {data['csv']}")

    df = pd.read_csv(data["csv"])
    df.columns = df.columns.str.strip()

    index = extractor.build_index(data["img_dir"])

    for i, row in df.iterrows():

        try:
            y = 1 if int(row["Winner"]) == 1 else 0
        except:
            continue

        f1 = extractor.get(row["Image 1"], index)
        f2 = extractor.get(row["Image 2"], index)

        # 🔥 FIX DATA BIAS
        if np.random.rand() < 0.5:
            f1, f2 = f2, f1
            y = 1 - y

        F1_list.append(f1)
        F2_list.append(f2)
        Y_list.append(y)

        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(df)}  (cache={len(extractor.cache)})")


# =============================
# SAVE
# =============================

F1 = np.array(F1_list, dtype=np.float32)
F2 = np.array(F2_list, dtype=np.float32)
Y  = np.array(Y_list, dtype=np.float32)

np.save(os.path.join(OUT_DIR, "F1.npy"), F1)
np.save(os.path.join(OUT_DIR, "F2.npy"), F2)
np.save(os.path.join(OUT_DIR, "Y.npy"),  Y)

print("\n[SUCCESS]")
print("F1:", F1.shape)
print("F2:", F2.shape)
print("Y :", Y.shape)
print(f"Feature dim: {F1.shape[1]} (should be 777)")
print(f"Label dist: 1={int(Y.sum())} 0={int((1-Y).sum())}")

print("NaN check:", np.isnan(F1).sum(), np.isnan(F2).sum())