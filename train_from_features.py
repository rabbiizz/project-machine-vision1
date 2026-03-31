# ============================================================
#  TRAIN MODEL v4 — FULL FIX (No Bias + Stable + Production)
# ============================================================

import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH  = "food_compare.keras"
SCALER_PATH = "scaler.pkl"

BATCH_SIZE = 32
EPOCHS     = 300
LR         = 2e-4   # 🔥 ลด LR กัน bias

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# LOAD FEATURES
# ============================================================

print("\n" + "="*50)
print("LOADING FEATURES")
print("="*50)

F1 = np.load("features_out/F1.npy")
F2 = np.load("features_out/F2.npy")
Y  = np.load("features_out/Y.npy")

print(f"F1={F1.shape}, F2={F2.shape}, Y={Y.shape}")
print(f"Label dist: 1={int(Y.sum())}  0={int((1-Y).sum())}")

FEAT_DIM = F1.shape[1]

# ============================================================
# NORMALIZE FEATURES
# ============================================================

print("\n[INFO] Normalizing features...")

scaler = StandardScaler()
scaler.fit(np.vstack([F1, F2]))

joblib.dump(scaler, SCALER_PATH)
print(f"[INFO] Saved scaler → {SCALER_PATH}")

F1 = scaler.transform(F1)
F2 = scaler.transform(F2)

# ============================================================
# SWAP AUGMENTATION (ทำให้ balanced)
# ============================================================

print("\n[INFO] Applying swap augmentation...")

F1_aug = np.concatenate([F1, F2])
F2_aug = np.concatenate([F2, F1])
Y_aug  = np.concatenate([Y, 1 - Y])

print(f"Total pairs after aug: {len(F1_aug)}")
print(f"Balanced labels → 1={int(Y_aug.sum())}, 0={int((1-Y_aug).sum())}")

# ============================================================
# SPLIT DATA
# ============================================================

F1_tmp, F1_test, F2_tmp, F2_test, y_tmp, y_test = train_test_split(
    F1_aug, F2_aug, Y_aug,
    test_size=0.15,
    stratify=Y_aug,
    random_state=42
)

F1_train, F1_val, F2_train, F2_val, y_train, y_val = train_test_split(
    F1_tmp, F2_tmp, y_tmp,
    test_size=0.15,
    stratify=y_tmp,
    random_state=42
)

print(f"Train={len(F1_train)} | Val={len(F1_val)} | Test={len(F1_test)}")

# ============================================================
# CLASS WEIGHT (🔥 แก้ bias จริง)
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}

print(f"\n[INFO] Class weight: {class_weight_dict}")

# ============================================================
# BUILD SCORING NETWORK
# ============================================================

def build_scoring_network(feat_dim):

    inp = Input(shape=(feat_dim,))

    # 🔥 noise กัน bias
    x = GaussianNoise(0.05)(inp)

    x = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation="relu")(x)

    score = Dense(1)(x)

    return Model(inp, score, name="scoring_net")

print("\n[INFO] Building model...")

scoring_net = build_scoring_network(FEAT_DIM)

# ============================================================
# SIAMESE MODEL (🔥 FIX BIAS สำคัญมาก)
# ============================================================

img1 = Input(shape=(FEAT_DIM,))
img2 = Input(shape=(FEAT_DIM,))

s1 = scoring_net(img1)
s2 = scoring_net(img2)

# 🔥 CENTERING trick (กัน bias)
mean_score = Average()([s1, s2])
s1 = Subtract()([s1, mean_score])
s2 = Subtract()([s2, mean_score])

diff = Subtract()([s1, s2])
out  = Activation("sigmoid")(diff)

model = Model([img1, img2], out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "="*50)
print("TRAINING")
print("="*50)

history = model.fit(
    [F1_train, F2_train], y_train,
    validation_data=([F1_val, F2_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight_dict,   # 🔥 สำคัญสุด
    verbose=1
)

# ============================================================
# EVALUATION
# ============================================================

print("\n" + "="*50)
print("EVALUATION")
print("="*50)

best_model = tf.keras.models.load_model(MODEL_PATH)

y_prob = best_model.predict([F1_test, F2_test], verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy : {acc:.4f}")
print(f"AUC      : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nSaved model : {MODEL_PATH}")
print(f"Saved scaler: {SCALER_PATH}")