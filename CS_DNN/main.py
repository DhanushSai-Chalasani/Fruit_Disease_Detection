"""
main.py — Fruit Quality Classification via Compressed Sensing + Deep Learning
==============================================================================
Uses PyTorch (compatible with Python 3.14).

Pipeline (Option A — sequential training):
  1. Load & preprocess dataset (128x128, normalised 0-1)
  2. Compressed Sensing  : downsample 128->32 (simulate CS measurements)
  3. Autoencoder         : reconstruct 128x128 from 32x32 compressed input
  4. CNN Classifier      : classify reconstructed image -> Normal / Damaged
  5. Evaluation          : MSE, PSNR (reconstruction) | Accuracy, Confusion Matrix (classification)
  6. Gradio UI           : upload / webcam -> full pipeline -> result

Run:
  python prepare_dataset.py   <- once, to build dataset/ folder
  python main.py              <- train + launch Gradio UI
"""

import os, math, json
import numpy as np
import cv2
import gradio as gr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE     = 128
COMPRESSED   = 32
CHANNELS     = 3
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
AE_EPOCHS    = 20
CLS_EPOCHS   = 20
BATCH_SIZE   = 16
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── STEP 1-2: Load & Preprocess Dataset ──────────────────────────────────────

_fruit_vocab = []

def build_vocab():
    global _fruit_vocab
    vocab = set()
    for split in ["train", "test"]:
        base = os.path.join(DATASET_PATH, split)
        if not os.path.exists(base): continue
        for cls_name in ["normal", "damaged"]:
            cls_dir = os.path.join(base, cls_name)
            if not os.path.isdir(cls_dir): continue
            for fname in os.listdir(cls_dir):
                parts = fname.split("_")
                if len(parts) >= 3:
                    vocab.add(parts[0])
                else:
                    vocab.add("unknown")
    _fruit_vocab = sorted(list(vocab))
    if not _fruit_vocab:
        _fruit_vocab = ["unknown"]

def load_dataset(split="train"):
    """Return (images, labels, fruit_labels) as float32 numpy arrays."""
    images, labels, fruit_labels = [], [], []
    label_map = {"normal": 0, "damaged": 1}
    base = os.path.join(DATASET_PATH, split)

    if not os.path.exists(base):
        raise FileNotFoundError(
            f"Dataset folder not found: {base}\n"
            "Please run  python prepare_dataset.py  first."
        )

    for cls_name, cls_idx in label_map.items():
        cls_dir = os.path.join(base, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            
            parts = fname.split("_")
            fruit_str = parts[0] if len(parts) >= 3 else "unknown"
            f_idx = _fruit_vocab.index(fruit_str) if fruit_str in _fruit_vocab else 0
            
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img.astype(np.float32) / 255.0)
            labels.append(cls_idx)
            fruit_labels.append(f_idx)

    return np.array(images), np.array(labels, dtype=np.int64), np.array(fruit_labels, dtype=np.int64)


def to_tensor(images_nhwc, labels=None):
    """Convert (N,H,W,C) numpy -> (N,C,H,W) torch tensor."""
    t = torch.from_numpy(images_nhwc).permute(0, 3, 1, 2)
    if labels is not None:
        l = torch.from_numpy(labels)
        return t, l
    return t


# ── STEP 3: Compressed Sensing (downsampling simulation) ─────────────────────

def compress_batch(images_nhwc):
    """Downsample (N,128,128,3) -> (N,32,32,3). Simulates CS measurements."""
    out = np.zeros((len(images_nhwc), COMPRESSED, COMPRESSED, CHANNELS), dtype=np.float32)
    for i, img in enumerate(images_nhwc):
        small = cv2.resize(
            (img * 255).astype(np.uint8),
            (COMPRESSED, COMPRESSED),
            interpolation=cv2.INTER_AREA
        )
        out[i] = small.astype(np.float32) / 255.0
    return out


# ── STEP 5: Autoencoder (Encoder-Decoder) ────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Encoder: 32x32 -> 16x16 -> 8x8
    Decoder: 8x8  -> 32x32 -> 64x64 -> 128x128
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                            # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                            # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(64,  32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),  # 32x32
            nn.ConvTranspose2d(32,  16, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.ReLU(),  # 64x64
            nn.ConvTranspose2d(16,   8, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(8), nn.ReLU(),   # 128x128
            nn.Conv2d(8, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# ── STEP 6: CNN Classifier ────────────────────────────────────────────────────

class FruitClassifier(nn.Module):
    """Multi-task CNN for Condition and Fruit Type."""
    def __init__(self, num_fruits=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 64x64
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        
        # Condition head (Normal vs Damaged)
        self.cond_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        
        # Fruit type head (Apple, Banana, etc.)
        self.fruit_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_fruits)
        )

    def forward(self, x):
        f = self.flat(self.pool(self.features(x)))
        cond = self.cond_head(f)
        fruit = self.fruit_head(f)
        return cond, fruit


# ── STEP 7: Training helpers ──────────────────────────────────────────────────

def train_autoencoder(ae, loader, epochs):
    ae.to(DEVICE)
    criterion = nn.MSELoss()
    opt = optim.Adam(ae.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        ae.train()
        running = 0.0
        for xc, xo in loader:
            xc, xo = xc.to(DEVICE), xo.to(DEVICE)
            opt.zero_grad()
            out = ae(xc)
            loss = criterion(out, xo)
            loss.backward()
            opt.step()
            running += loss.item() * xc.size(0)
        sched.step()
        print(f"  AE Epoch {ep:02d}/{epochs}  MSE Loss: {running/len(loader.dataset):.6f}")


def train_classifier(cls, loader, epochs):
    cls.to(DEVICE)
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(cls.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    # Data Augmentation
    aug = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    for ep in range(1, epochs + 1):
        cls.train()
        running, correct_cond, correct_fruit = 0.0, 0, 0
        for xr, y_cond, y_fruit in loader:
            xr = aug(xr)
            xr, y_cond, y_fruit = xr.to(DEVICE), y_cond.float().unsqueeze(1).to(DEVICE), y_fruit.to(DEVICE)
            
            opt.zero_grad()
            out_cond, out_fruit = cls(xr)
            
            loss_cond = bce(out_cond, y_cond)
            loss_fruit = ce(out_fruit, y_fruit)
            loss = loss_cond + loss_fruit
            
            loss.backward()
            opt.step()
            
            running += loss.item() * xr.size(0)
            correct_cond += ((out_cond >= 0.5).long() == y_cond.long()).sum().item()
            correct_fruit += (out_fruit.argmax(dim=1) == y_fruit).sum().item()
            
        sched.step()
        acc_cond = correct_cond / len(loader.dataset) * 100
        acc_fruit = correct_fruit / len(loader.dataset) * 100
        print(f"  CLS Ep {ep:02d}/{epochs} Loss: {running/len(loader.dataset):.4f} "
              f"Cond Acc: {acc_cond:.1f}% Fruit Acc: {acc_fruit:.1f}%")


@torch.no_grad()
def predict_batch(model, tensor, batch=32):
    model.eval()
    model.to(DEVICE)
    preds = []
    for i in range(0, len(tensor), batch):
        out = model(tensor[i:i+batch].to(DEVICE))
        preds.append(out.cpu())
    return torch.cat(preds, 0)

@torch.no_grad()
def predict_batch_cls(model, tensor, batch=32):
    model.eval()
    model.to(DEVICE)
    preds_cond, preds_fruit = [], []
    for i in range(0, len(tensor), batch):
        cond, fruit = model(tensor[i:i+batch].to(DEVICE))
        preds_cond.append(cond.cpu())
        preds_fruit.append(fruit.cpu())
    return torch.cat(preds_cond, 0), torch.cat(preds_fruit, 0)


# ── STEP 8: Full training + evaluation ───────────────────────────────────────

def run_pipeline():
    print("\n=== Scanning Dataset for Fruit Vocabulary ===")
    build_vocab()
    print(f"  Found fruit types: {_fruit_vocab}")

    print("\n=== Loading dataset ===")
    X_tr, y_tr, f_tr = load_dataset("train")
    X_te, y_te, f_te = load_dataset("test")
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # Compressed sensing
    print("\n=== Simulating Compressed Sensing ===")
    X_tr_c = compress_batch(X_tr)
    X_te_c  = compress_batch(X_te)
    print(f"Compressed shape: {X_tr_c.shape}")

    # Convert to torch
    Tc_tr, To_tr = to_tensor(X_tr_c), to_tensor(X_tr)
    Tc_te, To_te = to_tensor(X_te_c),  to_tensor(X_te)

    ae_loader = DataLoader(TensorDataset(Tc_tr, To_tr), batch_size=BATCH_SIZE, shuffle=True)

    # Train Autoencoder
    print("\n=== Training Autoencoder (Step 5) ===")
    ae = Autoencoder()
    train_autoencoder(ae, ae_loader, AE_EPOCHS)

    # Reconstruct all sets
    print("\n=== Reconstructing images ===")
    Tr_tr = predict_batch(ae, Tc_tr)   # reconstructed train
    Tr_te  = predict_batch(ae, Tc_te)   # reconstructed test

    # Reconstruction evaluation
    mse  = float(nn.MSELoss()(Tr_te, To_te).item())
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"\n=== Reconstruction Evaluation ===")
    print(f"  MSE  : {mse:.6f}")
    print(f"  PSNR : {psnr:.2f} dB")

    # Train Classifier
    print("\n=== Training CNN Classifier (Step 6-7) ===")
    y_tr_t = torch.from_numpy(y_tr)
    f_tr_t = torch.from_numpy(f_tr)
    cls_loader = DataLoader(TensorDataset(Tr_tr, y_tr_t, f_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    cls = FruitClassifier(num_fruits=len(_fruit_vocab))
    train_classifier(cls, cls_loader, CLS_EPOCHS)

    # Classification evaluation
    print("\n=== Classification Evaluation ===")
    prob_cond, out_fruit  = predict_batch_cls(cls, Tr_te)
    prob_cond = prob_cond.squeeze(1).numpy()
    y_pred = (prob_cond >= 0.5).astype(int)
    f_pred = out_fruit.argmax(dim=1).numpy()
    
    acc   = accuracy_score(y_te, y_pred)
    f_acc = accuracy_score(f_te, f_pred)
    cm    = confusion_matrix(y_te, y_pred)
    
    print(f"  Test Accuracy (Condition): {acc * 100:.2f}%")
    print(f"  Test Accuracy (Fruit): {f_acc * 100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Damaged"])
    ax.set_yticklabels(["Normal", "Damaged"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"  Saved confusion matrix -> {cm_path}")

    return ae, cls, mse, psnr, acc, f_acc, cm


# ── Global state ──────────────────────────────────────────────────────────────
_ae   = None
_cls  = None
_info = {}


def init():
    global _ae, _cls, _info, _fruit_vocab
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    ae_path = os.path.join(MODELS_DIR, "ae_weights.pth")
    cls_path = os.path.join(MODELS_DIR, "cls_weights.pth")
    info_path = os.path.join(MODELS_DIR, "info.json")

    # Offline Bypass Mode
    if os.path.exists(ae_path) and os.path.exists(cls_path) and os.path.exists(info_path):
        print("\n=== Loading existing models from disk (Skipping Training) ===")
        with open(info_path, "r") as f:
            _info = json.load(f)
        
        _fruit_vocab = _info.get("vocab", ["unknown"])
        
        _ae = Autoencoder().to(DEVICE)
        _ae.load_state_dict(torch.load(ae_path, map_location=DEVICE))
        _ae.eval()

        _cls = FruitClassifier(num_fruits=len(_fruit_vocab)).to(DEVICE)
        _cls.load_state_dict(torch.load(cls_path, map_location=DEVICE))
        _cls.eval()
        
        # Display check
        acc = _info.get('accuracy', 0)
        print(f"  Loaded successfully. Previous Accuracy: {acc*100:.2f}%\n")
        return

    # Train from scratch Mode
    try:
        ae, cls, mse, psnr, acc, f_acc, cm = run_pipeline()
        _ae  = ae
        _cls = cls
        _info = {
            "mse": float(mse), 
            "psnr": float(psnr), 
            "accuracy": float(acc), 
            "f_accuracy": float(f_acc), 
            "cm": cm.tolist() if hasattr(cm, 'tolist') else cm, 
            "vocab": _fruit_vocab
        }
        
        print("\n=== Saving Models for Offline Use ===")
        torch.save(_ae.state_dict(), ae_path)
        torch.save(_cls.state_dict(), cls_path)
        with open(info_path, "w") as f:
            json.dump(_info, f)
        print(f"  Models saved successfully -> {MODELS_DIR}")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")


# ── Inference (one image) ─────────────────────────────────────────────────────

def infer_single(img_rgb_uint8):
    if _ae is None or _cls is None:
        return None, "Models not trained. Run `python prepare_dataset.py` first, then restart."
    if img_rgb_uint8 is None:
        return None, "Please provide an image."
        
    if len(img_rgb_uint8.shape) == 3 and img_rgb_uint8.shape[2] == 4:
        img_rgb_uint8 = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGBA2RGB)
    elif len(img_rgb_uint8.shape) == 2:
        img_rgb_uint8 = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_GRAY2RGB)

    # Preprocess
    img = cv2.resize(img_rgb_uint8, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    orig = img[np.newaxis, ...]                         # (1, 128, 128, 3)

    # Compress
    comp = compress_batch(orig)                          # (1, 32, 32, 3)
    Tc   = to_tensor(comp)                              # (1, 3, 32, 32)

    # Reconstruct
    Tr = predict_batch(_ae, Tc)                         # (1, 3, 128, 128)

    # Classify
    prob_cond, out_fruit = predict_batch_cls(_cls, Tr)
    prob = float(prob_cond[0, 0].item())
    
    condition = "Healthy" if prob < 0.5 else "Spoiled"
    conf  = (1 - prob) * 100 if prob < 0.5 else prob * 100

    mse  = _info.get("mse",  0)
    psnr = _info.get("psnr", 0)
    acc  = _info.get("accuracy", 0)

    result = (
        f"**Condition**: {condition} - {conf:.1f}% Confidence\n\n"
        f"---\n"
        f"**Reconstruction** → PSNR: {psnr:.2f} dB  |  MSE: {mse:.6f}\n\n"
        f"**Classifier Test Accuracy (Condition):** {acc * 100:.2f}%"
    )

    # Reconstructed image -> uint8
    rec_np = Tr[0].permute(1, 2, 0).numpy()
    rec_uint8 = (rec_np * 255).clip(0, 255).astype(np.uint8)

    return rec_uint8, result


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Fruit Quality — CS + DL Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Fruit Quality Classification
            ### Compressed Sensing + Autoencoder Reconstruction + CNN Classification
            Upload a fruit image (or use your webcam) to classify as **Healthy** or **Spoiled**.
            """
        )
        with gr.Row():
            with gr.Column():
                inp     = gr.Image(sources=["upload", "webcam"], type="numpy", label="Input Image")
                run_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                out_img = gr.Image(type="numpy", label="Reconstructed Image (Autoencoder Output)")
                out_txt = gr.Markdown()

        run_btn.click(fn=infer_single, inputs=inp, outputs=[out_img, out_txt])

        gr.Markdown(
            "**Pipeline:** Input -> Compress 32x32 (CS) -> Autoencoder -> Reconstruct 128x128 -> CNN -> Healthy / Spoiled"
        )
    return demo


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Fruit Quality Deep Learning Pipeline (PyTorch) ===")
    print(f"Device: {DEVICE}")
    init()
    build_ui().launch()
