import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import joblib
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# 👇 核心修正：从 sklearn.metrics 导入，而不是 sklearn.base
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# 导入 FusionCRNN 模型
from distance_model import FusionCRNN

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "原始音频文件" / "gun_sound_test"
MODEL_SAVE_PATH = BASE_DIR / "fusion_distance_model.pth"
SCALER_SAVE_PATH = BASE_DIR / "fusion_scaler.pkl"
ENCODER_SAVE_PATH = BASE_DIR / "distance_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_distance(filename):
    try:
        parts = filename.split('_')
        for p in parts:
            if p.endswith('m') and p[:-1].isdigit(): return p
        if len(parts) > 1:
             val = parts[1].replace('m', '')
             if val.isdigit(): return parts[1]
        return None
    except: return None

def extract_dual_features(file_path, augment=False):
    """
    双流特征提取：
    1. Log-Mel (2D 图片)
    2. Statistical Features (1D 向量)
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=2.0)
        target_len = int(22050 * 2.0)
        if len(y) < target_len: y = np.pad(y, (0, target_len - len(y)))
        else: y = y[:target_len]
        
        if augment:
            gain = np.random.uniform(0.5, 1.5)
            y = y * gain
            noise = np.random.normal(0, 0.002, len(y))
            y = y + noise

        # --- Stream 1: Log-Mel Spectrogram ---
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel + 80) / 80.0 # 归一化到 0-1
        
        # --- Stream 2: High-Level Statistics ---
        hop_length = 512
        rmse = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=2048, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop_length)
        
        # 拼接: (4, Time)
        stats = np.vstack([rmse, cent, rolloff, zcr])
        
        # 对齐长度
        min_len = min(log_mel.shape[1], stats.shape[1])
        log_mel = log_mel[:, :min_len]
        stats = stats[:, :min_len]
        
        return log_mel, stats
    except: return None, None

def train_distance_model():
    print(f"\n📏 --- 双流融合距离模型训练 (Fusion CRNN) ---")
    
    mp3_files = list(AUDIO_DIR.glob("*.mp3"))
    if not mp3_files: 
        print("❌ 目录为空")
        return

    X_mel_list = []
    X_stat_list = []
    y_raw = []
    
    for f in tqdm(mp3_files, desc="提取双模特征"):
        lbl = parse_distance(f.name)
        if not lbl: continue
        
        # 提取 2 次增强
        for aug in [False, True]:
            m, s = extract_dual_features(f, augment=aug)
            if m is not None:
                X_mel_list.append(m)
                X_stat_list.append(s)
                y_raw.append(lbl)
    
    if not X_mel_list: 
        print("❌ 无有效数据")
        return
    
    # 堆叠数据
    X_mel = np.array(X_mel_list)[:, np.newaxis, :, :] # (N, 1, 128, T)
    X_stat = np.array(X_stat_list) # (N, 4, T)
    
    # --- 统计特征单独归一化 ---
    N, C, T = X_stat.shape
    X_stat_reshaped = X_stat.transpose(0, 2, 1).reshape(-1, C)
    scaler = StandardScaler()
    X_stat_norm = scaler.fit_transform(X_stat_reshaped)
    X_stat = X_stat_norm.reshape(N, T, C).transpose(0, 2, 1) # 还原
    
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # --- 标签处理 ---
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"🔍 识别类别: {le.classes_}")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    # --- 数据划分 ---
    indices = np.arange(len(y_enc))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_enc)
    
    # 构建 Tensor
    mel_train = torch.FloatTensor(X_mel[train_idx]).to(DEVICE)
    stat_train = torch.FloatTensor(X_stat[train_idx]).to(DEVICE)
    y_train = torch.LongTensor(y_enc[train_idx]).to(DEVICE)
    
    mel_val = torch.FloatTensor(X_mel[val_idx]).to(DEVICE)
    stat_val = torch.FloatTensor(X_stat[val_idx]).to(DEVICE)
    y_val = torch.LongTensor(y_enc[val_idx]).to(DEVICE)
    
    # --- 训练 ---
    model = FusionCRNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # 自定义 Dataset 支持双输入
    class DualDataset(torch.utils.data.Dataset):
        def __init__(self, m, s, y): self.m, self.s, self.y = m, s, y
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.m[i], self.s[i], self.y[i]
        
    train_loader = DataLoader(DualDataset(mel_train, stat_train, y_train), batch_size=32, shuffle=True)
    
    print("🚀 开始双流训练...")
    best_acc = 0.0
    
    for epoch in range(60):
        model.train()
        for bm, bs, by in train_loader:
            optimizer.zero_grad()
            out = model(bm, bs) # 输入两个特征
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(mel_val, stat_val)
                _, preds = torch.max(val_out, 1)
                acc = accuracy_score(y_val.cpu(), preds.cpu())
                print(f"   Epoch {epoch+1} | Val Acc: {acc:.2%}")
                
                if acc > best_acc:
                    best_acc = acc
                    torch.save({
                        'state_dict': model.state_dict(),
                        'num_classes': num_classes,
                        'classes': le.classes_
                    }, MODEL_SAVE_PATH)
    
    joblib.dump(le, ENCODER_SAVE_PATH)
    print(f"\n💾 最佳模型已保存 (Acc: {best_acc:.2%})")

if __name__ == "__main__":
    train_distance_model()
