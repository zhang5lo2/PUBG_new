import torch
import joblib
import numpy as np
import os
from pathlib import Path

# === 导入特征提取器 ===
from feature_extractor import extract_features  # 枪声分类用
from distance_trainer import extract_dual_features # 距离分类用 (新)

# === 导入模型架构 ===
from cnn_model import GunshotCNN
from distance_model import FusionCRNN  # 双流模型

# === 路径配置 ===
BASE_DIR = Path(__file__).resolve().parent

CLS_MODEL_PATH = BASE_DIR / "cnn_audio_model.pth"
CLS_SCALER_PATH = BASE_DIR / "scaler.pkl"

DIST_MODEL_PATH = BASE_DIR / "fusion_distance_model.pth" # 注意文件名变了
DIST_SCALER_PATH = BASE_DIR / "fusion_scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_user_audio():
    print("\n🎧 --- PUBG 战术音频分析终端 (Fusion Engine) ---")
    
    # 1. 基础检查
    if not CLS_MODEL_PATH.exists():
        print("❌ 错误：分类模型未训练。")
        return
    
    # 2. 获取输入
    filename = input("请输入 MP3 文件名 (如 test.mp3): ").strip()
    file_path = BASE_DIR / filename
    
    if not file_path.exists():
        print("❌ 文件不存在")
        return

    # ==========================
    # 🕵️‍♂️ Task 1: 枪声分类 (单流 1D-CNN)
    # ==========================
    print("\n🔍 [Task 1] 解析声纹特征 (MFCC+Delta)...")
    predicted_label = "Unknown"
    confidence = 0.0
    
    try:
        cls_feats = extract_features(file_path)
        if cls_feats is not None:
            scaler = joblib.load(CLS_SCALER_PATH)
            ckpt = torch.load(CLS_MODEL_PATH, map_location=DEVICE, weights_only=False)
            
            inp = torch.FloatTensor(scaler.transform(cls_feats.reshape(1, -1))).to(DEVICE)
            model_cls = GunshotCNN(ckpt['num_features'], ckpt['num_classes']).to(DEVICE)
            model_cls.load_state_dict(ckpt['state_dict'])
            model_cls.eval()
            
            with torch.no_grad():
                out = model_cls(inp)
                probs = torch.nn.functional.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                predicted_label = ckpt['classes'][idx.item()]
                confidence = conf.item() * 100
        else:
            print("❌ 分类特征提取失败")
    except Exception as e:
        print(f"❌ 分类推理出错: {e}")

    # ==========================
    # 📏 Task 2: 距离分类 (双流 FusionCRNN)
    # ==========================
    print("🔍 [Task 2] 解析时频与物理特征 (Log-Mel + Stats)...")
    dist_label = "Unknown"
    
    if DIST_MODEL_PATH.exists():
        try:
            # 提取双流特征
            mel, stat = extract_dual_features(file_path, augment=False)
            
            if mel is not None and stat is not None:
                # 1. Stat 特征归一化
                stat_scaler = joblib.load(DIST_SCALER_PATH)
                stat_norm = stat_scaler.transform(stat.T).T # (4, T)
                
                # 2. 转 Tensor
                t_mel = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, 128, T)
                t_stat = torch.FloatTensor(stat_norm).unsqueeze(0).to(DEVICE)       # (1, 4, T)
                
                # 3. 加载双流模型
                ckpt = torch.load(DIST_MODEL_PATH, map_location=DEVICE, weights_only=False)
                model_dist = FusionCRNN(num_classes=ckpt['num_classes']).to(DEVICE)
                model_dist.load_state_dict(ckpt['state_dict'])
                model_dist.eval()
                
                with torch.no_grad():
                    out = model_dist(t_mel, t_stat)
                    probs = torch.nn.functional.softmax(out, dim=1)
                    _, idx = torch.max(probs, 1)
                    dist_label = ckpt['classes'][idx.item()]
                    
        except Exception as e:
            print(f"⚠️ 测距失败: {e}")
    else:
        print("⚠️ 距离模型未训练")

    # ==========================
    # 📊 报告
    # ==========================
    print("\n" + "="*40)
    print(f"🎯 战术分析报告")
    print("="*40)
    print(f"🔫 武器类型:  [ {predicted_label.upper()} ]")
    print(f"📈 识别置信度: {confidence:.2f}%")
    print(f"📏 目标距离:  [ {dist_label} ]")
    
    # 战术建议
    d_str = str(dist_label)
    if "0m" in d_str or "10m" in d_str or "20m" in d_str:
        print("🚨 建议: 贴脸战斗！(CQB)")
    elif "50m" in d_str or "100m" in d_str:
        print("⚠️ 建议: 中近距离交火。")
    elif "200m" in d_str or "400m" in d_str:
        print("✅ 建议: 远距离狙击。")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    predict_user_audio()
