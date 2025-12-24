import torch
import joblib
import numpy as np
from pathlib import Path
from feature_extractor import extract_features
from cnn_model import GunshotCNN

# 路径配置
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn_audio_model.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_user_audio():
    print("\n🧠 --- CNN 武器声音识别 ---")
    
    # 1. 检查模型是否存在
    if not MODEL_PATH.exists():
        print("❌ 错误：模型未训练，请管理员运行训练模块。")
        return

    # 2. 获取文件名
    filename = input("请输入 MP3 文件名 (如 test.mp3): ").strip()
    file_path = BASE_DIR / filename
    
    if not file_path.exists():
        print("❌ 文件不存在")
        return

    print("🔍 提取深度特征...")
    features = extract_features(file_path)
    if features is None: return

    try:
        # 3. 加载工具
        scaler = joblib.load(SCALER_PATH)
        
        # ==================================================
        # 👇 核心修复点：添加 weights_only=False
        # ==================================================
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # 4. 预处理
        # 归一化 (reshape成 1, N)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
        
        # 5. 初始化并加载模型
        num_features = checkpoint['num_features']
        num_classes = checkpoint['num_classes']
        classes = checkpoint['classes']
        
        model = GunshotCNN(num_features, num_classes).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # 6. 推理
        with torch.no_grad():
            outputs = model(features_tensor)
            # Softmax 获取概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            label = classes[pred_idx.item()]
            conf_val = confidence.item() * 100
        
        print(f"\n🎯 识别结果: [ {label.upper()} ]")
        print(f"📊 置信度:   {conf_val:.2f}%")
        
        # 如果置信度低，给个提示
        if conf_val < 50:
            print("⚠️ 提示: 置信度较低，可能是环境噪音或未知枪声。")
            
    except Exception as e:
        print(f"❌ 推理错误: {e}")
        # 打印详细错误方便调试
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_user_audio()
