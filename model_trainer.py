import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 导入增强版提取器
from feature_extractor import extract_features_with_augmentation
from cnn_model import GunshotCNN

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "原始音频文件" / "gun_sound_test"
MODEL_SAVE_PATH = BASE_DIR / "cnn_audio_model.pth"
SCALER_SAVE_PATH = BASE_DIR / "scaler.pkl"
ENCODER_SAVE_PATH = BASE_DIR / "label_encoder.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_filename(filename):
    try:
        if '_' in filename: return filename.split('_')[0]
        else: return filename.split('.')[0]
    except: return None

def train_audio_model():
    print(f"\n🧠 --- PyTorch 深度学习训练 (数据增强版) ---")
    print(f"🖥️  设备: {DEVICE}")
    print("📢 策略: 物理音频增强 (仅训练集) + 类别权重惩罚")
    
    mp3_files = list(AUDIO_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"❌ 错误: {AUDIO_DIR} 为空")
        return

    # --- 1. 预计算所有特征 (原版 + 增强版) ---
    print(f"📂 正在预处理 {len(mp3_files)} 个音频 (生成增强数据)...")
    
    data_cache = []
    labels_for_split = [] 
    
    for file_path in tqdm(mp3_files, desc="物理增强与特征提取"):
        label = parse_filename(file_path.name)
        if not label: continue
        
        result = extract_features_with_augmentation(file_path)
        
        if result is not None:
            data_cache.append({
                'label': label,
                'orig': result['original'],
                'aug': result['augmented']
            })
            labels_for_split.append(label)

    if not data_cache:
        print("❌ 无有效数据")
        return
        
    # --- 2. 标签编码 ---
    le = LabelEncoder()
    y_all_encoded = le.fit_transform(labels_for_split)
    num_classes = len(le.classes_)
    num_samples = len(data_cache)
    
    print(f"✅ 预处理完成。原始样本数: {num_samples}, 类别数: {num_classes}")
    
    # --- 3. 准备归一化 ---
    # 使用所有原版数据拟合 Scaler
    all_orig_feats = np.array([item['orig'] for item in data_cache])
    scaler = StandardScaler()
    scaler.fit(all_orig_feats) 
    
    # 获取特征维度 (修复报错的关键点)
    input_feature_dim = all_orig_feats.shape[1]

    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_all_encoded), y=y_all_encoded)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    # --- 4. K-Fold 交叉验证 ---
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    best_macro_f1 = 0.0
    best_model_state = None
    best_fold_preds = []
    best_fold_targets = []
    
    print(f"🚀 开始 {k_folds} 折交叉验证 (验证集绝不增强)...")
    
    EPOCHS = 35
    BATCH_SIZE = 32
    
    # split 是基于原始文件的索引进行的
    for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(num_samples), y_all_encoded)):
        print(f"\n🔄 Fold {fold+1}/{k_folds}")
        
        # --- 构建训练集 (原版 + 增强版) ---
        X_train_list = []
        y_train_list = []
        
        for idx in train_indices:
            item = data_cache[idx]
            lbl = y_all_encoded[idx]
            
            # 1. 加入原版
            X_train_list.append(item['orig'])
            y_train_list.append(lbl)
            
            # 2. 加入增强版
            for aug_feat in item['aug']:
                X_train_list.append(aug_feat)
                y_train_list.append(lbl)
                
        # --- 构建验证集 (仅原版) ---
        X_val_list = []
        y_val_list = []
        
        for idx in val_indices:
            item = data_cache[idx]
            lbl = y_all_encoded[idx]
            X_val_list.append(item['orig'])
            y_val_list.append(lbl)
            
        # 归一化并转 Tensor
        X_train_np = scaler.transform(np.array(X_train_list))
        X_val_np = scaler.transform(np.array(X_val_list))
        
        X_train_t = torch.FloatTensor(X_train_np)
        y_train_t = torch.LongTensor(np.array(y_train_list))
        X_val_t = torch.FloatTensor(X_val_np)
        y_val_t = torch.LongTensor(np.array(y_val_list))
        
        print(f"   训练样本: {len(X_train_t)} (含增强), 验证样本: {len(X_val_t)} (纯净)")
        
        # DataLoader
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
        
        # 模型初始化
        model = GunshotCNN(num_features=input_feature_dim, num_classes=num_classes).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # 训练循环
        for epoch in range(EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 验证循环
        model.eval()
        fold_preds = []
        fold_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                fold_preds.extend(preds.cpu().numpy())
                fold_targets.extend(batch_y.cpu().numpy())
        
        curr_macro = f1_score(fold_targets, fold_preds, average='macro')
        print(f"   ✅ Fold {fold+1} Macro-F1: {curr_macro:.4f}")
        
        if curr_macro > best_macro_f1:
            best_macro_f1 = curr_macro
            best_model_state = model.state_dict()
            best_fold_preds = fold_preds
            best_fold_targets = fold_targets

    # --- 5. 结果展示 ---
    if not best_fold_targets:
        print("❌ 训练异常，无预测结果")
        return

    print("\n" + "="*50)
    print(f"🏆 增强版最佳模型报告 (Best Macro-F1: {best_macro_f1:.4f})")
    print("="*50)
    
    final_micro = f1_score(best_fold_targets, best_fold_preds, average='micro')
    final_macro = f1_score(best_fold_targets, best_fold_preds, average='macro')
    print(f"🌟 Micro-F1 (总体): {final_micro:.4f}")
    print(f"🌟 Macro-F1 (平均): {final_macro:.4f}")
    
    target_names = le.classes_
    print(classification_report(best_fold_targets, best_fold_preds, target_names=target_names, zero_division=0))
    
    # 绘图
    try:
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(best_fold_targets, best_fold_preds)
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'混淆矩阵 (Micro-F1: {final_micro:.2f})')
        plt.xlabel('预测')
        plt.ylabel('真实')
        
        plt.subplot(1, 2, 2)
        report_dict = classification_report(best_fold_targets, best_fold_preds, target_names=target_names, zero_division=0, output_dict=True)
        class_scores = []
        class_labels = []
        for key, value in report_dict.items():
            if key in ['accuracy', 'macro avg', 'weighted avg']: continue
            class_labels.append(key)
            class_scores.append(value['f1-score'])
            
        sorted_indices = np.argsort(class_scores)[::-1]
        sns.barplot(x=np.array(class_labels)[sorted_indices], y=np.array(class_scores)[sorted_indices], palette="viridis")
        plt.title('各武器 F1-Score 排名 (数据增强版)')
        plt.xticks(rotation=90)
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")

    # --- 6. 保存模型 ---
    # 修复点：这里原来是 X_tensor.shape[1]，现在改成 input_feature_dim
    final_save = {
        'state_dict': best_model_state,
        'num_features': input_feature_dim, 
        'num_classes': num_classes,
        'classes': le.classes_
    }
    
    torch.save(final_save, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(le, ENCODER_SAVE_PATH)
    
    print(f"\n💾 模型保存成功: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()
