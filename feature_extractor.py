import librosa
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def compute_features_from_waveform(y, sr):
    """
    内部工具函数：从波形数据 y 计算特征
    """
    feature_list = []
    
    # 1. MFCC + Delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feature_list.append(np.mean(mfcc, axis=1))
    feature_list.append(np.std(mfcc, axis=1))
    
    mfcc_delta = librosa.feature.delta(mfcc)
    feature_list.append(np.mean(mfcc_delta, axis=1))
    feature_list.append(np.std(mfcc_delta, axis=1))
    
    # 2. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feature_list.append(np.mean(contrast, axis=1))
    feature_list.append(np.std(contrast, axis=1))
    
    # 3. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feature_list.append(np.mean(chroma, axis=1))
    feature_list.append(np.std(chroma, axis=1))
    
    # 4. ZCR & RMS
    zcr = librosa.feature.zero_crossing_rate(y)
    feature_list.append(np.mean(zcr))
    feature_list.append(np.std(zcr))
    
    rms = librosa.feature.rms(y=y)
    feature_list.append(np.mean(rms))
    feature_list.append(np.std(rms))
    
    return np.hstack(feature_list)

def augment_audio(y, sr):
    """
    物理级数据增强生成器
    返回: [y_noise, y_stretch, y_shift] 增强后的波形列表
    """
    augmented_waveforms = []
    
    # 1. 添加微弱白噪声 (Noise Injection)
    # 幅度极小 (0.005)，避免掩盖枪声细节
    noise_amp = 0.005 * np.max(np.abs(y))
    y_noise = y + noise_amp * np.random.normal(size=len(y))
    augmented_waveforms.append(y_noise)
    
    # 2. 时间伸缩 (Time Stretch)
    # 稍微变快 (1.05倍) 或 变慢 (0.95倍)
    # 随机选一种
    rate = np.random.choice([0.9, 1.1])
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    # 保持长度一致 (裁剪或填充)，方便处理，虽然提取特征不强制要求长度一致，但为了稳定
    if len(y_stretch) > len(y):
        y_stretch = y_stretch[:len(y)]
    else:
        y_stretch = np.pad(y_stretch, (0, max(0, len(y) - len(y_stretch))))
    augmented_waveforms.append(y_stretch)
    
    # 3. 音调微调 (Pitch Shift)
    # 仅偏移 +/- 1 个半音，模拟不同批次枪械的差异
    n_steps = np.random.choice([-1, 1])
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    augmented_waveforms.append(y_shift)
    
    return augmented_waveforms

def extract_features(file_path):
    """
    普通提取：用于推理，只返回原声特征
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        return compute_features_from_waveform(y, sr)
    except Exception as e:
        print(f"❌ 特征提取失败 {file_path}: {e}")
        return None

def extract_features_with_augmentation(file_path):
    """
    训练专用提取：返回 { 'original': feat, 'augmented': [feat1, feat2, feat3] }
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        
        # 1. 计算原声特征
        feat_original = compute_features_from_waveform(y, sr)
        
        # 2. 生成增强波形并计算特征
        aug_waveforms = augment_audio(y, sr)
        feats_augmented = []
        for y_aug in aug_waveforms:
            f = compute_features_from_waveform(y_aug, sr)
            feats_augmented.append(f)
            
        return {
            'original': feat_original,
            'augmented': feats_augmented
        }
    except Exception as e:
        print(f"❌ 增强提取失败 {file_path}: {e}")
        return None
